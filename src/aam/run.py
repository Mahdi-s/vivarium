from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sqlite3
import sys
import time
import uuid
from importlib import metadata as importlib_metadata
from pathlib import Path
from shutil import copy2
from subprocess import DEVNULL, CalledProcessError, check_output

from aam.agent_langgraph import default_cognitive_policy
from aam.channel import InMemoryChannel
from aam.export import export_messages_to_parquet, export_trace_to_parquet
from aam.experiment_config import load_experiment_config
from aam.interpretability import CaptureConfig, CaptureContext
from aam.llama_cpp import LlamaServerConfig, run_llama_server
from aam.llm_gateway import LiteLLMGateway, MockLLMGateway, RateLimitConfig, TransformerLensGateway
from aam.model_discovery import discover_all_models, export_models
from aam.persistence import TraceDb, TraceDbConfig
from aam.policy import RandomAgentPolicy, stable_agent_seed
from aam.scheduler import BarrierScheduler, BarrierSchedulerConfig
from aam.types import RunMetadata
from aam.world_engine import WorldEngine, WorldEngineConfig
from aam.experiments.olmo_conformity.runner import run_suite as run_olmo_conformity_suite
from aam.experiments.olmo_conformity.probes import (
    ProbeCaptureSpec,
    capture_probe_dataset_to_db,
    train_probe_from_captured_activations,
    compute_and_store_probe_projections_for_trials,
)
from aam.experiments.olmo_conformity.analysis import generate_core_figures
from aam.experiments.olmo_conformity.logit_lens import compute_logit_lens_topk_for_trial, parse_and_store_think_tokens
from aam.experiments.olmo_conformity.intervention import run_intervention_sweep


def _validate_db(*, db_path: str, run_id: str, steps: int, num_agents: int) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute("SELECT 1 FROM runs WHERE run_id = ?;", (run_id,))
        if cur.fetchone() is None:
            raise RuntimeError(f"DB validation failed: missing runs row for run_id={run_id}")

        expected = steps * num_agents
        (count,) = conn.execute("SELECT COUNT(*) FROM trace WHERE run_id = ?;", (run_id,)).fetchone()
        if count != expected:
            raise RuntimeError(f"DB validation failed: expected {expected} trace rows, found {count}")

        (min_step, max_step) = conn.execute(
            "SELECT MIN(time_step), MAX(time_step) FROM trace WHERE run_id = ?;",
            (run_id,),
        ).fetchone()
        if steps == 0:
            if min_step is not None or max_step is not None:
                raise RuntimeError("DB validation failed: expected no trace rows for steps=0")
        else:
            if min_step != 0 or max_step != steps - 1:
                raise RuntimeError(
                    f"DB validation failed: expected time_step range 0..{steps-1}, found {min_step}..{max_step}"
                )
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    # Load .env file if python-dotenv is available
    # This allows users to configure environment variables (AAM_MODEL_DIR, etc.)
    # without modifying their shell profile.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed, skip
    
    p = argparse.ArgumentParser(prog="vvm", description="Vivarium")
    sub = p.add_subparsers(dest="mode")

    # Phase 1 (backwards compatible): random agents producing trace rows
    p1 = sub.add_parser("phase1", help="Phase 1 core simulation (RandomAgent)")
    p1.add_argument("--steps", type=int, default=100)
    p1.add_argument("--agents", type=int, default=5)
    p1.add_argument("--seed", type=int, default=42)
    p1.add_argument("--db", type=str, default="./simulation.db")
    p1.add_argument("--run-id", type=str, default=None)
    p1.add_argument("--no-validate", action="store_true")
    p1.add_argument("--nondeterministic-timestamps", action="store_true")

    # Phase 2: cognitive agents + basic tools via LiteLLM (or mock)
    p2 = sub.add_parser("phase2", help="Phase 2 cognitive simulation (LangGraph + LiteLLM)")
    p2.add_argument("--steps", type=int, default=10)
    p2.add_argument("--agents", type=int, default=2)
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--db", type=str, default="./simulation_cognitive.db")
    p2.add_argument("--run-id", type=str, default=None)
    p2.add_argument("--model", type=str, default="gpt-3.5-turbo")
    p2.add_argument("--mock-llm", action="store_true", help="Use deterministic offline mock LLM")
    p2.add_argument("--api-base", type=str, default=None, help="Override OpenAI-compatible API base (e.g. llama-server /v1)")
    p2.add_argument("--api-key", type=str, default=None, help="API key for the provider (optional for local servers)")
    p2.add_argument("--no-validate", action="store_true")
    p2.add_argument("--nondeterministic-timestamps", action="store_true")
    p2.add_argument("--message-history", type=int, default=20)
    p2.add_argument("--rate-limit-rpm", type=int, default=None, help="Rate limit: requests per minute")
    p2.add_argument("--rate-limit-tpm", type=int, default=None, help="Rate limit: tokens per minute")
    p2.add_argument("--rate-limit-max-concurrent", type=int, default=10, help="Rate limit: max concurrent requests")
    p2.add_argument("--no-rate-limit", action="store_true", help="Disable rate limiting")
    p2.add_argument("--export-parquet", action="store_true", help="Export trace and messages to Parquet format")

    # Phase 3: local TransformerLens model + activation capture aligned to trace
    p3 = sub.add_parser("phase3", help="Phase 3 interpretability run (TransformerLens + Safetensors)")
    p3.add_argument("--steps", type=int, default=10)
    p3.add_argument("--agents", type=int, default=2)
    p3.add_argument("--seed", type=int, default=42)
    p3.add_argument("--run-id", type=str, default=None)
    p3.add_argument("--model-id", type=str, required=True, help="HuggingFace model id for TransformerLens")
    p3.add_argument(
        "--runs-dir",
        type=str,
        default="./runs",
        help="Base directory for PRD-style outputs: runs/<timestamp>_<run_id>/",
    )
    p3.add_argument("--no-validate", action="store_true")
    p3.add_argument("--nondeterministic-timestamps", action="store_true")
    p3.add_argument("--message-history", type=int, default=20)
    p3.add_argument("--list-hooks", action="store_true", help="Print available TransformerLens hook names and exit")
    p3.add_argument("--hooks-head", type=int, default=200, help="When listing hooks, print first N hook names")
    p3.add_argument("--layers", type=str, default="0", help="Comma-separated layer indices, e.g. '0,1,2'")
    p3.add_argument(
        "--components",
        type=str,
        default="resid_post",
        help="Comma-separated component/hook suffixes, e.g. 'resid_post,attn.hook_z'",
    )
    p3.add_argument(
        "--trigger-actions",
        type=str,
        default="post_message",
        help="Comma-separated action_names to persist activations for (sparse sampling)",
    )
    p3.add_argument("--token-position", type=int, default=-1, help="Token position to slice (-1=last)")
    p3.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])

    # Phase 4: Experiment runner (Barrier Scheduler + config file)
    p4 = sub.add_parser("experiment", help="Phase 4 experiment runner (Barrier Scheduler + JSON config)")
    p4.add_argument("--config", type=str, required=True, help="Path to experiment JSON config")
    p4.add_argument("--run-id", type=str, default=None)
    p4.add_argument("--runs-dir", type=str, default=None, help="Override base runs directory (default from config)")
    p4.add_argument("--no-validate", action="store_true")
    p4.add_argument("--export-parquet", action="store_true", help="Export trace and messages to Parquet format")

    # Olmo Conformity suite runner (data capture focused; uses conformity_* tables)
    pc = sub.add_parser("olmo-conformity", help="Run Olmo conformity experiment suite (Synthetic Asch + logging)")
    pc.add_argument("--suite-config", type=str, required=True, help="Path to suite config JSON (experiments/olmo_conformity/configs/...)")
    pc.add_argument(
        "--runs-dir",
        type=str,
        default="./runs",
        help="Base directory for outputs: runs/<timestamp>_<run_id>/",
    )
    pc.add_argument("--run-id", type=str, default=None)
    pc.add_argument("--api-base", type=str, default=None, help="Override OpenAI-compatible API base")
    pc.add_argument("--api-key", type=str, default=None, help="API key for provider (optional for local servers)")
    pc.add_argument("--rate-limit-rpm", type=int, default=None, help="Rate limit: requests per minute")
    pc.add_argument("--rate-limit-tpm", type=int, default=None, help="Rate limit: tokens per minute")
    pc.add_argument("--rate-limit-max-concurrent", type=int, default=10, help="Rate limit: max concurrent requests")
    pc.add_argument("--no-rate-limit", action="store_true", help="Disable rate limiting")
    pc.add_argument("--capture-activations", action="store_true", help="Capture activations during trials (requires TransformerLens models)")
    pc.add_argument("--capture-layers", type=str, default=None, help="Comma-separated layer indices for activation capture (e.g. '10,11,12')")
    pc.add_argument("--capture-components", type=str, default=None, help="Comma-separated component names (e.g. 'resid_post')")
    pc.add_argument("--capture-dtype", type=str, default="float16", choices=["float16", "float32"], help="Dtype for activation tensors")
    pc.add_argument("--use-judgeval", action="store_true", help="Enable Judge Eval evaluation during trials")
    pc.add_argument("--judgeval-judge-model", type=str, default="gpt-oss:20b", help="Ollama model to use as judge")
    pc.add_argument("--judgeval-ollama-base", type=str, default="http://localhost:11434/v1", help="Ollama API base URL")

    # Olmo Conformity probe utilities (TransformerLens capture + probe training)
    pp = sub.add_parser("olmo-conformity-probe", help="Capture activations for probe dataset, train probe, and compute projections")
    pp.add_argument("--run-id", type=str, required=True, help="Existing run_id in runs/<ts>_<run_id>/simulation.db")
    pp.add_argument("--db", type=str, required=True, help="Path to simulation.db for the run")
    pp.add_argument("--model-id", type=str, required=True, help="HuggingFace model id for TransformerLens")
    pp.add_argument("--dataset-path", type=str, required=True, help="Path to labeled JSONL (e.g. experiments/.../truth_probe_train.jsonl)")
    pp.add_argument("--dataset-name", type=str, default="truth_probe_train")
    pp.add_argument("--dataset-version", type=str, default="v0")
    pp.add_argument("--probe-kind", type=str, default="truth", help="Probe kind label (truth/social)")
    pp.add_argument("--layers", type=str, default="0", help="Comma-separated layer indices, e.g. '0,1,2'")
    pp.add_argument("--component", type=str, default="hook_resid_post", help="TransformerLens hook component under blocks.<L>., e.g. hook_resid_post")
    pp.add_argument("--token-position", type=int, default=-1)
    pp.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    pp.add_argument("--temperature", type=float, default=0.0, help="Temperature for activation capture")

    pr = sub.add_parser("olmo-conformity-report", help="Generate figures/tables from conformity_* tables for a run")
    pr.add_argument("--run-id", type=str, required=True)
    pr.add_argument("--db", type=str, required=True, help="Path to simulation.db for the run")
    pr.add_argument("--run-dir", type=str, required=True, help="Path to run directory (writes artifacts/)")

    pj = sub.add_parser(
        "olmo-conformity-judgeval",
        help="Backfill Judge Eval scores into conformity_outputs.parsed_answer_json for an existing run",
    )
    pj.add_argument("--run-id", type=str, required=True)
    pj.add_argument("--db", type=str, required=True, help="Path to simulation.db for the run")
    pj.add_argument("--judge-model", type=str, default="gpt-oss:20b", help="Ollama model to use as judge")
    pj.add_argument("--ollama-base", type=str, default="http://localhost:11434/v1", help="Ollama API base URL")
    pj.add_argument("--force", action="store_true", help="Overwrite existing parsed_answer_json if present")
    pj.add_argument("--limit", type=int, default=None, help="Optional cap on number of trials to score")
    pj.add_argument("--max-concurrency", type=int, default=4, help="Max concurrent judge requests (default: 4)")
    pj.add_argument(
        "--trial-scope",
        type=str,
        default="behavioral-only",
        choices=["behavioral-only", "all"],
        help="Which trials to score (default: behavioral-only).",
    )

    pl = sub.add_parser("olmo-conformity-logit-lens", help="Compute logit-lens top-k across layers for each trial")
    pl.add_argument("--run-id", type=str, required=True)
    pl.add_argument("--db", type=str, required=True)
    pl.add_argument("--model-id", type=str, required=True)
    pl.add_argument("--layers", type=str, default="0", help="Comma-separated layer indices")
    pl.add_argument("--topk", type=int, default=10)
    pl.add_argument("--parse-think", action="store_true", help="Also parse <think>...</think> into conformity_think_tokens")
    pl.add_argument("--analyze-think", action="store_true", help="Also compute logit lens for intermediate <think> tokens")
    pl.add_argument(
        "--trial-scope",
        type=str,
        default="all",
        choices=["all", "behavioral-only"],
        help="Which trials to process (default: all trials in run).",
    )

    pi = sub.add_parser("olmo-conformity-intervene", help="Run social-vector subtraction intervention sweep (TransformerLens)")
    pi.add_argument("--run-id", type=str, required=True)
    pi.add_argument("--db", type=str, required=True)
    pi.add_argument("--model-id", type=str, required=True)
    pi.add_argument("--probe-path", type=str, required=True, help="Path to social probe safetensors (layer_*.weight)")
    pi.add_argument("--social-probe-id", type=str, required=True, help="conformity_probes.probe_id for the social vector")
    pi.add_argument("--layers", type=str, default="0", help="Comma-separated target layers")
    pi.add_argument("--alpha", type=str, default="1.0", help="Comma-separated alpha values, e.g. '0.5,1.0,2.0'")
    pi.add_argument("--component-hook", type=str, default="hook_resid_post")
    pi.add_argument("--max-new-tokens", type=int, default=64)

    pv = sub.add_parser("olmo-conformity-vector-analysis", help="Run Truth vs Social Vector analysis workflow")
    pv.add_argument("--run-id", type=str, required=True)
    pv.add_argument("--db", type=str, required=True)
    pv.add_argument("--model-id", type=str, required=True)
    pv.add_argument("--truth-probe-dataset", type=str, required=True, help="Path to truth probe training dataset JSONL")
    pv.add_argument("--social-probe-dataset", type=str, default=None, help="Path to social probe training dataset JSONL (optional)")
    pv.add_argument("--layers", type=str, default="10,11,12,13,14,15,16,17,18,19,20", help="Comma-separated layer indices")
    pv.add_argument("--component", type=str, default="hook_resid_post")
    pv.add_argument("--token-position", type=int, default=-1)
    pv.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    pv.add_argument("--artifacts-dir", type=str, required=True, help="Directory for probe artifacts and plots")

    prr = sub.add_parser(
        "olmo-conformity-resume",
        help="Resume an existing run from the projection step (optionally repairing overwritten trial activations)",
    )
    prr.add_argument("--run-id", type=str, required=True)
    prr.add_argument("--db", type=str, required=True, help="Path to simulation.db for the run")
    prr.add_argument("--model-id", type=str, required=True)
    prr.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to run directory (defaults to dirname(--db))",
    )
    default_layers_32 = ",".join(str(i) for i in range(32))
    prr.add_argument("--layers", type=str, default=default_layers_32, help="Comma-separated layer indices")
    prr.add_argument("--component", type=str, default="hook_resid_post")
    prr.add_argument("--max-new-tokens", type=int, default=128)
    prr.add_argument("--no-repair-activations", action="store_true", help="Skip trial activation repair step")

    pph = sub.add_parser(
        "olmo-conformity-posthoc",
        help="Backfill missing analyses for an existing run (logit lens + think parsing + interventions + report refresh)",
    )
    pph.add_argument("--run-dir", type=str, required=True, help="Path to run directory: runs/<timestamp>_<run_id>/")
    pph.add_argument("--db", type=str, default=None, help="Path to simulation.db (defaults to <run-dir>/simulation.db)")
    pph.add_argument("--run-id", type=str, default=None, help="Run UUID (defaults to derived from run-dir name)")
    pph.add_argument("--model-id", type=str, default=None, help="Model id (defaults to first trial's model_id)")
    pph.add_argument("--layers", type=str, default=default_layers_32, help="Comma-separated layer indices")
    pph.add_argument("--logit-lens-k", type=int, default=10)
    pph.add_argument("--trial-scope", type=str, default="behavioral-only", choices=["all", "behavioral-only"])
    pph.add_argument("--parse-think-tokens", action="store_true", help="Parse <think>...</think> blocks into conformity_think_tokens")
    pph.add_argument("--no-logit-lens", action="store_true", help="Skip logit lens computation")
    pph.add_argument("--no-interventions", action="store_true", help="Skip interventions")
    pph.add_argument("--no-report", action="store_true", help="Skip report regeneration (figures/tables)")
    pph.add_argument(
        "--intervention-scope",
        type=str,
        default="pressure-only",
        choices=["pressure-only", "all-immutable"],
        help="Which trials to run interventions on.",
    )
    pph.add_argument("--intervention-layers", type=str, default="15,16,17,18,19,20")
    pph.add_argument("--alphas", type=str, default="0.5,1.0,2.0")
    pph.add_argument("--component-hook", type=str, default="hook_resid_post")
    pph.add_argument("--max-new-tokens", type=int, default=64)
    pph.add_argument("--clear-existing", action="store_true", help="Delete existing posthoc rows for this run and recompute")

    pe = sub.add_parser("olmo-conformity-full", help="Run complete experiment workflow (trials → probes → interventions → analysis)")
    pe.add_argument("--suite-config", type=str, required=True, help="Path to suite config JSON")
    pe.add_argument("--runs-dir", type=str, default="./runs", help="Base directory for outputs")
    pe.add_argument("--run-id", type=str, default=None)
    pe.add_argument("--api-base", type=str, default=None)
    pe.add_argument("--api-key", type=str, default=None)
    pe.add_argument("--no-rate-limit", action="store_true")
    pe.add_argument("--capture-activations", action="store_true")
    pe.add_argument("--capture-layers", type=str, default=None)
    pe.add_argument("--truth-probe-dataset", type=str, default=None)
    pe.add_argument("--social-probe-dataset", type=str, default=None)
    pe.add_argument("--probe-layers", type=str, default=default_layers_32)
    pe.add_argument("--run-interventions", action="store_true")
    pe.add_argument("--intervention-layers", type=str, default="15,16,17,18,19,20")
    pe.add_argument("--intervention-alphas", type=str, default="0.5,1.0,2.0")
    pe.add_argument("--social-probe-path", type=str, default=None)
    pe.add_argument("--social-probe-id", type=str, default=None)
    pe.add_argument("--no-reports", action="store_true")
    pe.add_argument("--run-vector-analysis", action="store_true")

    # llama: Model management commands
    llama_parser = sub.add_parser("llama", help="llama.cpp model management")
    llama_sub = llama_parser.add_subparsers(dest="llama_command", required=True)

    llama_list = llama_sub.add_parser("list", help="List discovered GGUF models")
    llama_list.add_argument("--export-dir", type=str, default=None, help="Also export models to this directory")

    llama_export = llama_sub.add_parser("export", help="Export discovered models to models/ directory")
    llama_export.add_argument("--export-dir", type=str, default="./models", help="Directory to export models to")
    llama_export.add_argument("--mode", type=str, default="symlink", choices=["symlink", "copy"], help="Export mode")

    llama_serve = llama_sub.add_parser("serve", help="Serve a GGUF model via llama.cpp server")
    llama_serve.add_argument("model_path", type=str, help="Path to GGUF model file")
    llama_serve.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    llama_serve.add_argument("--port", type=int, default=8081, help="Server port")
    llama_serve.add_argument("--ctx-size", type=int, default=4096, help="Context size")
    llama_serve.add_argument(
        "--n-gpu-layers",
        type=int,
        default=None,
        help="Number of GPU layers (-1 for all layers, 0 for CPU-only). Default: auto-detect (all layers on macOS Apple Silicon, CPU-only otherwise)",
    )

    # list-layers: List available layers for a TransformerLens model
    list_layers = sub.add_parser("list-layers", help="List available layers and components for a TransformerLens model")
    list_layers.add_argument("--model-id", type=str, required=True, help="HuggingFace model ID for TransformerLens")
    list_layers.add_argument("--format", type=str, default="text", choices=["text", "json"], help="Output format")

    args = p.parse_args(argv)
    # Backwards compatibility: no subcommand => behave like phase1.
    mode = args.mode or "phase1"

    # Handle list-layers command
    if mode == "list-layers":
        try:
            from transformer_lens import HookedTransformer  # type: ignore

            model = HookedTransformer.from_pretrained(args.model_id)
            layer_info = CaptureContext.get_model_layers(model)

            if args.format == "json":
                print(json.dumps(layer_info, indent=2))
            else:
                print(f"Model: {args.model_id}")
                print(f"Total Layers: {layer_info['num_layers']}")
                print(f"\nLayer Names:")
                for layer_name in layer_info["layer_names"]:
                    print(f"  - {layer_name}")
                print(f"\nComponents by Layer:")
                for layer_idx, components in sorted(layer_info["components"].items()):
                    print(f"  Layer {layer_idx}:")
                    for comp in components:
                        print(f"    - {comp}")

            return 0
        except ImportError:
            print("Error: transformer-lens is not installed. Install extras: `pip install -e .[interpretability]`")
            return 1
        except Exception as e:
            print(f"Error loading model: {e}")
            return 1

    # Handle llama subcommands
    if mode == "llama":
        if args.llama_command == "list":
            models = discover_all_models()
            if not models:
                print("No GGUF models discovered.")
                print("Check ~/.ollama/models/ or ~/Library/Application Support/LM Studio/models/")
                return 0

            print(f"Discovered {len(models)} model(s):\n")
            for m in models:
                size_mb = m.size_bytes / (1024 * 1024)
                print(f"  {m.source}: {m.model_name}")
                print(f"    Path: {m.gguf_path}")
                print(f"    Size: {size_mb:.1f} MB\n")

            if args.export_dir:
                results = export_models(models=models, export_dir=args.export_dir, mode="symlink")
                print(f"\nExported {len(results)} model(s) to {args.export_dir}")
                for r in results:
                    print(f"  {r['model_name']}: {r['status']} -> {r['path']}")

            return 0

        elif args.llama_command == "export":
            models = discover_all_models()
            if not models:
                print("No GGUF models discovered.")
                return 1

            results = export_models(models=models, export_dir=args.export_dir, mode=args.mode)
            print(f"Exported {len(results)} model(s) to {args.export_dir}:")
            for r in results:
                print(f"  {r['model_name']}: {r['status']} -> {r['path']}")

            return 0

        elif args.llama_command == "serve":
            # Use provided value or None (which will trigger platform-specific default)
            n_gpu_layers = args.n_gpu_layers if args.n_gpu_layers is not None else None
            config = LlamaServerConfig(
                model_path=args.model_path,
                host=args.host,
                port=args.port,
                ctx_size=args.ctx_size,
                n_gpu_layers=n_gpu_layers,
            )
            print(f"Starting llama.cpp server on {config.host}:{config.port}")
            print(f"Model: {config.model_path}")
            print("Press Ctrl+C to stop\n")

            try:
                process = run_llama_server(config)
                # Stream output
                for line in process.stdout:
                    print(line.rstrip())
                    if "Uvicorn running on" in line or "listening on" in line.lower():
                        # Server is ready
                        pass
            except KeyboardInterrupt:
                print("\nStopping server...")
                process.terminate()
                process.wait()
                print("Server stopped.")
            except Exception as e:
                print(f"Error: {e}")
                return 1

            return 0

    # Special handling for experiment: it has its own config-based runner and artifact layout.
    if mode == "experiment":
        cfg = load_experiment_config(str(args.config))
        run_id = str(args.run_id or cfg.run.run_id or str(uuid.uuid4()))
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        runs_dir = str(args.runs_dir or cfg.run.runs_dir)
        run_dir = os.path.join(runs_dir, f"{ts}_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        db_path = os.path.join(run_dir, "simulation.db")

        activations_dir = os.path.join(run_dir, "activations")
        capture_context = None

        # Snapshot run metadata (best-effort, offline-friendly).
        meta_path = os.path.join(run_dir, "run_metadata.json")
        git_hash = None
        try:
            git_hash = check_output(["git", "rev-parse", "HEAD"], stderr=DEVNULL).decode("utf-8").strip()
        except (OSError, CalledProcessError):
            git_hash = None
        deps = sorted(
            [{"name": d.metadata["Name"], "version": d.version} for d in importlib_metadata.distributions()],
            key=lambda x: (x["name"] or "").lower(),
        )
        run_metadata = {
            "run_id": run_id,
            "created_at": time.time(),
            "argv": list(argv or sys.argv[1:]),
            "git_hash": git_hash,
            "python": {"version": sys.version},
            "config": cfg.model_dump(mode="json"),
            "dependencies": deps,
        }
        Path(meta_path).write_text(
            __import__("json").dumps(run_metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        # Copy the config file into the run dir for reproducibility.
        try:
            copy2(str(args.config), os.path.join(run_dir, "experiment_config.json"))
        except OSError:
            pass

        trace_db = TraceDb(TraceDbConfig(db_path=db_path))
        trace_db.connect()
        trace_db.init_schema()

        # CaptureContext requires a live TraceDb for activation_metadata indexing.
        if cfg.policy.kind == "transformerlens" and cfg.capture is not None:
            os.makedirs(activations_dir, exist_ok=True)
            layers = [int(x) for x in str(cfg.capture.layers).split(",") if str(x).strip() != ""]
            components = [s.strip() for s in str(cfg.capture.components).split(",") if s.strip()]
            triggers = [s.strip() for s in str(cfg.capture.trigger_actions).split(",") if s.strip()]
            cap_cfg = CaptureConfig(
                layers=layers,
                components=components,
                trigger_actions=triggers,
                token_position=int(cfg.capture.token_position),
            )
            capture_context = CaptureContext(
                output_dir=str(activations_dir), config=cap_cfg, dtype=str(cfg.capture.dtype), trace_db=trace_db
            )

        meta = RunMetadata(
            run_id=run_id,
            seed=int(cfg.run.seed),
            created_at=time.time(),
            config={"mode": "experiment", **cfg.model_dump(mode="json")},
        )
        trace_db.insert_run(meta)

        # Build async-capable agent policies.
        agents = {}
        for i in range(int(cfg.run.agents)):
            agent_id = f"agent_{i:03d}"
            agent_seed = stable_agent_seed(int(cfg.run.seed), agent_id)
            if cfg.policy.kind == "random":
                sync = RandomAgentPolicy(random.Random(agent_seed))

                class _AsyncRandom:
                    async def adecide(self, *, run_id: str, time_step: int, agent_id: str, observation):  # type: ignore[no-untyped-def]
                        return sync.decide(
                            run_id=run_id, time_step=time_step, agent_id=agent_id, observation=observation
                        )

                agents[agent_id] = _AsyncRandom()
            elif cfg.policy.kind == "cognitive":
                gateway = (
                    MockLLMGateway(seed=agent_seed)
                    if bool(cfg.policy.mock_llm)
                    else LiteLLMGateway(
                        api_base=cfg.policy.api_base,
                        api_key=cfg.policy.api_key,
                        rate_limit_config=(
                            None
                            if not cfg.policy.rate_limit_enabled
                            else RateLimitConfig(
                                max_concurrent_requests=cfg.policy.rate_limit_max_concurrent,
                                requests_per_minute=cfg.policy.rate_limit_rpm,
                                tokens_per_minute=cfg.policy.rate_limit_tpm,
                            )
                        ),
                    )
                )
                agents[agent_id] = default_cognitive_policy(gateway=gateway, model=str(cfg.policy.model))
            else:
                if not cfg.policy.model_id:
                    raise RuntimeError("experiment policy.kind=transformerlens requires policy.model_id")
                gateway = TransformerLensGateway(model_id=str(cfg.policy.model_id), capture_context=capture_context)
                agents[agent_id] = default_cognitive_policy(gateway=gateway, model=str(cfg.policy.model_id))

        engine = WorldEngine(
            config=WorldEngineConfig(
                run_id=run_id,
                deterministic_timestamps=bool(cfg.run.deterministic_timestamps),
                message_history_limit=(int(cfg.policy.message_history) if cfg.policy.kind != "random" else 0),
            ),
            agents={},  # unused by scheduler path; kept for backwards compatibility
            channel=InMemoryChannel(),
            trace_db=trace_db,
            capture_context=capture_context,
        )

        scheduler = BarrierScheduler(
            config=BarrierSchedulerConfig(
                per_agent_timeout_s=float(cfg.scheduler.per_agent_timeout_s),
                max_concurrency=int(cfg.scheduler.max_concurrency),
                sort_mode=cfg.scheduler.sort_mode,
                seed=int(cfg.run.seed),
            ),
            engine=engine,
            agents=agents,  # async policies
        )

        asyncio.run(scheduler.run(steps=int(cfg.run.steps)))
        trace_db.close()

        if not args.no_validate:
            _validate_db(db_path=db_path, run_id=run_id, steps=int(cfg.run.steps), num_agents=int(cfg.run.agents))

        # Export to Parquet if requested
        if args.export_parquet:
            try:
                trace_parquet_path = os.path.join(run_dir, "trace.parquet")
                messages_parquet_path = os.path.join(run_dir, "messages.parquet")
                export_trace_to_parquet(trace_db=trace_db, run_id=run_id, output_path=trace_parquet_path)
                export_messages_to_parquet(trace_db=trace_db, run_id=run_id, output_path=messages_parquet_path)
                print(f"trace_parquet={trace_parquet_path}")
                print(f"messages_parquet={messages_parquet_path}")
            except Exception as e:
                print(f"Warning: Parquet export failed: {e}", file=sys.stderr)

        print(f"run_id={run_id}")
        print(f"run_dir={run_dir}")
        print(f"db={db_path}")
        if capture_context is not None:
            print(f"activations_dir={activations_dir}")
        return 0

    if mode == "olmo-conformity":
        capture_layers = None
        capture_components = None
        if args.capture_activations:
            if args.capture_layers:
                capture_layers = [int(x) for x in str(args.capture_layers).split(",") if str(x).strip() != ""]
            if args.capture_components:
                capture_components = [x.strip() for x in str(args.capture_components).split(",") if x.strip() != ""]
        
        paths = run_olmo_conformity_suite(
            suite_config_path=str(args.suite_config),
            runs_dir=str(args.runs_dir),
            run_id=str(args.run_id) if args.run_id else None,
            api_base=args.api_base,
            api_key=args.api_key,
            rate_limit_enabled=(not bool(args.no_rate_limit)),
            rate_limit_rpm=args.rate_limit_rpm,
            rate_limit_tpm=args.rate_limit_tpm,
            rate_limit_max_concurrent=int(args.rate_limit_max_concurrent),
            capture_activations=bool(args.capture_activations),
            capture_layers=capture_layers,
            capture_components=capture_components,
            capture_dtype=str(args.capture_dtype),
            use_judgeval=bool(args.use_judgeval),
            judgeval_judge_model=str(args.judgeval_judge_model),
            judgeval_ollama_base=str(args.judgeval_ollama_base),
        )
        print(f"run_dir={paths.run_dir}")
        print(f"db={paths.db_path}")
        return 0

    if mode == "olmo-conformity-probe":
        trace_db = TraceDb(TraceDbConfig(db_path=str(args.db)))
        trace_db.connect()
        trace_db.init_schema()

        layers = [int(x) for x in str(args.layers).split(",") if str(x).strip() != ""]
        cap = ProbeCaptureSpec(
            model_id=str(args.model_id),
            layers=layers,
            component=str(args.component),
            token_position=int(args.token_position),
            dtype=str(args.dtype),
        )

        # 1) Capture dataset activations into activations/*.safetensors and index metadata
        dataset_id = capture_probe_dataset_to_db(
            trace_db=trace_db,
            run_id=str(args.run_id),
            dataset_name=str(args.dataset_name),
            dataset_version=str(args.dataset_version),
            dataset_path=str(args.dataset_path),
            capture=cap,
            system_prompt="You are a truthful assistant. Answer True/False accurately.",
            temperature=float(args.temperature),
        )

        # 2) Train probe and save weights near the DB (run dir)
        run_dir = os.path.dirname(str(args.db))
        probe_path = os.path.join(run_dir, "artifacts", "tables", f"{args.probe_kind}_probe.safetensors")
        probe_id = train_probe_from_captured_activations(
            trace_db=trace_db,
            run_id=str(args.run_id),
            train_dataset_id=dataset_id,
            model_id=str(args.model_id),
            probe_kind=str(args.probe_kind),
            layers=layers,
            component=str(args.component),
            token_position=int(args.token_position),
            output_artifact_path=probe_path,
        )

        # 3) Compute projections for all trials in the run (if activations exist for them)
        inserted = compute_and_store_probe_projections_for_trials(
            trace_db=trace_db,
            run_id=str(args.run_id),
            probe_id=probe_id,
            probe_artifact_path=probe_path,
            model_id=str(args.model_id),
            component=str(args.component),
            layers=layers,
        )

        trace_db.close()
        print(f"dataset_id={dataset_id}")
        print(f"probe_id={probe_id}")
        print(f"probe_path={probe_path}")
        print(f"projection_rows_inserted={inserted}")
        return 0

    # Render plots/tables from conformity_* tables for a run
    if mode == "olmo-conformity-report":
        trace_db = TraceDb(TraceDbConfig(db_path=str(args.db)))
        trace_db.connect()
        trace_db.init_schema()
        out = generate_core_figures(trace_db=trace_db, run_id=str(args.run_id), run_dir=str(args.run_dir))
        trace_db.close()
        for k, v in out.items():
            print(f"{k}={v}")
        return 0

    if mode == "olmo-conformity-judgeval":
        # Post-hoc LLM judge: populate conformity_outputs.parsed_answer_json for the first output per trial.
        # Uses a condition-aware rubric and provides the exact prompts to the judge.
        from aam.experiments.olmo_conformity.ollama_judge import (
            JudgeInput,
            OllamaJudgeClient,
            OllamaJudgeConfig,
        )

        trace_db = TraceDb(TraceDbConfig(db_path=str(args.db)))
        trace_db.connect()
        trace_db.init_schema()

        # First prompt + first output per trial (stable; avoids accidentally judging later intervention outputs).
        where = "t.run_id = ?"
        params: list[object] = [str(args.run_id)]
        if str(args.trial_scope) == "behavioral-only":
            where += " AND c.name IN ('control', 'asch_history_5', 'authoritative_bias')"
        if not bool(args.force):
            where += " AND (o.parsed_answer_json IS NULL OR o.parsed_answer_json = '')"
        if args.limit is not None:
            limit_sql = " LIMIT ?"
            params.append(int(args.limit))
        else:
            limit_sql = ""

        rows = trace_db.conn.execute(
            f"""
            WITH first_prompts AS (
              SELECT trial_id, MIN(created_at) AS min_created_at
              FROM conformity_prompts
              GROUP BY trial_id
            ),
            first_prompt_ids AS (
              SELECT MIN(p.prompt_id) AS prompt_id, p.trial_id
              FROM conformity_prompts p
              JOIN first_prompts fp ON fp.trial_id = p.trial_id AND fp.min_created_at = p.created_at
              GROUP BY p.trial_id
            ),
            first_outputs AS (
              SELECT trial_id, MIN(created_at) AS min_created_at
              FROM conformity_outputs
              GROUP BY trial_id
            )
            SELECT
              t.trial_id,
              t.variant,
              t.item_id AS item_id,
              c.name AS condition_name,
              c.params_json AS condition_params_json,
              d.name AS dataset_name,
              i.question AS question,
              i.ground_truth_text AS ground_truth_text,
              i.source_json AS source_json,
              p.system_prompt AS system_prompt,
              p.user_prompt AS user_prompt,
              p.chat_history_json AS chat_history_json,
              o.output_id,
              o.raw_text AS raw_text,
              o.parsed_answer_json AS parsed_answer_json
            FROM conformity_trials t
            JOIN conformity_conditions c ON c.condition_id = t.condition_id
            JOIN conformity_items i ON i.item_id = t.item_id
            JOIN conformity_datasets d ON d.dataset_id = i.dataset_id
            LEFT JOIN first_prompt_ids fpi ON fpi.trial_id = t.trial_id
            LEFT JOIN conformity_prompts p ON p.prompt_id = fpi.prompt_id
            JOIN first_outputs fo ON fo.trial_id = t.trial_id
            JOIN conformity_outputs o ON o.trial_id = t.trial_id AND o.created_at = fo.min_created_at
            WHERE {where}
            ORDER BY t.created_at ASC
            {limit_sql};
            """,
            tuple(params),
        ).fetchall()

        if not rows:
            print("No trials to score (either none exist, or all already have parsed_answer_json).")
            trace_db.close()
            return 0

        updated = 0
        failed = 0

        def _parse_json(s: object) -> dict[str, object]:
            if s is None:
                return {}
            try:
                return json.loads(str(s))
            except Exception:
                return {}

        def _wrong_answer(source_json: object) -> Optional[str]:
            try:
                d = _parse_json(source_json)
                wa = d.get("wrong_answer")
                return str(wa) if wa is not None and str(wa).strip() != "" else None
            except Exception:
                return None

        def _condition_type(params_json: object, fallback: str) -> str:
            d = _parse_json(params_json)
            t = d.get("type")
            if t is None or str(t).strip() == "":
                return fallback
            return str(t)

        async def _score_all() -> list[tuple[str, Optional[dict[str, object]], Optional[str]]]:
            import asyncio

            cfg = OllamaJudgeConfig(
                model=str(args.judge_model),
                ollama_base=str(args.ollama_base),
            )
            results: list[tuple[str, Optional[dict[str, object]], Optional[str]]] = []

            sem = asyncio.Semaphore(max(1, int(args.max_concurrency)))
            async with OllamaJudgeClient(cfg) as judge:

                async def score_one(r: Any) -> None:
                    nonlocal results
                    output_id = str(r["output_id"])
                    try:
                        cond_name = str(r["condition_name"] or "unknown")
                        ji = JudgeInput(
                            condition_name=cond_name,
                            condition_type=_condition_type(r["condition_params_json"], cond_name),
                            system_prompt=str(r["system_prompt"] or ""),
                            user_prompt=str(r["user_prompt"] or ""),
                            chat_history_json=str(r["chat_history_json"] or "[]"),
                            question=str(r["question"] or ""),
                            model_output_raw=str(r["raw_text"] or ""),
                            reference_answer=(str(r["ground_truth_text"]) if r["ground_truth_text"] is not None else None),
                            injected_wrong_answer=_wrong_answer(r["source_json"]),
                            dataset_name=str(r["dataset_name"] or ""),
                            item_id=str(r["item_id"] or ""),
                            variant=str(r["variant"] or ""),
                        )
                        async with sem:
                            scored = await judge.judge(ji)
                        results.append((output_id, scored, None))
                    except Exception as e:
                        results.append((output_id, None, str(e)))

                tasks = [asyncio.create_task(score_one(r)) for r in rows]
                for idx, fut in enumerate(asyncio.as_completed(tasks), start=1):
                    await fut
                    if idx % 25 == 0 or idx == len(tasks):
                        print(f"Judged {idx}/{len(tasks)}...")

            return results

        import asyncio

        scored_rows = asyncio.run(_score_all())
        for output_id, scored, err in scored_rows:
            if scored is None:
                failed += 1
                print(f"Warning: Judge scoring failed for output_id={output_id[:8]}: {err}")
                continue
            try:
                trace_db.conn.execute(
                    "UPDATE conformity_outputs SET parsed_answer_json = ? WHERE output_id = ?;",
                    (json.dumps(scored, ensure_ascii=False), output_id),
                )
                updated += 1
            except Exception as e:
                failed += 1
                print(f"Warning: Judge DB update failed for output_id={output_id[:8]}: {e}")

        trace_db.conn.commit()
        trace_db.close()
        print(f"judgeval_scored={updated}")
        if failed:
            print(f"judgeval_failed={failed}")
        return 0

    if mode == "olmo-conformity-logit-lens":
        trace_db = TraceDb(TraceDbConfig(db_path=str(args.db)))
        trace_db.connect()
        trace_db.init_schema()
        layers = [int(x) for x in str(args.layers).split(",") if str(x).strip() != ""]
        if str(args.trial_scope) == "behavioral-only":
            trials = trace_db.conn.execute(
                """
                SELECT t.trial_id
                FROM conformity_trials t
                JOIN conformity_conditions c ON c.condition_id = t.condition_id
                WHERE t.run_id = ? AND c.name IN ('control', 'asch_history_5', 'authoritative_bias')
                ORDER BY t.created_at ASC;
                """,
                (str(args.run_id),),
            ).fetchall()
        else:
            trials = trace_db.conn.execute(
                "SELECT trial_id FROM conformity_trials WHERE run_id = ? ORDER BY created_at ASC;",
                (str(args.run_id),),
            ).fetchall()
        from aam.experiments.olmo_conformity.logit_lens import (
            analyze_think_rationalization,
            compute_logit_lens_for_think_tokens,
            compute_logit_lens_topk_for_trials,
            parse_and_store_think_tokens,
        )
        
        total = 0
        think_total = 0
        think_analysis_total = 0
        # Reuse a single HF model load across all trials.
        trial_ids = [str(tr["trial_id"]) for tr in trials]
        total += compute_logit_lens_topk_for_trials(
            trace_db=trace_db,
            trial_ids=trial_ids,
            model_id=str(args.model_id),
            layers=layers,
            k=int(args.topk),
            skip_existing=True,
        )

        for tr in trials:
            if bool(args.parse_think):
                think_total += parse_and_store_think_tokens(trace_db=trace_db, trial_id=str(tr["trial_id"]))
            if bool(args.analyze_think):
                think_analysis_total += compute_logit_lens_for_think_tokens(
                    trace_db=trace_db,
                    trial_id=str(tr["trial_id"]),
                    model_id=str(args.model_id),
                    layers=layers,
                    k=int(args.topk),
                )
                # Also analyze rationalization
                analysis = analyze_think_rationalization(trace_db=trace_db, trial_id=str(tr["trial_id"]))
                if analysis["rationalization_score"] > 0:
                    print(f"Trial {tr['trial_id'][:8]}: rationalization_score={analysis['rationalization_score']:.2f}, has_conflict={analysis['has_conflict']}")
        trace_db.close()
        print(f"logit_lens_rows={total}")
        if think_total > 0:
            print(f"think_tokens_parsed={think_total}")
        if think_analysis_total > 0:
            print(f"think_logit_lens_rows={think_analysis_total}")
        print(f"logit_rows_inserted={total}")
        if bool(args.parse_think):
            print(f"think_tokens_inserted={think_total}")
        return 0

    if mode == "olmo-conformity-posthoc":
        # Resolve run_id + db path.
        run_dir = str(args.run_dir)
        run_base = os.path.basename(run_dir.rstrip("/"))
        run_id = str(args.run_id) if args.run_id else (run_base.split("_")[-1] if "_" in run_base else run_base)
        db_path = str(args.db) if args.db else os.path.join(run_dir, "simulation.db")

        trace_db = TraceDb(TraceDbConfig(db_path=str(db_path)))
        trace_db.connect()
        trace_db.init_schema()

        # Resolve model_id if not provided.
        model_id = str(args.model_id) if args.model_id else None
        if not model_id:
            row = trace_db.conn.execute(
                "SELECT model_id FROM conformity_trials WHERE run_id = ? ORDER BY created_at ASC LIMIT 1;",
                (run_id,),
            ).fetchone()
            if row is None:
                trace_db.close()
                raise RuntimeError(f"No trials found for run_id={run_id}")
            model_id = str(row["model_id"])

        layers = [int(x) for x in str(args.layers).split(",") if str(x).strip() != ""]
        k = int(args.logit_lens_k)

        # Trial selection for logit-lens / think parsing.
        if str(args.trial_scope) == "behavioral-only":
            trials = trace_db.conn.execute(
                """
                SELECT t.trial_id
                FROM conformity_trials t
                JOIN conformity_conditions c ON c.condition_id = t.condition_id
                WHERE t.run_id = ? AND c.name IN ('control', 'asch_history_5', 'authoritative_bias')
                ORDER BY t.created_at ASC;
                """,
                (run_id,),
            ).fetchall()
        else:
            trials = trace_db.conn.execute(
                "SELECT trial_id FROM conformity_trials WHERE run_id = ? ORDER BY created_at ASC;",
                (run_id,),
            ).fetchall()
        trial_ids = [str(r["trial_id"]) for r in trials]

        # Optionally clear existing derived rows.
        if bool(args.clear_existing) and trial_ids:
            trace_db.conn.execute(
                f"DELETE FROM conformity_logit_lens WHERE trial_id IN ({','.join(['?']*len(trial_ids))});",
                trial_ids,
            )
            trace_db.conn.execute(
                f"DELETE FROM conformity_think_tokens WHERE trial_id IN ({','.join(['?']*len(trial_ids))});",
                trial_ids,
            )
            # Remove prior intervention rows for this run (results first).
            trace_db.conn.execute(
                """
                DELETE FROM conformity_intervention_results
                WHERE intervention_id IN (SELECT intervention_id FROM conformity_interventions WHERE run_id = ?);
                """,
                (run_id,),
            )
            trace_db.conn.execute("DELETE FROM conformity_interventions WHERE run_id = ?;", (run_id,))
            trace_db.conn.commit()

        # Think tokens
        from aam.experiments.olmo_conformity.logit_lens import (
            compute_logit_lens_topk_for_trials,
            parse_and_store_think_tokens,
        )

        think_inserted = 0
        if bool(args.parse_think_tokens):
            for tid in trial_ids:
                think_inserted += parse_and_store_think_tokens(trace_db=trace_db, trial_id=str(tid))

        # Logit lens (skip if requested)
        logit_inserted = 0
        if not bool(args.no_logit_lens):
            logit_inserted = compute_logit_lens_topk_for_trials(
                trace_db=trace_db,
                trial_ids=trial_ids,
                model_id=str(model_id),
                layers=layers,
                k=k,
                skip_existing=(not bool(args.clear_existing)),
            )

        # Interventions
        intervention_inserted = 0
        if not bool(args.no_interventions):
            # Find latest social probe for this run.
            sp = trace_db.conn.execute(
                """
                SELECT probe_id, artifact_path
                FROM conformity_probes
                WHERE run_id = ? AND probe_kind = 'social'
                ORDER BY created_at DESC
                LIMIT 1;
                """,
                (run_id,),
            ).fetchone()
            if sp is None:
                print("Warning: No social probe found for run; skipping interventions")
            else:
                social_probe_id = str(sp["probe_id"])
                probe_path = str(sp["artifact_path"])

                intervention_layers = [int(x) for x in str(args.intervention_layers).split(",") if str(x).strip() != ""]
                alphas = [float(x) for x in str(args.alphas).split(",") if str(x).strip() != ""]

                if str(args.intervention_scope) == "pressure-only":
                    trial_filter_sql = (
                        "i.ground_truth_text IS NOT NULL "
                        "AND t.condition_id IN (SELECT condition_id FROM conformity_conditions WHERE name IN ('asch_history_5','authoritative_bias'))"
                    )
                else:
                    trial_filter_sql = "i.ground_truth_text IS NOT NULL"

                intervention_inserted = run_intervention_sweep(
                    trace_db=trace_db,
                    run_id=run_id,
                    model_id=str(model_id),
                    probe_artifact_path=probe_path,
                    social_probe_id=social_probe_id,
                    target_layers=intervention_layers,
                    component_hook=str(args.component_hook),
                    alpha_values=alphas,
                    max_new_tokens=int(args.max_new_tokens),
                    trial_filter_sql=trial_filter_sql,
                )

        # Reporting refresh
        if not bool(args.no_report):
            try:
                _ = generate_core_figures(trace_db=trace_db, run_id=run_id, run_dir=run_dir)
            except Exception as e:
                print(f"Warning: report generation failed: {e}", file=sys.stderr)

        trace_db.close()

        print("=" * 60)
        print("Posthoc backfill complete")
        print("=" * 60)
        print(f"run_id={run_id}")
        print(f"db={db_path}")
        print(f"model_id={model_id}")
        print(f"trial_scope={args.trial_scope} (n_trials={len(trial_ids)})")
        print(f"logit_lens_rows_inserted={logit_inserted}")
        if bool(args.parse_think_tokens):
            print(f"think_tokens_inserted={think_inserted}")
        print(f"intervention_results_inserted={intervention_inserted}")
        return 0

    if mode == "olmo-conformity-intervene":
        trace_db = TraceDb(TraceDbConfig(db_path=str(args.db)))
        trace_db.connect()
        trace_db.init_schema()
        layers = [int(x) for x in str(args.layers).split(",") if str(x).strip() != ""]
        alphas = [float(x) for x in str(args.alpha).split(",") if str(x).strip() != ""]
        inserted = run_intervention_sweep(
            trace_db=trace_db,
            run_id=str(args.run_id),
            model_id=str(args.model_id),
            probe_artifact_path=str(args.probe_path),
            social_probe_id=str(args.social_probe_id),
            target_layers=layers,
            component_hook=str(args.component_hook),
            alpha_values=alphas,
            max_new_tokens=int(args.max_new_tokens),
        )
        trace_db.close()
        print(f"intervention_results_inserted={inserted}")
        return 0

    if mode == "olmo-conformity-vector-analysis":
        from aam.experiments.olmo_conformity.vector_analysis import run_truth_social_vector_analysis
        
        trace_db = TraceDb(TraceDbConfig(db_path=str(args.db)))
        trace_db.connect()
        trace_db.init_schema()
        
        layers = [int(x) for x in str(args.layers).split(",") if str(x).strip() != ""]
        
        results = run_truth_social_vector_analysis(
            trace_db=trace_db,
            run_id=str(args.run_id),
            model_id=str(args.model_id),
            truth_probe_dataset_path=str(args.truth_probe_dataset),
            social_probe_dataset_path=str(args.social_probe_dataset) if args.social_probe_dataset else None,
            layers=layers,
            component=str(args.component),
            token_position=int(args.token_position),
            dtype=str(args.dtype),
            artifacts_dir=str(args.artifacts_dir),
        )
        
        trace_db.close()
        
        print("\n" + "="*60)
        print("Vector Analysis Results")
        print("="*60)
        print(f"Truth Probe ID: {results['truth_probe_id']}")
        if results['social_probe_id']:
            print(f"Social Probe ID: {results['social_probe_id']}")
        print(f"Projection Stats: {results['projection_stats']}")
        print(f"Turn Layers: {results['turn_layers']}")
        print(f"Analysis Artifacts: {results['analysis_artifacts']}")
        return 0

    if mode == "olmo-conformity-resume":
        from aam.experiments.olmo_conformity.resume import resume_from_projections

        trace_db = TraceDb(TraceDbConfig(db_path=str(args.db)))
        trace_db.connect()
        trace_db.init_schema()

        run_dir = str(args.run_dir) if args.run_dir else os.path.dirname(str(args.db))
        layers = [int(x) for x in str(args.layers).split(",") if str(x).strip() != ""]

        results = resume_from_projections(
            trace_db=trace_db,
            run_id=str(args.run_id),
            model_id=str(args.model_id),
            run_dir=run_dir,
            layers=layers,
            component=str(args.component),
            repair_activations_first=(not bool(args.no_repair_activations)),
            max_new_tokens=int(args.max_new_tokens),
        )
        trace_db.close()
        print(f"\nResume results: {results}")
        return 0

    if mode == "olmo-conformity-full":
        from aam.experiments.olmo_conformity.orchestration import ExperimentConfig, run_full_experiment
        from aam.experiments.olmo_conformity.io import load_suite_config
        
        # Parse layers and alphas
        capture_layers = None
        if args.capture_layers:
            capture_layers = [int(x) for x in str(args.capture_layers).split(",") if str(x).strip() != ""]
        
        probe_layers = None
        if args.probe_layers:
            probe_layers = [int(x) for x in str(args.probe_layers).split(",") if str(x).strip() != ""]
        
        intervention_layers = None
        if args.intervention_layers:
            intervention_layers = [int(x) for x in str(args.intervention_layers).split(",") if str(x).strip() != ""]
        
        intervention_alphas = None
        if args.intervention_alphas:
            intervention_alphas = [float(x) for x in str(args.intervention_alphas).split(",") if str(x).strip() != ""]
        
        # Extract temperature from suite config (not experiment config)
        suite_cfg = load_suite_config(str(args.suite_config))
        temperature = float(suite_cfg.get("run", {}).get("temperature", 0.0))
        
        config = ExperimentConfig(
            suite_config_path=str(args.suite_config),
            runs_dir=str(args.runs_dir),
            run_id=str(args.run_id) if args.run_id else None,
            api_base=args.api_base,
            api_key=args.api_key,
            rate_limit_enabled=(not bool(args.no_rate_limit)),
            capture_activations=bool(args.capture_activations),
            capture_layers=capture_layers,
            truth_probe_dataset_path=str(args.truth_probe_dataset) if args.truth_probe_dataset else None,
            social_probe_dataset_path=str(args.social_probe_dataset) if args.social_probe_dataset else None,
            probe_layers=probe_layers,
            run_interventions=bool(args.run_interventions),
            intervention_layers=intervention_layers,
            intervention_alphas=intervention_alphas,
            social_probe_artifact_path=str(args.social_probe_path) if args.social_probe_path else None,
            social_probe_id=str(args.social_probe_id) if args.social_probe_id else None,
            generate_reports=(not bool(args.no_reports)),
            run_vector_analysis=bool(args.run_vector_analysis),
            temperature=temperature,
        )
        
        results = run_full_experiment(config)
        print(f"\nFull experiment results: {results}")
        return 0

    run_id = args.run_id or str(uuid.uuid4())

    # Phase 3 uses PRD-style run artifact layout: runs/<timestamp>_<run_id>/simulation.db + activations/
    if mode == "phase3":
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        run_dir = os.path.join(args.runs_dir, f"{ts}_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        db_path = os.path.join(run_dir, "simulation.db")
        activations_dir = os.path.join(run_dir, "activations")
        os.makedirs(activations_dir, exist_ok=True)
    else:
        db_path = args.db
        run_dir = None
        activations_dir = None

    trace_db = TraceDb(TraceDbConfig(db_path=db_path))
    trace_db.connect()
    trace_db.init_schema()

    config = {
        "mode": mode,
        "steps": args.steps,
        "agents": args.agents,
        "deterministic_timestamps": (not args.nondeterministic_timestamps),
    }
    if mode == "phase2":
        config.update(
            {
                "model": args.model,
                "mock_llm": bool(args.mock_llm),
                "message_history": args.message_history,
            }
        )
    if mode == "phase3":
        config.update(
            {
                "model_id": args.model_id,
                "runs_dir": args.runs_dir,
                "output_dir": run_dir,
                "message_history": args.message_history,
                "capture": {
                    "layers": args.layers,
                    "components": args.components,
                    "trigger_actions": args.trigger_actions,
                    "token_position": args.token_position,
                    "dtype": args.dtype,
                },
            }
        )

    meta = RunMetadata(run_id=run_id, seed=args.seed, created_at=time.time(), config=config)
    trace_db.insert_run(meta)

    capture_context = None
    if mode == "phase3":
        layers = [int(x) for x in str(args.layers).split(",") if str(x).strip() != ""]
        components = [s.strip() for s in str(args.components).split(",") if s.strip()]
        triggers = [s.strip() for s in str(args.trigger_actions).split(",") if s.strip()]
        cap_cfg = CaptureConfig(
            layers=layers,
            components=components,
            trigger_actions=triggers,
            token_position=int(args.token_position),
        )
        capture_context = CaptureContext(
            output_dir=str(activations_dir), config=cap_cfg, dtype=str(args.dtype), trace_db=trace_db
        )

    agents = {}
    for i in range(args.agents):
        agent_id = f"agent_{i:03d}"
        agent_seed = stable_agent_seed(args.seed, agent_id)
        if mode == "phase1":
            agents[agent_id] = RandomAgentPolicy(random.Random(agent_seed))
        elif mode == "phase2":
            gateway = (
                MockLLMGateway(seed=agent_seed)
                if args.mock_llm
                else LiteLLMGateway(
                    api_base=args.api_base,
                    api_key=args.api_key,
                    rate_limit_config=(
                        None
                        if args.no_rate_limit
                        else RateLimitConfig(
                            max_concurrent_requests=args.rate_limit_max_concurrent,
                            requests_per_minute=args.rate_limit_rpm,
                            tokens_per_minute=args.rate_limit_tpm,
                        )
                    ),
                )
            )
            agents[agent_id] = default_cognitive_policy(gateway=gateway, model=args.model)
        else:
            gateway = TransformerLensGateway(model_id=str(args.model_id), capture_context=capture_context)
            if bool(args.list_hooks):
                hooks = CaptureContext.list_available_hooks(gateway._model)
                for h in hooks[: int(args.hooks_head)]:
                    print(h)
                trace_db.close()
                return 0
            agents[agent_id] = default_cognitive_policy(gateway=gateway, model=str(args.model_id))

    engine = WorldEngine(
        config=WorldEngineConfig(
            run_id=run_id,
            deterministic_timestamps=(not args.nondeterministic_timestamps),
            message_history_limit=(args.message_history if mode in ("phase2", "phase3") else 0),
        ),
        agents=agents,
        channel=InMemoryChannel(),
        trace_db=trace_db,
        capture_context=capture_context,
    )
    engine.run(steps=args.steps)
    trace_db.close()

    if not args.no_validate:
        _validate_db(db_path=db_path, run_id=run_id, steps=args.steps, num_agents=args.agents)

    # Export to Parquet if requested
    if hasattr(args, "export_parquet") and args.export_parquet:
        try:
            if run_dir:
                trace_parquet_path = os.path.join(run_dir, "trace.parquet")
                messages_parquet_path = os.path.join(run_dir, "messages.parquet")
            else:
                base_path = os.path.splitext(db_path)[0]
                trace_parquet_path = f"{base_path}_trace.parquet"
                messages_parquet_path = f"{base_path}_messages.parquet"
            export_trace_to_parquet(trace_db=trace_db, run_id=run_id, output_path=trace_parquet_path)
            export_messages_to_parquet(trace_db=trace_db, run_id=run_id, output_path=messages_parquet_path)
            print(f"trace_parquet={trace_parquet_path}")
            print(f"messages_parquet={messages_parquet_path}")
        except Exception as e:
            print(f"Warning: Parquet export failed: {e}", file=sys.stderr)

    print(f"run_id={run_id}")
    print(f"db={db_path}")
    if mode == "phase3":
        print(f"run_dir={run_dir}")
        print(f"activations_dir={activations_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
