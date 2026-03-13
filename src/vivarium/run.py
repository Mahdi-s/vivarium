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
from typing import Any

from vivarium.agent_langgraph import default_cognitive_policy
from vivarium.channel import InMemoryChannel
from vivarium.experiment_config import load_experiment_config

try:
    from vivarium.export import export_messages_to_parquet, export_trace_to_parquet
except ImportError:
    export_messages_to_parquet = None  # type: ignore[assignment]
    export_trace_to_parquet = None  # type: ignore[assignment]
from vivarium.interpretability import CaptureConfig, CaptureContext
from vivarium.llama_cpp import LlamaServerConfig, run_llama_server
from vivarium.model_discovery import discover_all_models, export_models
from vivarium.persistence import TraceDb, TraceDbConfig
from vivarium.policy import RandomAgentPolicy, stable_agent_seed
from vivarium.scheduler import BarrierScheduler, BarrierSchedulerConfig
from vivarium.types import RunMetadata
from vivarium.world_engine import WorldEngine, WorldEngineConfig


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


def _import_llm_gateways() -> tuple[Any, Any, Any, Any]:
    """
    Lazy import LLM gateways to avoid importing optional heavy deps at CLI import time.
    """
    from vivarium.llm_gateway import LiteLLMGateway, MockLLMGateway, RateLimitConfig, TransformerLensGateway

    return LiteLLMGateway, MockLLMGateway, RateLimitConfig, TransformerLensGateway


def main(argv: list[str] | None = None) -> int:
    # Load .env file if python-dotenv is available
    # This allows users to configure environment variables (VIVARIUM_MODEL_DIR, etc.)
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
    p1.add_argument("--deterministic-ids", action="store_true", help="Use deterministic IDs for trace/message rows")

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
    p2.add_argument("--deterministic-ids", action="store_true", help="Use deterministic IDs for trace/message rows")
    p2.add_argument("--message-history", type=int, default=20)
    p2.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (0=greedy)")
    p2.add_argument("--top-k", "--top-n", dest="top_k", type=int, default=None, help="Top-k sampling cutoff (aka top_n)")
    p2.add_argument("--top-p", "--nucleus-p", dest="top_p", type=float, default=None, help="Nucleus sampling cutoff (aka nucleus_p)")
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
    p3.add_argument("--deterministic-ids", action="store_true", help="Use deterministic IDs for trace/message rows")
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

    # Experiment subcommands (olmo-conformity, etc.) registered by experiment modules
    from vivarium.experiments import register_experiment_cli

    register_experiment_cli(sub)

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

    # Dispatch to experiment handlers (olmo-conformity, etc.)
    from vivarium.experiments import handle_experiment_command

    result = handle_experiment_command(mode, args)
    if result is not None:
        return result

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

        LiteLLMGateway, MockLLMGateway, RateLimitConfig, TransformerLensGateway = _import_llm_gateways()

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
                agents[agent_id] = default_cognitive_policy(
                    gateway=gateway,
                    model=str(cfg.policy.model),
                    temperature=float(cfg.policy.temperature),
                    top_k=(int(cfg.policy.top_k) if cfg.policy.top_k is not None else None),
                    top_p=(float(cfg.policy.top_p) if cfg.policy.top_p is not None else None),
                )
            else:
                if not cfg.policy.model_id:
                    raise RuntimeError("experiment policy.kind=transformerlens requires policy.model_id")
                gateway = TransformerLensGateway(model_id=str(cfg.policy.model_id), capture_context=capture_context)
                agents[agent_id] = default_cognitive_policy(
                    gateway=gateway,
                    model=str(cfg.policy.model_id),
                    temperature=float(cfg.policy.temperature),
                    top_k=(int(cfg.policy.top_k) if cfg.policy.top_k is not None else None),
                    top_p=(float(cfg.policy.top_p) if cfg.policy.top_p is not None else None),
                )

        engine = WorldEngine(
            config=WorldEngineConfig(
                run_id=run_id,
                deterministic_timestamps=bool(cfg.run.deterministic_timestamps),
                deterministic_ids=bool(cfg.run.deterministic_ids),
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

    # Phase 1/2/3 fallthrough: run_id used below
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
        "deterministic_ids": bool(getattr(args, "deterministic_ids", False)),
    }
    if mode == "phase2":
        config.update(
            {
                "model": args.model,
                "mock_llm": bool(args.mock_llm),
                "message_history": args.message_history,
                "temperature": float(args.temperature),
                "top_k": (int(args.top_k) if args.top_k is not None else None),
                "top_p": (float(args.top_p) if args.top_p is not None else None),
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

    LiteLLMGateway = MockLLMGateway = RateLimitConfig = TransformerLensGateway = None
    if mode in ("phase2", "phase3"):
        LiteLLMGateway, MockLLMGateway, RateLimitConfig, TransformerLensGateway = _import_llm_gateways()

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
            agents[agent_id] = default_cognitive_policy(
                gateway=gateway,
                model=args.model,
                temperature=float(args.temperature),
                top_k=(int(args.top_k) if args.top_k is not None else None),
                top_p=(float(args.top_p) if args.top_p is not None else None),
            )
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
            deterministic_ids=bool(getattr(args, "deterministic_ids", False)),
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
