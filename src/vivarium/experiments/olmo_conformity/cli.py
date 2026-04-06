"""OLMo Conformity experiment CLI subcommands and handlers."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from vivarium.persistence import TraceDb, TraceDbConfig

OLMO_CONFORMITY_MODES = [
    "olmo-conformity",
    "olmo-conformity-probe",
    "olmo-conformity-report",
    "olmo-conformity-judgeval",
    "olmo-conformity-logit-lens",
    "olmo-conformity-intervene",
    "olmo-conformity-vector-analysis",
    "olmo-conformity-resume",
    "olmo-conformity-posthoc",
    "olmo-conformity-full",
    "olmo-conformity-backfill-behavioral",
    "olmo-conformity-complete-suite",
]


def register_subparsers(subparsers: Any) -> None:
    """Register olmo-conformity subcommands with the main CLI subparsers."""
    default_layers_32 = ",".join(str(i) for i in range(32))

    pc = subparsers.add_parser("olmo-conformity", help="Run Olmo conformity experiment suite (Synthetic Asch + logging)")
    pc.add_argument("--suite-config", type=str, required=True, help="Path to suite config JSON (experiments/olmo_conformity/configs/...)")
    pc.add_argument("--runs-dir", type=str, default="./runs", help="Base directory for outputs: runs/<timestamp>_<run_id>/")
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
    pc.add_argument("--resume-auto", action="store_true", help="Auto-detect and resume the most recent incomplete run for this suite config instead of starting fresh")
    pc.add_argument(
        "--openrouter-provider",
        type=str,
        default=None,
        help=(
            "JSON string for OpenRouter provider routing preferences, injected as extra_body.provider. "
            "Example: '{\"order\": [\"Groq\"], \"allow_fallbacks\": false}'. "
            "Per-model openrouter_provider in the suite config takes precedence over this flag."
        ),
    )
    pc.add_argument(
        "--async-mode",
        action="store_true",
        help=(
            "Enable async concurrent trial execution: fan out LLM calls in parallel "
            "and batch-write results. Ideal for API-backed models. "
            "Incompatible with --capture-activations (falls back to serial)."
        ),
    )
    pc.add_argument(
        "--db-flush-batch-size",
        type=int,
        default=None,
        help="Number of trials to batch per async fan-out + DB flush window (default: 20 or suite config value).",
    )

    pp = subparsers.add_parser("olmo-conformity-probe", help="Capture activations for probe dataset, train probe, and compute projections")
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

    pr = subparsers.add_parser("olmo-conformity-report", help="Generate figures/tables from conformity_* tables for a run")
    pr.add_argument("--run-id", type=str, required=True)
    pr.add_argument("--db", type=str, required=True, help="Path to simulation.db for the run")
    pr.add_argument("--run-dir", type=str, required=True, help="Path to run directory (writes artifacts/)")

    pj = subparsers.add_parser(
        "olmo-conformity-judgeval",
        help="Backfill LLM judge scores into conformity_outputs.parsed_answer_json for an existing run",
    )
    pj.add_argument("--run-id", type=str, required=True)
    pj.add_argument("--db", type=str, required=True, help="Path to simulation.db for the run")
    pj.add_argument("--judge-config", type=str, default=None, help="Path to judge config JSON (model, prompt templates, temperature, etc.)")
    pj.add_argument("--judge-model", type=str, default=None, help="Judge model id (overrides --judge-config; also accepts OpenRouter model ids)")
    pj.add_argument("--ollama-base", type=str, default=None, help="API base URL — Ollama local or OpenRouter (overrides --judge-config)")
    pj.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "Bearer API key for the judge provider (e.g. OpenRouter sk-or-v1-…). "
            "Falls back to the OPENROUTER_API_KEY / JUDGE_API_KEY env vars when not set."
        ),
    )
    pj.add_argument("--force", action="store_true", help="Overwrite existing parsed_answer_json regardless of current content")
    pj.add_argument(
        "--no-llm-judge",
        action="store_true",
        help=(
            "Target rows that have non-empty parsed_answer_json but were NEVER LLM-judged "
            "(i.e. the heuristic rule-based parser ran but no _llm_judge key is present). "
            "Useful for backfilling new runs that already have heuristic labels."
        ),
    )
    pj.add_argument(
        "--retry-parse-errors",
        action="store_true",
        help="Re-judge rows that have [parse_error] in parsed_answer_json",
    )
    pj.add_argument("--limit", type=int, default=None, help="Optional cap on number of trials to score")
    pj.add_argument("--max-concurrency", type=int, default=4, help="Max concurrent judge requests (default: 4)")
    pj.add_argument(
        "--verbose",
        action="store_true",
        help="Print raw judge model output for each example (for debugging parse errors)",
    )
    pj.add_argument(
        "--no-json-format",
        action="store_true",
        help="Disable JSON-mode constraint (use if the provider does not support it)",
    )
    pj.add_argument(
        "--trial-scope",
        type=str,
        default="all",
        choices=["behavioral-only", "all"],
        help=(
            "Which trials to score. "
            "'all' (default): every trial in the run (all conditions). "
            "'behavioral-only': restrict to the three legacy conditions "
            "(control, asch_history_5, authoritative_bias)."
        ),
    )
    pj.add_argument(
        "--parse-error-retries",
        type=int,
        default=5,
        help="Per-row retry limit when the judge returns an unparseable response (default: 5).",
    )
    pj.add_argument(
        "--variant-filter",
        type=str,
        default=None,
        help=(
            "Comma-separated list of variant names to judge (e.g. 'base,instruct,instruct_sft,instruct_dpo'). "
            "When set, only trials whose variant matches one of the listed names are scored. "
            "Default: score all variants."
        ),
    )

    pl = subparsers.add_parser("olmo-conformity-logit-lens", help="Compute logit-lens top-k across layers for each trial")
    pl.add_argument("--run-id", type=str, required=True)
    pl.add_argument("--db", type=str, required=True)
    pl.add_argument("--model-id", type=str, required=True)
    pl.add_argument("--layers", type=str, default="0", help="Comma-separated layer indices")
    pl.add_argument("--topk", type=int, default=10)
    pl.add_argument("--parse-think", action="store_true", help="Also parse <think>...</think> into vivarium_think_tokens")
    pl.add_argument("--analyze-think", action="store_true", help="Also compute logit lens for intermediate <think> tokens")
    pl.add_argument(
        "--trial-scope",
        type=str,
        default="all",
        choices=["all", "behavioral-only"],
        help="Which trials to process (default: all trials in run).",
    )

    pi = subparsers.add_parser("olmo-conformity-intervene", help="Run social-vector subtraction intervention sweep (TransformerLens)")
    pi.add_argument("--run-id", type=str, required=True)
    pi.add_argument("--db", type=str, required=True)
    pi.add_argument("--model-id", type=str, required=True)
    pi.add_argument("--probe-path", type=str, required=True, help="Path to social probe safetensors (layer_*.weight)")
    pi.add_argument("--social-probe-id", type=str, required=True, help="vivarium_probes.probe_id for the social vector")
    pi.add_argument("--layers", type=str, default="0", help="Comma-separated target layers")
    pi.add_argument("--alpha", type=str, default="1.0", help="Comma-separated alpha values, e.g. '0.5,1.0,2.0'")
    pi.add_argument("--component-hook", type=str, default="hook_resid_post")
    pi.add_argument("--max-new-tokens", type=int, default=64)

    pv = subparsers.add_parser("olmo-conformity-vector-analysis", help="Run Truth vs Social Vector analysis workflow")
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

    prr = subparsers.add_parser(
        "olmo-conformity-resume",
        help="Resume an existing run from the projection step (optionally repairing overwritten trial activations)",
    )
    prr.add_argument("--run-id", type=str, required=True)
    prr.add_argument("--db", type=str, required=True, help="Path to simulation.db for the run")
    prr.add_argument("--model-id", type=str, required=True)
    prr.add_argument("--run-dir", type=str, default=None, help="Path to run directory (defaults to dirname(--db))")
    prr.add_argument("--layers", type=str, default=default_layers_32, help="Comma-separated layer indices")
    prr.add_argument("--component", type=str, default="hook_resid_post")
    prr.add_argument("--max-new-tokens", type=int, default=128)
    prr.add_argument("--no-repair-activations", action="store_true", help="Skip trial activation repair step")

    pph = subparsers.add_parser(
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
    pph.add_argument("--parse-think-tokens", action="store_true", help="Parse <think>...</think> blocks into vivarium_think_tokens")
    pph.add_argument("--no-logit-lens", action="store_true", help="Skip logit lens computation")
    pph.add_argument(
        "--no-answer-logprobs",
        action="store_true",
        help="Skip answer-level logprob probes (correct vs conforming) for behavioral trials",
    )
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
    # --- New interpretability analyses ---
    pph.add_argument("--no-logit-lens-tug-of-war", action="store_true", help="Skip logit lens tug-of-war (P(truth) vs P(sycophantic) per layer)")
    pph.add_argument("--no-contrastive-steering", action="store_true", help="Skip contrastive vector steering (RepE / CAA)")
    pph.add_argument("--no-activation-patching", action="store_true", help="Skip activation patching (causal tracing)")
    pph.add_argument(
        "--steering-alphas", type=str, default="-2.0,-1.0,0.5,1.0,2.0,4.0",
        help="Comma-separated alpha values for contrastive steering (per Panickssery et al., ACL 2024)",
    )
    pph.add_argument(
        "--steering-layers", type=str, default="10,11,12,13,14,15,16,17,18,19,20",
        help="Comma-separated layers for steering vector computation and application",
    )
    pph.add_argument(
        "--steering-min-pairs", type=int, default=50,
        help="Minimum Control/Authority pairs required (per Hao et al., ICLR 2025 Workshops)",
    )
    pph.add_argument(
        "--patching-layers", type=str, default=None,
        help="Comma-separated layers for activation patching (defaults to --layers)",
    )

    pe = subparsers.add_parser("olmo-conformity-full", help="Run complete experiment workflow (trials → probes → interventions → analysis)")
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

    pbf = subparsers.add_parser(
        "olmo-conformity-backfill-behavioral",
        help="Backfill missing behavioral outputs for trials without conformity_outputs rows",
    )
    pbf.add_argument(
        "--runs-dir",
        type=str,
        default="./runs_latest/runs",
        help="Base directory containing run folders with simulation.db",
    )
    pbf.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional: only backfill this specific run (filters by run_dir ending with _<run_id>)",
    )
    pbf.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="Ollama/OpenAI-compatible API base (e.g. http://localhost:11434/v1). If unset, uses local HuggingFace.",
    )
    pbf.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for remote inference (optional)",
    )
    pbf.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts of missing trials per run without executing",
    )

    pcs = subparsers.add_parser(
        "olmo-conformity-complete-suite",
        help="Complete missing trials/outputs per suite_expanded_temp{T}.json for runs in metadata",
    )
    pcs.add_argument(
        "--runs-dir",
        type=str,
        required=True,
        help="Base directory containing run folders (e.g. runs_latest/runs)",
    )
    pcs.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to runs_metadata JSON (e.g. Comparing_Experiments/runs_metadata_v5.json)",
    )
    pcs.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional: only complete this specific run",
    )
    pcs.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="Ollama/OpenAI-compatible API base. If unset, uses local HuggingFace.",
    )
    pcs.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for remote inference (optional)",
    )
    pcs.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts of missing items, trials, outputs per run without executing",
    )


def handle_command(mode: str, args: Any) -> Optional[int]:
    """Handle olmo-conformity commands. Returns exit code if handled, None otherwise."""
    if mode not in OLMO_CONFORMITY_MODES:
        return None

    if mode == "olmo-conformity":
        return _handle_olmo_conformity(args)
    if mode == "olmo-conformity-probe":
        return _handle_probe(args)
    if mode == "olmo-conformity-report":
        return _handle_report(args)
    if mode == "olmo-conformity-judgeval":
        return _handle_judgeval(args)
    if mode == "olmo-conformity-logit-lens":
        return _handle_logit_lens(args)
    if mode == "olmo-conformity-posthoc":
        return _handle_posthoc(args)
    if mode == "olmo-conformity-intervene":
        return _handle_intervene(args)
    if mode == "olmo-conformity-vector-analysis":
        return _handle_vector_analysis(args)
    if mode == "olmo-conformity-resume":
        return _handle_resume(args)
    if mode == "olmo-conformity-full":
        return _handle_full(args)
    if mode == "olmo-conformity-backfill-behavioral":
        return _handle_backfill_behavioral(args)
    if mode == "olmo-conformity-complete-suite":
        return _handle_complete_suite(args)

    return None


def _handle_olmo_conformity(args: Any) -> int:
    from vivarium.experiments.olmo_conformity.runner import run_suite as run_olmo_conformity_suite

    capture_layers = None
    capture_components = None
    if args.capture_activations:
        if args.capture_layers:
            capture_layers = [int(x) for x in str(args.capture_layers).split(",") if str(x).strip() != ""]
        if args.capture_components:
            capture_components = [x.strip() for x in str(args.capture_components).split(",") if x.strip() != ""]

    # Parse optional OpenRouter provider routing JSON.
    openrouter_provider = None
    _or_provider_raw = getattr(args, "openrouter_provider", None)
    if _or_provider_raw:
        try:
            openrouter_provider = json.loads(_or_provider_raw)
        except Exception as e:
            print(f"Warning: --openrouter-provider is not valid JSON, ignoring: {e}")

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
        resume_auto=bool(args.resume_auto),
        execution_mode="async" if getattr(args, "async_mode", False) else None,
        db_flush_batch_size=getattr(args, "db_flush_batch_size", None),
        openrouter_provider=openrouter_provider,
    )
    print(f"run_dir={paths.run_dir}")
    print(f"db={paths.db_path}")
    return 0


def _handle_probe(args: Any) -> int:
    from vivarium.experiments.olmo_conformity.probes import (
        ProbeCaptureSpec,
        capture_probe_dataset_to_db,
        compute_and_store_probe_projections_for_trials,
        train_probe_from_captured_activations,
    )

    trace_db = TraceDb(TraceDbConfig(db_path=str(args.db)))
    trace_db.connect()
    trace_db.init_schema()
    trace_db.init_conformity_schema()

    layers = [int(x) for x in str(args.layers).split(",") if str(x).strip() != ""]
    cap = ProbeCaptureSpec(
        model_id=str(args.model_id),
        layers=layers,
        component=str(args.component),
        token_position=int(args.token_position),
        dtype=str(args.dtype),
    )

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


def _handle_report(args: Any) -> int:
    from vivarium.experiments.olmo_conformity.analysis import generate_core_figures

    trace_db = TraceDb(TraceDbConfig(db_path=str(args.db)))
    trace_db.connect()
    trace_db.init_schema()
    trace_db.init_conformity_schema()
    out = generate_core_figures(trace_db=trace_db, run_id=str(args.run_id), run_dir=str(args.run_dir))
    trace_db.close()
    for k, v in out.items():
        print(f"{k}={v}")
    return 0


def _handle_judgeval(args: Any) -> int:
    from vivarium.experiments.olmo_conformity.ollama_judge import (
        JudgeInput,
        OllamaJudgeClient,
        OllamaJudgeConfig,
        _is_parse_error,
    )

    # Build judge config: start from JSON file (if given), then apply any
    # explicit CLI overrides.
    if args.judge_config:
        cfg = OllamaJudgeConfig.from_json_file(str(args.judge_config))
        print(f"Loaded judge config from {args.judge_config} (prompt_version={cfg.prompt_version})")
    else:
        cfg = OllamaJudgeConfig()

    # Resolve API key: CLI flag > env vars > config file value
    api_key = getattr(args, "api_key", None) or None
    if api_key is None:
        import os as _os
        api_key = _os.environ.get("OPENROUTER_API_KEY") or _os.environ.get("JUDGE_API_KEY") or None

    cfg = cfg.merge_cli(
        judge_model=args.judge_model if args.judge_model else None,
        ollama_base=args.ollama_base if args.ollama_base else None,
        api_key=api_key,
        verbose=bool(args.verbose),
        use_json_format=False if args.no_json_format else None,
    )
    # Disable inner parse-error retries; the outer loop in score_one handles them.
    import dataclasses as _dc
    cfg = _dc.replace(cfg, max_retries=0)
    max_conc = max(1, int(args.max_concurrency))
    max_parse_error_retries = max(1, int(args.parse_error_retries))
    provider_label = "openai-compatible" if cfg.is_openai_compatible else "ollama"
    print(
        f"Judge: model={cfg.model}  base={cfg.api_base}  provider={provider_label}"
        f"  auth={'yes' if cfg.api_key else 'no'}"
        f"  max_concurrency={max_conc}  parse_error_retries={max_parse_error_retries}"
    )

    trace_db = TraceDb(TraceDbConfig(db_path=str(args.db)))
    trace_db.connect()
    trace_db.init_schema()
    trace_db.init_conformity_schema()

    where = "t.run_id = ?"
    params: list[object] = [str(args.run_id)]
    if str(args.trial_scope) == "behavioral-only":
        where += " AND c.name IN ('control', 'asch_history_5', 'authoritative_bias')"
    variant_filter = getattr(args, "variant_filter", None)
    if variant_filter:
        allowed_variants = [v.strip() for v in str(variant_filter).split(",") if v.strip()]
        if allowed_variants:
            placeholders = ",".join("?" * len(allowed_variants))
            where += f" AND t.variant IN ({placeholders})"
            params.extend(allowed_variants)
    if not bool(args.force):
        # --no-llm-judge: target rows that have non-empty content but were never
        # LLM-judged (no _llm_judge key — typically heuristic-only labels).
        no_llm_judge = getattr(args, "no_llm_judge", False)
        needs_judge = "(o.parsed_answer_json IS NULL OR o.parsed_answer_json = '')"
        if no_llm_judge:
            needs_judge = (
                f"({needs_judge}"
                " OR (o.parsed_answer_json IS NOT NULL AND o.parsed_answer_json != ''"
                " AND json_extract(o.parsed_answer_json, '$._llm_judge') IS NULL))"
            )
        if bool(args.retry_parse_errors):
            needs_judge = f"({needs_judge} OR o.parsed_answer_json LIKE '%[parse_error]%')"
        where += f" AND {needs_judge}"
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
        print(
            "No trials to score (none exist, or all already have LLM judge labels — "
            "use --no-llm-judge to target heuristic-only rows, or --force to overwrite all)."
        )
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
        results: list[tuple[str, Optional[dict[str, object]], Optional[str]]] = []

        sem = asyncio.Semaphore(max_conc)
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
                    last_scored = None
                    for attempt in range(max_parse_error_retries):
                        async with sem:
                            last_scored = await judge.judge(ji)
                        if not _is_parse_error(last_scored):
                            break
                        if attempt < max_parse_error_retries - 1:
                            await asyncio.sleep(0.5)
                    results.append((output_id, last_scored, None))
                except Exception as e:
                    results.append((output_id, None, str(e)))

            tasks = [asyncio.create_task(score_one(r)) for r in rows]
            for idx, fut in enumerate(asyncio.as_completed(tasks), start=1):
                await fut
                if idx % 25 == 0 or idx == len(tasks):
                    print(f"Judged {idx}/{len(tasks)}...")

        return results

    scored_rows = asyncio.run(_score_all())
    parse_error_skipped = 0
    for output_id, scored, err in scored_rows:
        if scored is None:
            failed += 1
            print(f"Warning: Judge scoring failed for output_id={output_id[:8]}: {err}")
            continue
        if _is_parse_error(scored):
            parse_error_skipped += 1
            print(
                f"Warning: parse_error after {max_parse_error_retries} retries for "
                f"output_id={output_id[:8]}; row left un-updated (will retry on next run)."
            )
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
    if parse_error_skipped:
        print(f"judgeval_parse_error_skipped={parse_error_skipped} (re-run to retry these rows)")
    if failed:
        print(f"judgeval_failed={failed}")
    return 0


def _handle_logit_lens(args: Any) -> int:
    from vivarium.experiments.olmo_conformity.logit_lens import (
        analyze_think_rationalization,
        compute_logit_lens_for_think_tokens,
        compute_logit_lens_topk_for_trials,
        parse_and_store_think_tokens,
    )

    trace_db = TraceDb(TraceDbConfig(db_path=str(args.db)))
    trace_db.connect()
    trace_db.init_schema()
    trace_db.init_conformity_schema()
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

    total = 0
    think_total = 0
    think_analysis_total = 0
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


def _handle_posthoc(args: Any) -> int:
    import sys

    from vivarium.experiments.olmo_conformity.activation_patching import (
        plot_activation_patching_heatmap,
        run_activation_patching,
    )
    from vivarium.experiments.olmo_conformity.analysis import generate_core_figures
    from vivarium.experiments.olmo_conformity.answer_logprobs import compute_and_store_answer_logprobs_for_model
    from vivarium.experiments.olmo_conformity.contrastive_steering import (
        compute_deference_vector,
        plot_contrastive_steering_results,
        run_contrastive_steering_test,
    )
    from vivarium.experiments.olmo_conformity.intervention import run_intervention_sweep
    from vivarium.experiments.olmo_conformity.logit_lens import (
        compute_logit_lens_topk_for_trials,
        compute_logit_lens_tug_of_war_for_run,
        parse_and_store_think_tokens,
        plot_logit_lens_tug_of_war,
    )

    run_dir = str(args.run_dir)
    run_base = os.path.basename(run_dir.rstrip("/"))
    run_id = str(args.run_id) if args.run_id else (run_base.split("_")[-1] if "_" in run_base else run_base)
    db_path = str(args.db) if args.db else os.path.join(run_dir, "simulation.db")

    trace_db = TraceDb(TraceDbConfig(db_path=str(db_path)))
    trace_db.connect()
    trace_db.init_schema()
    trace_db.init_conformity_schema()

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

    if bool(args.clear_existing) and trial_ids:
        trace_db.conn.execute(
            f"DELETE FROM conformity_logit_lens WHERE trial_id IN ({','.join(['?']*len(trial_ids))});",
            trial_ids,
        )
        trace_db.conn.execute(
            f"DELETE FROM vivarium_think_tokens WHERE trial_id IN ({','.join(['?']*len(trial_ids))});",
            trial_ids,
        )
        trace_db.conn.execute(
            f"DELETE FROM vivarium_answer_logprobs WHERE trial_id IN ({','.join(['?']*len(trial_ids))});",
            trial_ids,
        )
        trace_db.conn.execute(
            """
            DELETE FROM vivarium_intervention_results
            WHERE intervention_id IN (SELECT intervention_id FROM vivarium_interventions WHERE run_id = ?);
            """,
            (run_id,),
        )
        trace_db.conn.execute("DELETE FROM vivarium_interventions WHERE run_id = ?;", (run_id,))
        trace_db.conn.execute(
            f"DELETE FROM vivarium_logit_lens_tug_of_war WHERE trial_id IN ({','.join(['?']*len(trial_ids))});",
            trial_ids,
        )
        trace_db.conn.execute("DELETE FROM vivarium_contrastive_steering WHERE run_id = ?;", (run_id,))
        trace_db.conn.execute("DELETE FROM vivarium_activation_patching WHERE run_id = ?;", (run_id,))
        trace_db.conn.commit()

    think_inserted = 0
    if bool(args.parse_think_tokens):
        for tid in trial_ids:
            think_inserted += parse_and_store_think_tokens(trace_db=trace_db, trial_id=str(tid))

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

    answer_inserted = 0
    answer_errors = 0
    if not bool(args.no_answer_logprobs):
        groups: Dict[str, List[str]] = {}
        if trial_ids:
            rows = trace_db.conn.execute(
                f"SELECT trial_id, model_id FROM conformity_trials WHERE trial_id IN ({','.join(['?']*len(trial_ids))});",
                trial_ids,
            ).fetchall()
            for r in rows:
                mid = str(r["model_id"])
                groups.setdefault(mid, []).append(str(r["trial_id"]))

        for mid, tids in groups.items():
            res = compute_and_store_answer_logprobs_for_model(
                trace_db=trace_db,
                run_id=str(run_id),
                model_id=str(mid),
                trial_ids=list(tids),
                include_empty_think=True,
                include_observed_think=True,
                include_alternate_answer=True,
                skip_existing=(not bool(args.clear_existing)),
            )
            answer_inserted += int(res.get("inserted") or 0)
            answer_errors += int(res.get("errors") or 0)

    intervention_inserted = 0
    if not bool(args.no_interventions):
        sp = trace_db.conn.execute(
            """
            SELECT probe_id, artifact_path
            FROM vivarium_probes
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
                    "AND t.condition_id IN (SELECT condition_id FROM conformity_conditions WHERE name != 'control' AND name NOT LIKE '%probe_capture%')"
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

    # --- Logit Lens Tug-of-War ---
    tow_stats: Dict[str, Any] = {"skipped": True}
    if not bool(args.no_logit_lens_tug_of_war):
        try:
            tow_stats = compute_logit_lens_tug_of_war_for_run(
                trace_db=trace_db,
                run_id=run_id,
                model_id=str(model_id),
                layers=layers,
                skip_existing=(not bool(args.clear_existing)),
            )
            plot_logit_lens_tug_of_war(
                trace_db=trace_db,
                run_id=run_id,
                output_dir=run_dir,
                model_id=str(model_id),
            )
        except Exception as e:
            print(f"Warning: logit lens tug-of-war failed: {e}", file=sys.stderr)

    # --- Contrastive Vector Steering ---
    steering_inserted = 0
    if not bool(args.no_contrastive_steering):
        try:
            steering_layers = [int(x) for x in str(args.steering_layers).split(",") if str(x).strip() != ""]
            steering_alphas = [float(x) for x in str(args.steering_alphas).split(",") if str(x).strip() != ""]
            steering_min_pairs = int(args.steering_min_pairs)
            artifacts_dir = os.path.join(run_dir, "artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)

            vec_result = compute_deference_vector(
                trace_db=trace_db,
                run_id=run_id,
                model_id=str(model_id),
                layers=steering_layers,
                output_dir=os.path.join(artifacts_dir, "interpretability", "contrastive_steering"),
                min_pairs=steering_min_pairs,
            )
            if not vec_result.get("skipped"):
                steering_inserted = run_contrastive_steering_test(
                    trace_db=trace_db,
                    run_id=run_id,
                    model_id=str(model_id),
                    vector_paths=vec_result["vector_paths"],
                    alpha_values=steering_alphas,
                    max_new_tokens=int(args.max_new_tokens),
                )
                plot_contrastive_steering_results(
                    trace_db=trace_db,
                    run_id=run_id,
                    output_dir=run_dir,
                )
            else:
                print(f"Warning: contrastive steering skipped: {vec_result.get('reason')}")
        except Exception as e:
            print(f"Warning: contrastive steering failed: {e}", file=sys.stderr)

    # --- Activation Patching ---
    patching_inserted = 0
    if not bool(args.no_activation_patching):
        try:
            patching_layers_str = str(args.patching_layers) if args.patching_layers else str(args.layers)
            patching_layers = [int(x) for x in patching_layers_str.split(",") if str(x).strip() != ""]
            patching_inserted = run_activation_patching(
                trace_db=trace_db,
                run_id=run_id,
                model_id=str(model_id),
                layers=patching_layers,
                max_new_tokens=int(args.max_new_tokens),
            )
            plot_activation_patching_heatmap(
                trace_db=trace_db,
                run_id=run_id,
                output_dir=run_dir,
            )
        except Exception as e:
            print(f"Warning: activation patching failed: {e}", file=sys.stderr)

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
    if not bool(args.no_answer_logprobs):
        print(f"answer_logprobs_inserted={answer_inserted} (errors={answer_errors})")
    if bool(args.parse_think_tokens):
        print(f"think_tokens_inserted={think_inserted}")
    print(f"intervention_results_inserted={intervention_inserted}")
    if not bool(args.no_logit_lens_tug_of_war):
        print(f"tug_of_war={tow_stats}")
    if not bool(args.no_contrastive_steering):
        print(f"contrastive_steering_inserted={steering_inserted}")
    if not bool(args.no_activation_patching):
        print(f"activation_patching_inserted={patching_inserted}")
    return 0


def _handle_intervene(args: Any) -> int:
    from vivarium.experiments.olmo_conformity.intervention import run_intervention_sweep

    trace_db = TraceDb(TraceDbConfig(db_path=str(args.db)))
    trace_db.connect()
    trace_db.init_schema()
    trace_db.init_conformity_schema()
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


def _handle_vector_analysis(args: Any) -> int:
    from vivarium.experiments.olmo_conformity.vector_analysis import run_truth_social_vector_analysis

    trace_db = TraceDb(TraceDbConfig(db_path=str(args.db)))
    trace_db.connect()
    trace_db.init_schema()
    trace_db.init_conformity_schema()

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

    print("\n" + "=" * 60)
    print("Vector Analysis Results")
    print("=" * 60)
    print(f"Truth Probe ID: {results['truth_probe_id']}")
    if results["social_probe_id"]:
        print(f"Social Probe ID: {results['social_probe_id']}")
    print(f"Projection Stats: {results['projection_stats']}")
    print(f"Turn Layers: {results['turn_layers']}")
    print(f"Analysis Artifacts: {results['analysis_artifacts']}")
    return 0


def _handle_resume(args: Any) -> int:
    from vivarium.experiments.olmo_conformity.resume import resume_from_projections

    trace_db = TraceDb(TraceDbConfig(db_path=str(args.db)))
    trace_db.connect()
    trace_db.init_schema()
    trace_db.init_conformity_schema()

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


def _handle_full(args: Any) -> int:
    from vivarium.experiments.olmo_conformity.io import load_suite_config
    from vivarium.experiments.olmo_conformity.orchestration import ExperimentConfig, run_full_experiment

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


def _handle_backfill_behavioral(args: Any) -> int:
    from vivarium.experiments.olmo_conformity.backfill_behavioral import run_backfill

    result = run_backfill(
        runs_dir=str(args.runs_dir),
        run_id=str(args.run_id) if args.run_id else None,
        api_base=args.api_base,
        api_key=args.api_key,
        dry_run=bool(args.dry_run),
    )
    print(f"\nBackfill complete: {result['total_backfilled']} outputs stored, {result['total_errors']} errors")
    return 0 if result["total_errors"] == 0 else 1


def _handle_complete_suite(args: Any) -> int:
    from vivarium.experiments.olmo_conformity.suite_completion import run_suite_completion

    result = run_suite_completion(
        runs_dir=str(args.runs_dir),
        metadata_path=str(args.metadata),
        run_id=str(args.run_id) if args.run_id else None,
        api_base=args.api_base,
        api_key=args.api_key,
        dry_run=bool(args.dry_run),
        behavioral_only=True,
    )
    print(
        f"\nSuite completion: {result['total_trials_created']} trials created, "
        f"{result['total_outputs_backfilled']} outputs backfilled, "
        f"{result['total_errors']} errors"
    )
    return 0 if result["total_errors"] == 0 else 1
