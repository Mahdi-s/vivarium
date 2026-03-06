"""Backfill missing behavioral outputs for trials that have no conformity_outputs row."""

from __future__ import annotations

import gc
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from vivarium.llm_gateway import HuggingFaceHookedGateway, LiteLLMGateway
from vivarium.output_parsing import OutputParsingConfig, classify_output
from vivarium.persistence import TraceDb, TraceDbConfig
from vivarium.settings import settings

from .olmo_utils import (
    detect_olmo_variant,
    ensure_olmo_model_downloaded,
    get_olmo_model_config,
    get_ollama_model_name,
)
from .prompts import build_messages
from .scoring import evaluate_correctness, is_refusal, parse_answer_text


def _discover_runs(runs_dir: str, run_id_filter: Optional[str] = None) -> List[str]:
    """Return paths to run directories containing simulation.db."""
    if not os.path.isdir(runs_dir):
        return []
    result: List[str] = []
    for name in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, name)
        if not os.path.isdir(run_dir):
            continue
        db_path = os.path.join(run_dir, "simulation.db")
        if not os.path.isfile(db_path):
            continue
        if run_id_filter:
            suffix = f"_{run_id_filter}"
            if not name.endswith(suffix):
                continue
        result.append(run_dir)
    return result


def _create_gateway(
    *,
    model_id: str,
    variant: str,
    api_base: Optional[str],
    api_key: Optional[str],
    hf_cache_dir: Optional[str],
    repo_root: str,
) -> tuple[Any, str]:
    """
    Create gateway for model inference. Returns (gateway, model_id_for_api).
    """
    model_config = get_olmo_model_config(model_id) if model_id else {}
    max_tokens = int(model_config.get("max_new_tokens", 128))

    if variant == "unknown" and model_id:
        variant = detect_olmo_variant(model_id)

    if api_base and model_id and "allenai/Olmo" in model_id:
        olmo_model_name = get_ollama_model_name(model_id)
        gateway = LiteLLMGateway(
            api_base=api_base,
            api_key=api_key,
            rate_limit_config=None,
        )
        return gateway, olmo_model_name

    # Local HuggingFace
    if hf_cache_dir:
        model_cache_path = os.path.join(str(hf_cache_dir), model_id.replace("/", "_"))
    else:
        model_cache_path = os.path.join(
            repo_root, "models", "huggingface_cache", model_id.replace("/", "_")
        )
    if not os.path.isdir(model_cache_path):
        models_dir_for_download = str(Path(model_cache_path).parent) if hf_cache_dir else None
        if models_dir_for_download:
            ensure_olmo_model_downloaded(
                model_id=model_id,
                models_dir=models_dir_for_download,
                import_to_ollama=False,
            )
        model_cache_path = model_id  # fallback to model_id for download

    gateway = HuggingFaceHookedGateway(
        model_id_or_path=model_cache_path if os.path.isdir(model_cache_path) else model_id,
        device=os.environ.get("VVM_DEVICE"),
        capture_context=None,
        max_new_tokens=max_tokens,
    )
    return gateway, model_id


def run_backfill(
    *,
    runs_dir: str,
    run_id: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Backfill missing behavioral outputs for runs under runs_dir.

    For each run, finds trials that have no conformity_outputs row (behavioral
    conditions only), runs inference, and stores outputs.

    Returns dict with per-run stats: {run_dir: {backfilled: N, skipped: M, errors: [...]}}
    """
    run_dirs = _discover_runs(runs_dir, run_id_filter=run_id)
    if not run_dirs:
        return {"runs": [], "total_backfilled": 0, "total_errors": 0}

    repo_root = str(settings.PROJECT_ROOT)
    hf_cache_dir = (
        os.environ.get("VIVARIUM_HF_CACHE")
        or os.environ.get("AAM_HF_CACHE")
        or os.environ.get("VIVARIUM_MODEL_DIR")
        or os.environ.get("AAM_MODEL_DIR")
    )

    output_parse_cfg = OutputParsingConfig()

    def _load_paths_config() -> Dict[str, str]:
        try:
            from .io import load_paths_config
            from .io import load_suite_config
            # Try to find a suite config to get paths
            suite_path = Path(repo_root) / "experiments" / "olmo_conformity" / "configs" / "test_scripts" / "suite_expanded_localtest_by_model.json"
            if suite_path.exists():
                cfg = load_suite_config(str(suite_path))
                return load_paths_config(str(suite_path), cfg)
        except Exception:
            pass
        return {}

    paths_cfg = _load_paths_config()
    if not hf_cache_dir and paths_cfg.get("models_dir"):
        hf_cache_dir = str(paths_cfg["models_dir"])

    results: Dict[str, Dict[str, Any]] = {}
    total_backfilled = 0
    total_errors = 0

    for run_dir in run_dirs:
        db_path = os.path.join(run_dir, "simulation.db")
        run_result: Dict[str, Any] = {"backfilled": 0, "skipped": 0, "errors": []}
        results[run_dir] = run_result

        trace_db = TraceDb(TraceDbConfig(db_path=db_path))
        trace_db.connect()
        trace_db.init_schema()
        trace_db.init_conformity_schema()

        try:
            row = trace_db.conn.execute(
                "SELECT run_id FROM runs LIMIT 1;"
            ).fetchone()
            if not row:
                run_result["errors"].append("No runs row in DB")
                trace_db.close()
                continue
            db_run_id = str(row["run_id"])

            missing = trace_db.conn.execute(
                """
                SELECT
                    t.trial_id,
                    t.model_id,
                    t.variant,
                    t.temperature,
                    t.seed,
                    t.item_id,
                    i.ground_truth_text,
                    i.source_json
                FROM conformity_trials t
                JOIN conformity_items i ON i.item_id = t.item_id
                LEFT JOIN conformity_outputs o ON o.trial_id = t.trial_id
                WHERE t.run_id = ?
                  AND o.output_id IS NULL
                  AND t.condition_id IN (
                    SELECT condition_id FROM conformity_conditions
                    WHERE name IN ('control', 'asch_history_5', 'authoritative_bias')
                  )
                ORDER BY t.model_id, t.created_at;
                """,
                (db_run_id,),
            ).fetchall()

            if not missing:
                run_result["skipped"] = 0
                trace_db.close()
                continue

            if dry_run:
                run_result["skipped"] = len(missing)
                print(f"[Backfill] {run_dir}: would backfill {len(missing)} trials (dry-run)")
                trace_db.close()
                continue

            # Load prompts for missing trials
            trial_data: Dict[str, Dict[str, Any]] = {}
            for tr in missing:
                tid = str(tr["trial_id"])
                prow = trace_db.conn.execute(
                    """
                    SELECT system_prompt, user_prompt, chat_history_json
                    FROM conformity_prompts
                    WHERE trial_id = ?
                    ORDER BY created_at ASC LIMIT 1;
                    """,
                    (tid,),
                ).fetchone()
                if not prow:
                    run_result["errors"].append(f"Trial {tid[:8]}... missing prompt")
                    continue
                history: List[Dict[str, Any]] = []
                raw_hist = prow["chat_history_json"]
                if raw_hist:
                    try:
                        history = json.loads(raw_hist)
                    except Exception:
                        pass
                trial_data[tid] = {
                    "model_id": str(tr["model_id"]),
                    "variant": str(tr["variant"]),
                    "temperature": float(tr["temperature"]),
                    "seed": int(tr["seed"]),
                    "ground_truth_text": tr["ground_truth_text"],
                    "source_json": tr["source_json"],
                    "system_prompt": str(prow["system_prompt"] or ""),
                    "user_prompt": str(prow["user_prompt"] or ""),
                    "history": history,
                }

            # Trial metadata for top_k, top_p
            meta_by_trial: Dict[str, Dict] = {}
            if trial_data:
                placeholders = ",".join("?" * len(trial_data))
                meta_rows = trace_db.conn.execute(
                    f"""
                    SELECT trial_id, metadata_json
                    FROM conformity_trial_metadata
                    WHERE trial_id IN ({placeholders});
                    """,
                    list(trial_data.keys()),
                ).fetchall()
                for mr in meta_rows:
                    tid = str(mr["trial_id"])
                    try:
                        meta_by_trial[tid] = json.loads(mr["metadata_json"] or "{}")
                    except Exception:
                        meta_by_trial[tid] = {}
            for tid, data in trial_data.items():
                gen = meta_by_trial.get(tid, {}).get("generation", {})
                data["top_k"] = gen.get("top_k")
                data["top_p"] = gen.get("top_p")
                if data["top_k"] is not None and data["top_k"] <= 0:
                    data["top_k"] = None
                if data["top_p"] is not None and (data["top_p"] <= 0 or data["top_p"] > 1):
                    data["top_p"] = None

            # Group by model_id
            by_model: Dict[str, List[str]] = {}
            for tid, data in trial_data.items():
                mid = data["model_id"]
                by_model.setdefault(mid, []).append(tid)

            gateway_cache: Dict[str, tuple[Any, str]] = {}

            for mid, tids in by_model.items():
                if mid not in gateway_cache:
                    try:
                        variant = trial_data[tids[0]]["variant"]
                        gw, api_id = _create_gateway(
                            model_id=mid,
                            variant=variant,
                            api_base=api_base,
                            api_key=api_key,
                            hf_cache_dir=hf_cache_dir,
                            repo_root=repo_root,
                        )
                        gateway_cache[mid] = (gw, api_id)
                    except Exception as e:
                        run_result["errors"].append(f"Model {mid}: {e}")
                        continue

                gateway, model_id_for_api = gateway_cache[mid]

                for tid in tids:
                    data = trial_data[tid]
                    msgs = build_messages(
                        system=data["system_prompt"],
                        user=data["user_prompt"],
                        history=data["history"],
                    )
                    try:
                        t0 = time.time()
                        resp = gateway.chat(
                            model=model_id_for_api,
                            messages=msgs,
                            tools=None,
                            tool_choice=None,
                            temperature=data["temperature"],
                            top_k=data.get("top_k"),
                            top_p=data.get("top_p"),
                            seed=data["seed"],
                        )
                        latency_ms = (time.time() - t0) * 1000.0
                    except Exception as e:
                        run_result["errors"].append(f"Trial {tid[:8]}: {e}")
                        total_errors += 1
                        continue

                    raw_text = ""
                    try:
                        raw_text = str(resp["choices"][0]["message"].get("content") or "")
                    except Exception:
                        raw_text = str(resp)

                    gt = data["ground_truth_text"]
                    expected = [str(gt)] if gt is not None else []
                    classified = classify_output(
                        raw_text=raw_text,
                        cfg=output_parse_cfg,
                        system_prompt=data["system_prompt"],
                        user_prompt=data["user_prompt"],
                        expected_answer_texts=expected,
                        token_logprobs=None,
                    )
                    parsed = parse_answer_text(raw_text)
                    refusal = is_refusal(raw_text)
                    is_correct = evaluate_correctness(
                        parsed_answer_text=parsed,
                        ground_truth_text=gt,
                    )

                    output_id = str(uuid.uuid4())
                    trace_db.insert_conformity_output(
                        output_id=output_id,
                        trial_id=tid,
                        raw_text=raw_text,
                        parsed_answer_text=parsed,
                        parsed_answer_json=None,
                        is_correct=is_correct,
                        refusal_flag=refusal,
                        latency_ms=latency_ms,
                        token_usage_json={
                            "_output_quality": {
                                "label": classified.label.value,
                                "metadata": classified.metadata,
                            }
                        },
                    )
                    run_result["backfilled"] += 1
                    total_backfilled += 1

                # Cleanup model between model groups
                if mid in gateway_cache:
                    try:
                        import torch
                        gw = gateway_cache[mid][0]
                        if hasattr(gw, "_model"):
                            del gw._model
                        if hasattr(gw, "_tokenizer"):
                            del gw._tokenizer
                        del gateway_cache[mid]
                        gc.collect()
                        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                            torch.mps.empty_cache()
                    except Exception:
                        pass

            print(f"[Backfill] {run_dir}: backfilled {run_result['backfilled']} trials")

        finally:
            trace_db.close()

    return {
        "runs": results,
        "total_backfilled": total_backfilled,
        "total_errors": total_errors,
    }
