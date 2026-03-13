"""Suite completion: complete missing trials/outputs per suite_expanded_temp{T}.json definition."""

from __future__ import annotations

import gc
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from vivarium.llm_gateway import create_gateway
from vivarium.output_parsing import OutputParsingConfig, classify_output
from vivarium.persistence import TraceDb, TraceDbConfig
from vivarium.settings import settings

from .io import clamp_items, deterministic_prompt_hash, load_paths_config, load_suite_config, read_jsonl, sha256_file
from .olmo_utils import ensure_olmo_model_downloaded, get_olmo_model_config
from .prompts import PROMPT_RENDERER_VERSION, build_messages
from .runner import _build_prompt_for_condition
from .enhanced_scoring import score_single_output

BEHAVIORAL_CONDITIONS = ("control", "asch_history_5", "authoritative_bias")
DEFAULT_TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
CONFIG_TEMPLATE = "suite_expanded_temp{temp}.json"


def _discover_run_dir(runs_dir: str, run_id: str) -> Optional[str]:
    """Return path to run directory whose name ends with _<run_id>, or None."""
    if not run_id or not os.path.isdir(runs_dir):
        return None
    suffix = f"_{run_id}"
    for name in os.listdir(runs_dir):
        if name.endswith(suffix) and os.path.isdir(os.path.join(runs_dir, name)):
            return os.path.join(runs_dir, name)
    return None


def _make_gateway(
    *,
    model_id: str,
    variant: str,
    api_base: Optional[str],
    api_key: Optional[str],
    hf_cache_dir: Optional[str],
) -> Tuple[Any, str]:
    """Create gateway for model inference. Delegates to shared factory."""
    model_config = get_olmo_model_config(model_id) if model_id else {}
    max_tokens = int(model_config.get("max_new_tokens", 128))

    # Pre-flight: verify local OLMo models
    if not api_base and model_id.startswith("allenai/"):
        models_dir = str(Path(str(hf_cache_dir)).parent) if hf_cache_dir else None
        ensure_olmo_model_downloaded(model_id=model_id, models_dir=models_dir, import_to_ollama=False)

    return create_gateway(
        model_id=model_id,
        variant=variant,
        api_base=api_base,
        api_key=api_key,
        hf_cache_dir=hf_cache_dir,
        max_new_tokens=max_tokens,
    )


def _ensure_items_and_get_dataset_ids(
    trace_db: TraceDb,
    suite_config: Dict[str, Any],
    suite_dir: Path,
    repo_root: str,
) -> Dict[str, str]:
    """Register datasets and add missing items. Return dataset_ids (name -> id)."""
    dataset_ids: Dict[str, str] = {}

    def _resolve_data_path(path_str: str) -> str:
        p = Path(str(path_str))
        if p.is_absolute():
            return str(p)
        suite_rel = (suite_dir / p).resolve()
        if suite_rel.exists():
            return str(suite_rel)
        return str((Path(repo_root) / p).resolve())

    for ds in suite_config.get("datasets", []):
        name = str(ds["name"])
        version = str(ds.get("version", "v0"))
        rel_path = str(ds["path"])
        abs_path = _resolve_data_path(rel_path)

        row = trace_db.conn.execute(
            "SELECT dataset_id FROM conformity_datasets WHERE name = ? AND version = ?;",
            (name, version),
        ).fetchone()
        dataset_id = str(row["dataset_id"]) if row else str(uuid.uuid4())
        dataset_ids[name] = dataset_id

        trace_db.upsert_conformity_dataset(
            dataset_id=dataset_id,
            name=name,
            version=version,
            path=rel_path,
            sha256=sha256_file(abs_path),
        )

        items = clamp_items(
            read_jsonl(abs_path),
            suite_config.get("run", {}).get("max_items_per_dataset"),
        )
        for it in items:
            item_id = str(it.get("item_id") or str(uuid.uuid4()))
            source_data = it.get("source") if isinstance(it.get("source"), dict) else {}
            if it.get("wrong_answer"):
                source_data["wrong_answer"] = str(it["wrong_answer"])

            existing = trace_db.conn.execute(
                "SELECT 1 FROM conformity_items WHERE item_id = ? AND dataset_id = ?;",
                (item_id, dataset_id),
            ).fetchone()
            if existing:
                continue

            trace_db.insert_conformity_item(
                item_id=item_id,
                dataset_id=dataset_id,
                domain=str(it.get("domain") or "unknown"),
                question=str(it.get("question") or ""),
                ground_truth_text=(
                    None if it.get("ground_truth_text") is None
                    else str(it.get("ground_truth_text"))
                ),
                ground_truth_json=(
                    it.get("ground_truth_json")
                    if isinstance(it.get("ground_truth_json"), dict)
                    else None
                ),
                source_json=source_data if source_data else None,
            )

    return dataset_ids


def _ensure_behavioral_conditions(
    trace_db: TraceDb, suite_config: Dict[str, Any]
) -> Dict[str, str]:
    """Ensure behavioral conditions exist; return condition_id for each."""
    cond_by_name: Dict[str, Dict[str, Any]] = {}
    for c in suite_config.get("conditions", []):
        name = str(c.get("name") or "")
        if name in BEHAVIORAL_CONDITIONS:
            cond_by_name[name] = c

    out: Dict[str, str] = {}
    for name in BEHAVIORAL_CONDITIONS:
        row = trace_db.conn.execute(
            "SELECT condition_id FROM conformity_conditions WHERE name = ?;",
            (name,),
        ).fetchone()
        if row:
            out[name] = str(row["condition_id"])
        elif name in cond_by_name:
            cond_id = str(uuid.uuid4())
            cond = cond_by_name[name]
            trace_db.upsert_conformity_condition(
                condition_id=cond_id,
                name=name,
                params=dict(cond.get("params") or {}),
            )
            out[name] = cond_id
    return out


def complete_single_run(
    *,
    db_path: str,
    run_id: str,
    suite_config: Dict[str, Any],
    suite_config_path: str,
    api_base: Optional[str],
    api_key: Optional[str],
    hf_cache_dir: Optional[str],
    dry_run: bool,
    behavioral_only: bool,
) -> Dict[str, Any]:
    """
    Complete missing trials and outputs for a single run.

    Returns dict with keys: items_added, trials_created, outputs_backfilled, errors.
    """
    repo_root = str(settings.PROJECT_ROOT)
    suite_dir = Path(suite_config_path).resolve().parent

    prompts_root = str(Path(repo_root) / "experiments" / "olmo_conformity" / "prompts")
    cursor = Path(suite_config_path).resolve().parent
    while cursor != cursor.parent:
        candidate = cursor.parent / "prompts"
        if candidate.is_dir():
            prompts_root = str(candidate)
            break
        cursor = cursor.parent

    output_parse_cfg = OutputParsingConfig()
    result: Dict[str, Any] = {
        "items_added": 0,
        "trials_created": 0,
        "outputs_backfilled": 0,
        "errors": [],
    }

    trace_db = TraceDb(TraceDbConfig(db_path=db_path))
    trace_db.connect()
    trace_db.init_schema()
    trace_db.init_conformity_schema()

    try:
        db_run_id_row = trace_db.conn.execute(
            "SELECT run_id FROM runs LIMIT 1;"
        ).fetchone()
        if not db_run_id_row:
            result["errors"].append("No runs row in DB")
            trace_db.close()
            return result

        db_run_id = str(db_run_id_row["run_id"])
        temperature = float(suite_config.get("run", {}).get("temperature", 0.0))
        seed = int(suite_config.get("run", {}).get("seed", 42))
        top_k_raw = suite_config.get("run", {}).get("top_k", suite_config.get("run", {}).get("top_n"))
        top_p_raw = suite_config.get("run", {}).get("top_p", suite_config.get("run", {}).get("nucleus_p"))

        top_k: Optional[int] = None
        try:
            top_k = None if top_k_raw is None else int(top_k_raw)
        except Exception:
            top_k = None
        if top_k is not None and top_k <= 0:
            top_k = None

        top_p: Optional[float] = None
        try:
            top_p = None if top_p_raw is None else float(top_p_raw)
        except Exception:
            top_p = None
        if top_p is not None and not (0.0 < top_p <= 1.0):
            top_p = None

        dataset_ids = _ensure_items_and_get_dataset_ids(
            trace_db, suite_config, suite_dir, repo_root
        )
        cond_ids = _ensure_behavioral_conditions(trace_db, suite_config)
        if len(cond_ids) < 3:
            result["errors"].append(
                f"Missing behavioral conditions: need control, asch_history_5, authoritative_bias"
            )
            trace_db.close()
            return result

        dataset_id_set = set(dataset_ids.values())
        placeholders = ",".join("?" * len(dataset_id_set))
        rows = trace_db.conn.execute(
            f"""
            SELECT item_id, dataset_id, domain, question, ground_truth_text, source_json
            FROM conformity_items
            WHERE dataset_id IN ({placeholders})
            ORDER BY dataset_id, item_id;
            """,
            list(dataset_id_set),
        ).fetchall()

        distractor_pool_by_dataset: Dict[str, List[str]] = {}
        global_distractor_pool: List[str] = []
        for r in rows:
            dsid = str(r["dataset_id"] or "unknown_dataset")
            wrong = None
            src = r["source_json"]
            if src:
                try:
                    src_data = json.loads(src)
                    wrong = src_data.get("wrong_answer")
                except Exception:
                    wrong = None
            if wrong:
                distractor_pool_by_dataset.setdefault(dsid, []).append(str(wrong))
                global_distractor_pool.append(str(wrong))

        existing_trials = trace_db.conn.execute(
            """
            SELECT t.trial_id, t.model_id, t.variant, t.item_id, t.condition_id, o.output_id
            FROM conformity_trials t
            LEFT JOIN conformity_outputs o ON o.trial_id = t.trial_id
            WHERE t.run_id = ?;
            """,
            (db_run_id,),
        ).fetchall()

        existing_with_output: Set[Tuple[str, str, str]] = set()
        existing_without_output: Dict[Tuple[str, str, str], str] = {}
        for r in existing_trials:
            key = (str(r["model_id"]), str(r["item_id"]), str(r["condition_id"]))
            if r["output_id"] is not None:
                existing_with_output.add(key)
            else:
                existing_without_output[key] = str(r["trial_id"])

        expected: Set[Tuple[str, str, str, str, str]] = set()
        for m in suite_config.get("models", []):
            model_id = str(m.get("model_id") or "mock")
            variant = str(m.get("variant") or "unknown")
            for row in rows:
                item_id = str(row["item_id"])
                for cond_name, cond_id in cond_ids.items():
                    expected.add((model_id, variant, item_id, cond_id, cond_name))

        to_create: List[Tuple[str, str, str, str, str]] = []
        for (model_id, variant, item_id, cond_id, cond_name) in expected:
            key = (model_id, item_id, cond_id)
            if key in existing_with_output:
                continue
            if key in existing_without_output:
                continue
            to_create.append((model_id, variant, item_id, cond_id, cond_name))

        to_backfill: List[str] = []
        for (model_id, item_id, cond_id) in existing_without_output:
            if (model_id, item_id, cond_id) not in existing_with_output:
                to_backfill.append(existing_without_output[(model_id, item_id, cond_id)])

        if dry_run:
            print(
                f"[Complete] {db_path}: would create {len(to_create)} trials, "
                f"backfill {len(to_backfill)} outputs (dry-run)"
            )
            trace_db.close()
            result["trials_created"] = len(to_create)
            result["outputs_backfilled"] = len(to_backfill)
            return result

        item_by_id: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            item_id = str(r["item_id"])
            item = {
                "item_id": item_id,
                "dataset_id": r["dataset_id"],
                "domain": r["domain"],
                "question": r["question"],
                "ground_truth_text": r["ground_truth_text"],
                "_run_seed": seed,
                "_distractor_pool": distractor_pool_by_dataset.get(str(r["dataset_id"]), []),
                "_global_distractor_pool": global_distractor_pool,
            }
            if r["source_json"]:
                try:
                    src = json.loads(r["source_json"])
                    if src.get("wrong_answer"):
                        item["wrong_answer"] = src["wrong_answer"]
                except Exception:
                    pass
            item_by_id[item_id] = item

        for (model_id, variant, item_id, cond_id, cond_name) in to_create:
            item = item_by_id.get(item_id)
            if not item:
                result["errors"].append(f"Item {item_id} not found")
                continue

            cond_row = trace_db.conn.execute(
                "SELECT params_json FROM conformity_conditions WHERE condition_id = ?;",
                (cond_id,),
            ).fetchone()
            params = {}
            if cond_row and cond_row["params_json"]:
                try:
                    params = json.loads(cond_row["params_json"])
                except Exception:
                    pass
            condition = {"name": cond_name, "params": params}

            try:
                system, user, history, prompt_meta = _build_prompt_for_condition(
                    condition=condition,
                    item=item,
                    prompts_root=prompts_root,
                )
            except Exception as e:
                result["errors"].append(f"Prompt build failed for {item_id}/{cond_name}: {e}")
                continue

            trial_id = str(uuid.uuid4())
            trace_db.insert_conformity_trial(
                trial_id=trial_id,
                run_id=db_run_id,
                model_id=model_id,
                variant=variant,
                item_id=item_id,
                condition_id=cond_id,
                seed=seed,
                temperature=temperature,
            )

            model_config = get_olmo_model_config(model_id) if model_id != "mock" else {}
            trace_db.upsert_conformity_trial_metadata(
                trial_id=trial_id,
                metadata={
                    "prompt_renderer_version": PROMPT_RENDERER_VERSION,
                    "suite_name": str(suite_config.get("suite_name", "")),
                    "suite_version": str(suite_config.get("suite_version", "")),
                    "generation": {
                        "seed": seed,
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                    },
                    "model": {"model_id": model_id, "variant": variant, "model_config": model_config},
                },
            )

            prompt_hash = deterministic_prompt_hash(system=system, user=user, history=history)
            prompt_id = str(uuid.uuid4())
            trace_db.insert_conformity_prompt(
                prompt_id=prompt_id,
                trial_id=trial_id,
                system_prompt=system,
                user_prompt=user,
                chat_history=history,
                rendered_prompt_hash=prompt_hash,
            )

            meta_to_store = dict(prompt_meta or {})
            meta_to_store.update({"prompt_id": prompt_id, "trial_id": trial_id, "rendered_prompt_hash": prompt_hash})
            try:
                trace_db.upsert_conformity_prompt_metadata(prompt_id=prompt_id, metadata=meta_to_store)
            except Exception:
                pass

            to_backfill.append(trial_id)
            result["trials_created"] += 1

        if not to_backfill:
            trace_db.close()
            return result

        trial_data: Dict[str, Dict[str, Any]] = {}
        for tid in to_backfill:
            tr = trace_db.conn.execute(
                """
                SELECT t.trial_id, t.model_id, t.variant, t.temperature, t.seed, t.item_id,
                       i.ground_truth_text, i.source_json,
                       c.name AS condition_name,
                       d.name AS dataset_name
                FROM conformity_trials t
                JOIN conformity_items i ON i.item_id = t.item_id
                JOIN conformity_conditions c ON c.condition_id = t.condition_id
                JOIN conformity_datasets d ON d.dataset_id = i.dataset_id
                WHERE t.trial_id = ?
                """,
                (tid,),
            ).fetchone()
            if not tr:
                continue
            prow = trace_db.conn.execute(
                """
                SELECT system_prompt, user_prompt, chat_history_json
                FROM conformity_prompts WHERE trial_id = ?
                ORDER BY created_at ASC LIMIT 1;
                """,
                (tid,),
            ).fetchone()
            if not prow:
                result["errors"].append(f"Trial {tid[:8]}... missing prompt")
                continue
            history = []
            if prow["chat_history_json"]:
                try:
                    history = json.loads(prow["chat_history_json"])
                except Exception:
                    pass
            # Extract wrong_answer from source_json
            wrong_answer = None
            if tr["source_json"]:
                try:
                    src = json.loads(tr["source_json"]) if isinstance(tr["source_json"], str) else tr["source_json"]
                    wrong_answer = src.get("wrong_answer")
                except Exception:
                    pass
            trial_data[tid] = {
                "model_id": str(tr["model_id"]),
                "variant": str(tr["variant"]),
                "temperature": float(tr["temperature"]),
                "seed": int(tr["seed"]),
                "ground_truth_text": tr["ground_truth_text"],
                "source_json": tr["source_json"],
                "wrong_answer": wrong_answer,
                "condition_name": str(tr["condition_name"]),
                "dataset_name": str(tr["dataset_name"]),
                "system_prompt": str(prow["system_prompt"] or ""),
                "user_prompt": str(prow["user_prompt"] or ""),
                "history": history,
                "top_k": top_k,
                "top_p": top_p,
            }

        by_model: Dict[str, List[str]] = {}
        for tid, data in trial_data.items():
            by_model.setdefault(data["model_id"], []).append(tid)

        gateway_cache: Dict[str, Tuple[Any, str]] = {}

        for mid, tids in by_model.items():
            if mid not in gateway_cache:
                try:
                    variant = trial_data[tids[0]]["variant"]
                    gw, api_id = _make_gateway(
                        model_id=mid,
                        variant=variant,
                        api_base=api_base,
                        api_key=api_key,
                        hf_cache_dir=hf_cache_dir,
                    )
                    gateway_cache[mid] = (gw, api_id)
                except Exception as e:
                    result["errors"].append(f"Model {mid}: {e}")
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
                    result["errors"].append(f"Trial {tid[:8]}: {e}")
                    continue

                raw_text = ""
                try:
                    raw_text = str(resp["choices"][0]["message"].get("content") or "")
                except Exception:
                    raw_text = str(resp)

                gt = data["ground_truth_text"]
                expected_answers = [str(gt)] if gt is not None else []
                classified = classify_output(
                    raw_text=raw_text,
                    cfg=output_parse_cfg,
                    system_prompt=data["system_prompt"],
                    user_prompt=data["user_prompt"],
                    expected_answer_texts=expected_answers,
                    token_logprobs=None,
                )
                scoring_result = score_single_output(
                    raw_text=raw_text,
                    ground_truth_text=gt,
                    wrong_answer=data.get("wrong_answer"),
                    condition_name=data.get("condition_name", "unknown"),
                    dataset_name=data.get("dataset_name", "unknown"),
                )

                output_id = str(uuid.uuid4())
                trace_db.insert_conformity_output(
                    output_id=output_id,
                    trial_id=tid,
                    raw_text=raw_text,
                    parsed_answer_text=scoring_result.parsed_answer_text,
                    parsed_answer_json={
                        "endorsement": scoring_result.endorsement,
                        "endorsement_evidence": scoring_result.endorsement_evidence,
                        "candidates": scoring_result.candidates,
                        "winning_candidate": scoring_result.winning_candidate,
                    },
                    is_correct=scoring_result.is_correct,
                    refusal_flag=scoring_result.refusal_flag,
                    latency_ms=latency_ms,
                    token_usage_json={
                        "_output_quality": {"label": classified.label.value, "metadata": classified.metadata}
                    },
                )
                result["outputs_backfilled"] += 1

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

        trace_db.conn.commit()
        print(
            f"[Complete] {db_path}: created {result['trials_created']} trials, "
            f"backfilled {result['outputs_backfilled']} outputs"
        )

    finally:
        trace_db.close()

    return result


def run_suite_completion(
    *,
    runs_dir: str,
    metadata_path: str,
    run_id: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    dry_run: bool = False,
    behavioral_only: bool = True,
) -> Dict[str, Any]:
    """
    Complete missing trials and outputs for runs per suite_expanded_temp{T}.json.

    Loads metadata (temp -> run_id, run_dir), loads matching suite config for each temp,
    and completes whatever is missing (items, trials, outputs).
    """
    repo_root = str(settings.PROJECT_ROOT)
    configs_dir = Path(repo_root) / "experiments" / "olmo_conformity" / "configs"

    with open(metadata_path, encoding="utf-8") as f:
        meta = json.load(f)

    experiments = meta.get("experiments", {})
    if not experiments:
        return {"runs": [], "total_trials_created": 0, "total_outputs_backfilled": 0, "total_errors": 0}

    hf_cache_dir = (
        os.environ.get("VIVARIUM_HF_CACHE")
        or os.environ.get("AAM_HF_CACHE")
        or os.environ.get("VIVARIUM_MODEL_DIR")
        or os.environ.get("AAM_MODEL_DIR")
    )

    def _load_paths() -> Dict[str, str]:
        try:
            suite_path = configs_dir / "test_scripts" / "suite_expanded_localtest_by_model.json"
            if suite_path.exists():
                cfg = load_suite_config(str(suite_path))
                return load_paths_config(str(suite_path), cfg)
        except Exception:
            pass
        return {}

    paths_cfg = _load_paths()
    if not hf_cache_dir and paths_cfg.get("models_dir"):
        hf_cache_dir = str(paths_cfg["models_dir"])

    results: Dict[str, Dict[str, Any]] = {}
    total_trials_created = 0
    total_outputs_backfilled = 0
    total_errors = 0

    for temp_str, info in experiments.items():
        if info.get("status") != "completed":
            continue
        rid = str(info.get("run_id", ""))
        if run_id and rid != run_id:
            continue

        run_dir = _discover_run_dir(runs_dir, rid)
        if not run_dir:
            print(f"[Complete] Skipping temp {temp_str}: run dir not found for {rid[:8]}...")
            continue

        db_path = os.path.join(run_dir, "simulation.db")
        if not os.path.isfile(db_path):
            print(f"[Complete] Skipping temp {temp_str}: no simulation.db at {db_path}")
            continue

        try:
            temp = float(temp_str)
        except ValueError:
            continue

        config_file = CONFIG_TEMPLATE.format(temp=temp)
        suite_path = configs_dir / config_file
        if not suite_path.exists():
            print(f"[Complete] Skipping temp {temp}: suite config not found: {suite_path}")
            continue

        suite_config = load_suite_config(str(suite_path))

        run_result = complete_single_run(
            db_path=db_path,
            run_id=rid,
            suite_config=suite_config,
            suite_config_path=str(suite_path),
            api_base=api_base,
            api_key=api_key,
            hf_cache_dir=hf_cache_dir,
            dry_run=dry_run,
            behavioral_only=behavioral_only,
        )

        results[run_dir] = run_result
        total_trials_created += run_result.get("trials_created", 0)
        total_outputs_backfilled += run_result.get("outputs_backfilled", 0)
        total_errors += len(run_result.get("errors", []))

    return {
        "runs": results,
        "total_trials_created": total_trials_created,
        "total_outputs_backfilled": total_outputs_backfilled,
        "total_errors": total_errors,
    }
