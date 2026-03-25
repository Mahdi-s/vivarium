from __future__ import annotations

import gc
import json
import os
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from vivarium.interpretability import CaptureConfig, CaptureContext
from vivarium.llm_gateway import RateLimitConfig, create_gateway
from vivarium.output_parsing import OutputParsingConfig, classify_output
from vivarium.persistence import TraceDb, TraceDbConfig
from vivarium.types import RunMetadata
from vivarium.settings import settings

from .io import clamp_items, deterministic_prompt_hash, load_paths_config, load_suite_config, read_jsonl, sha256_file
try:
    from .judgeval_scorers import ConformityExample, ConformityScorer, RationalizationScorer, TruthfulnessScorer
    JUDGEVAL_AVAILABLE = True
except ImportError:
    JUDGEVAL_AVAILABLE = False
    ConformityExample = None
    ConformityScorer = None
    RationalizationScorer = None
    TruthfulnessScorer = None
from .olmo_utils import ensure_olmo_model_downloaded
from .prompts import (
    PROMPT_RENDERER_VERSION,
    build_messages,
    load_text,
    make_confederate_block,
    make_participant_dialogue_block,
    normalize_confederate_tone,
    render_asch_user,
    render_template,
    render_authority_claim_prompt,
    render_zhu_conversation_prompt,
    render_zhu_question_distillation_prompt,
    stable_int_seed,
)

from .enhanced_scoring import score_single_output

JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class RunPaths:
    run_dir: str
    db_path: str
    artifacts_dir: str
    figures_dir: str
    tables_dir: str
    exports_dir: str


def _ensure_dirs(run_dir: str) -> RunPaths:
    artifacts_dir = os.path.join(run_dir, "artifacts")
    figures_dir = os.path.join(artifacts_dir, "figures")
    tables_dir = os.path.join(artifacts_dir, "tables")
    exports_dir = os.path.join(run_dir, "exports")
    for d in (run_dir, artifacts_dir, figures_dir, tables_dir, exports_dir):
        os.makedirs(d, exist_ok=True)
    return RunPaths(
        run_dir=run_dir,
        db_path=os.path.join(run_dir, "simulation.db"),
        artifacts_dir=artifacts_dir,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        exports_dir=exports_dir,
    )


def _find_existing_run_dir(runs_dir: str, run_id: str) -> Optional[str]:
    """Return path to an existing run directory whose name ends with _<run_id>, or None."""
    if not run_id or not os.path.isdir(runs_dir):
        return None
    suffix = f"_{run_id}"
    for name in os.listdir(runs_dir):
        if name.endswith(suffix) and os.path.isdir(os.path.join(runs_dir, name)):
            return os.path.join(runs_dir, name)
    return None


def _get_wrong_answer(item: JsonDict, condition_type: str) -> str:
    """
    Get the wrong answer for pressure conditions (Asch, authoritative_bias).
    
    SCIENTIFIC REQUIREMENT: For valid Asch-style manipulation, the confederates/user
    must claim an INCORRECT answer, not the ground truth. This function fails fast
    if no wrong_answer is provided, preventing answer leakage that would invalidate
    the pressure condition.
    
    Args:
        item: Item dict with question, ground_truth_text, and optionally wrong_answer
        condition_type: The condition type (for error messages)
        
    Returns:
        The wrong answer string
        
    Raises:
        ValueError: If wrong_answer is missing or equals ground_truth_text (answer leakage)
    """
    wrong_answer = item.get("wrong_answer")
    ground_truth = item.get("ground_truth_text")
    item_id = item.get("item_id", "unknown")
    
    if not wrong_answer:
        raise ValueError(
            f"SCIENTIFIC VALIDITY ERROR: Item '{item_id}' is missing 'wrong_answer' field. "
            f"Pressure conditions ({condition_type}) require an explicit wrong answer to avoid "
            f"answer leakage. Add 'wrong_answer' to the dataset item or use a dataset with "
            f"wrong answers (e.g., minimal_items_wrong.jsonl)."
        )
    
    # Validate that wrong_answer != ground_truth (whitespace-normalized, case-sensitive).
    #
    # NOTE: Do not use case-insensitive comparison here. Some domains encode meaning in
    # capitalization (e.g., genetics genotypes like "Bb" vs "bb", chemical formulas),
    # and lowercasing would incorrectly flag valid wrong answers as leakage.
    wrong_norm = " ".join(str(wrong_answer).strip().split())
    gt_norm = " ".join(str(ground_truth).strip().split()) if ground_truth is not None else ""
    if ground_truth and wrong_norm == gt_norm:
        raise ValueError(
            f"SCIENTIFIC VALIDITY ERROR: Item '{item_id}' has wrong_answer='{wrong_answer}' "
            f"which equals ground_truth_text='{ground_truth}'. This would cause answer leakage. "
            f"The wrong_answer must be different from the correct answer."
        )
    
    return str(wrong_answer)


def _build_prompt_for_condition(
    *,
    condition: JsonDict,
    item: JsonDict,
    prompts_root: str,
) -> Tuple[str, str, List[JsonDict], JsonDict]:
    def _load_system_prompt(style: str) -> Tuple[str, str]:
        """
        Return (system_prompt_text, system_prompt_source).

        style options:
          - control: use controls/control_system.txt
          - pressure_conservative: use synthetic_asch/asch_system.txt (anti-conformity)
          - none: empty system prompt
        """
        s = str(style or "control").strip().lower().replace("-", "_")
        if s in {"none", "empty"}:
            return "", "system:none"
        if s in {"pressure", "pressure_conservative", "asch_system", "conservative"}:
            p = os.path.join(prompts_root, "synthetic_asch", "asch_system.txt")
            return load_text(p), f"file:{p}"
        p = os.path.join(prompts_root, "controls", "control_system.txt")
        return load_text(p), f"file:{p}"

    def _norm(s: Optional[str]) -> str:
        return str(s or "").strip().lower()

    def _pick_alt_answer(*, pool: List[str], exclude: List[str], rng: random.Random) -> Tuple[str, JsonDict]:
        """
        Pick a deterministic alternate distractor answer from a pool, excluding known strings.
        Returns (answer, metadata).
        """
        exclude_norm = {_norm(x) for x in exclude if _norm(x)}
        candidates = [x for x in pool if _norm(x) and _norm(x) not in exclude_norm]
        if candidates:
            idx = int(rng.randrange(len(candidates)))
            return str(candidates[idx]), {"source": "pool", "pool_size": len(pool), "candidates": len(candidates), "index": idx}
        # Fallback: generate a stable-but-obviously-alternate string; track that it is a fallback.
        fallback = "some other answer"
        return fallback, {"source": "fallback", "pool_size": len(pool), "candidates": 0}

    def _pick_k_distinct(*, pool: List[str], exclude: List[str], k: int, rng: random.Random) -> Tuple[List[str], JsonDict]:
        exclude_norm = {_norm(x) for x in exclude if _norm(x)}
        candidates_raw = [x for x in pool if _norm(x) and _norm(x) not in exclude_norm]
        # Deduplicate by normalized form to better approximate a "no-majority" Diverse setting.
        seen: set[str] = set()
        candidates: List[str] = []
        for x in candidates_raw:
            nx = _norm(x)
            if not nx or nx in seen:
                continue
            seen.add(nx)
            candidates.append(str(x))
        rng.shuffle(candidates)
        picked = [str(x) for x in candidates[: max(0, int(k))]]
        return picked, {"pool_size": len(pool), "candidates": len(candidates), "k": int(k), "picked": len(picked)}

    cond_name = str(condition.get("name") or "unknown_condition")
    params = condition.get("params", {}) if isinstance(condition.get("params", {}), dict) else {}
    ctype = str(params.get("type") or cond_name or "control")
    question = str(item.get("question") or "")
    history: List[JsonDict] = []
    item_id = str(item.get("item_id") or "unknown_item")
    dataset_id = str(item.get("dataset_id") or "unknown_dataset")
    base_wrong = str(item.get("wrong_answer") or "")
    ground_truth = str(item.get("ground_truth_text") or "")

    run_seed = str(item.get("_run_seed") or "")
    render_seed = stable_int_seed(
        "olmo_conformity",
        PROMPT_RENDERER_VERSION,
        run_seed,
        str(item_id),
        str(cond_name),
        str(ctype),
    )
    rng = random.Random(render_seed)

    if ctype == "control":
        system, sys_src = _load_system_prompt("control")
        user = question
        meta: JsonDict = {
            "prompt_renderer_version": PROMPT_RENDERER_VERSION,
            "condition_name": cond_name,
            "condition_type": ctype,
            "mechanism": "control",
            "system_prompt_source": sys_src,
            "user_prompt_style": "direct_question",
            "item_id": item_id,
            "dataset_id": dataset_id,
            "ground_truth_text": (ground_truth if ground_truth else None),
            "wrong_answer": (base_wrong if base_wrong else None),
            "run_seed": run_seed,
            "render_seed": render_seed,
        }
        return system, user, history, meta

    if ctype == "synthetic_asch_history":
        system, sys_src = _load_system_prompt("pressure_conservative")
        template_path = os.path.join(prompts_root, "synthetic_asch", "asch_user_template.txt")
        template = load_text(template_path)
        confederates = int(condition.get("params", {}).get("confederates", 5))
        confidence = str(condition.get("params", {}).get("confidence", "high"))
        # FIXED: Require explicit wrong_answer - no fallback to ground_truth
        wrong = _get_wrong_answer(item, ctype)
        conf_block = make_confederate_block(confederates=confederates, wrong_answer=wrong, confidence=confidence)
        user = render_asch_user(template=template, question=question, confederate_block=conf_block)
        meta = {
            "prompt_renderer_version": PROMPT_RENDERER_VERSION,
            "condition_name": cond_name,
            "condition_type": ctype,
            "mechanism": "asch",
            "prompt_family": "olmo_conservative",
            "system_prompt_source": sys_src,
            "user_prompt_template_source": f"file:{template_path}",
            "user_prompt_style": "prior_users",
            "confederates": int(confederates),
            "tone": normalize_confederate_tone("confident" if str(confidence).lower() != "low" else "uncertain"),
            "consensus": "unanimous",
            "distillation": False,
            "devils_advocate": False,
            "wrong_answer": wrong,
            "ground_truth_text": (ground_truth if ground_truth else None),
            "run_seed": run_seed,
            "render_seed": render_seed,
            "confederate_lines": [ln for ln in str(conf_block).splitlines() if ln.strip()],
        }
        return system, user, history, meta

    if ctype == "authoritative_bias":
        system, sys_src = _load_system_prompt("pressure_conservative")
        # Authoritative bias lives in user prompt directly (no history)
        claim_strength = str(params.get("user_claim_strength", "high"))
        # FIXED: Require explicit wrong_answer - no fallback to ground_truth
        wrong = _get_wrong_answer(item, ctype)
        tmpl_rel = params.get("user_template")
        if tmpl_rel:
            template_path = (tmpl_rel if os.path.isabs(str(tmpl_rel)) else os.path.join(prompts_root, str(tmpl_rel)))
        else:
            if claim_strength == "low":
                template_path = os.path.join(prompts_root, "authoritative_bias", "authority_user_low.txt")
            else:
                template_path = os.path.join(prompts_root, "authoritative_bias", "authority_user_high.txt")
        template = load_text(template_path)
        user = render_template(template=template, vars={"wrong_answer": wrong, "question": question})
        meta = {
            "prompt_renderer_version": PROMPT_RENDERER_VERSION,
            "condition_name": cond_name,
            "condition_type": ctype,
            "mechanism": "authority",
            "prompt_family": "olmo_conservative",
            "system_prompt_source": sys_src,
            "user_prompt_template_source": f"file:{template_path}",
            "user_prompt_style": "user_claim",
            "claim_strength": str(claim_strength),
            "wrong_answer": wrong,
            "ground_truth_text": (ground_truth if ground_truth else None),
            "run_seed": run_seed,
            "render_seed": render_seed,
        }
        return system, user, history, meta

    if ctype == "asch_peer_pressure":
        system_style = str(params.get("system_style", "control"))
        system, sys_src = _load_system_prompt(system_style)

        confederates = int(params.get("confederates", 5))
        tone = normalize_confederate_tone(str(params.get("tone", "plain")))
        consensus = str(params.get("consensus", "unanimous")).strip().lower()
        prompt_style = str(params.get("prompt_style", "conversation")).strip().lower()
        distillation = bool(params.get("distillation", False))
        devils_advocate = bool(params.get("devils_advocate", False))

        wrong = _get_wrong_answer(item, ctype)
        # Use dataset-specific pool if provided; else fall back to the item's wrong answer only.
        pool = list(item.get("_distractor_pool") or [])  # injected by runner
        global_pool = list(item.get("_global_distractor_pool") or [])
        effective_pool = pool if pool else global_pool

        alt = None
        alt_meta = None
        if devils_advocate:
            alt, alt_meta = _pick_alt_answer(pool=effective_pool, exclude=[wrong, ground_truth], rng=rng)

        total_participants = max(1, confederates + 1)
        participant_index = confederates + 1

        conf_meta: List[JsonDict] = []
        consensus_meta: Optional[JsonDict] = None
        if distillation:
            user = render_zhu_question_distillation_prompt(
                question=question,
                majority_answer=wrong,
                total_participants=total_participants,
                participant_index=participant_index,
            )
            consensus_meta = {"mode": "unanimous", "distilled": True}
        else:
            answers: List[str]
            if consensus == "diverse":
                others, pick_meta = _pick_k_distinct(
                    pool=effective_pool,
                    exclude=[wrong, ground_truth],
                    k=max(0, confederates - 1),
                    rng=rng,
                )
                # If the dataset pool is too small after deduplication, top up from the global pool,
                # then (as a last resort) use deterministic placeholders to maintain dialogue shape.
                needed = max(0, confederates - 1 - len(others))
                if needed > 0 and global_pool and (effective_pool is pool):
                    extra, extra_meta = _pick_k_distinct(
                        pool=global_pool,
                        exclude=[wrong, ground_truth, *list(others)],
                        k=needed,
                        rng=rng,
                    )
                    others = list(others) + list(extra)
                    pick_meta["global_fill"] = extra_meta
                    needed = max(0, confederates - 1 - len(others))
                if needed > 0:
                    placeholders = [f"some other answer ({i + 1})" for i in range(needed)]
                    others = list(others) + placeholders
                    pick_meta["fallback_fill"] = {"needed": int(needed), "placeholders": placeholders}
                answers = [wrong] + list(others)
                rng.shuffle(answers)
                consensus_meta = {"mode": "diverse", **pick_meta}
            else:
                answers = [wrong for _ in range(max(0, confederates))]
                consensus_meta = {"mode": "unanimous"}
                if devils_advocate and alt:
                    # Break unanimity with a single dissenter (DA).
                    answers[-1] = alt

            conf_block, conf_meta = make_participant_dialogue_block(answers=answers, tone=tone, rng=rng)
            user = render_zhu_conversation_prompt(
                question=question,
                confederate_block=conf_block,
                total_participants=total_participants,
                participant_index=participant_index,
            )

        meta = {
            "prompt_renderer_version": PROMPT_RENDERER_VERSION,
            "condition_name": cond_name,
            "condition_type": ctype,
            "mechanism": "asch",
            "prompt_family": "zhu2024",
            "source": {"paper": "arXiv:2410.12428"},
            "system_style": system_style,
            "system_prompt_source": sys_src,
            "user_prompt_style": prompt_style,
            "consensus": consensus,
            "consensus_meta": consensus_meta,
            "tone": tone,
            "confederates": int(confederates),
            "distillation": bool(distillation),
            "devils_advocate": bool(devils_advocate),
            "wrong_answer": wrong,
            "alternate_answer": alt,
            "alternate_answer_meta": alt_meta,
            "ground_truth_text": (ground_truth if ground_truth else None),
            "run_seed": run_seed,
            "render_seed": render_seed,
            "confederate_utterances": conf_meta,
        }
        return system, user, history, meta

    if ctype == "authority_claim":
        system_style = str(params.get("system_style", "control"))
        system, sys_src = _load_system_prompt(system_style)

        claim_style = str(params.get("claim_style", "trust"))
        devils_advocate = bool(params.get("devils_advocate", False))

        wrong = _get_wrong_answer(item, ctype)
        pool = list(item.get("_distractor_pool") or [])  # injected by runner
        global_pool = list(item.get("_global_distractor_pool") or [])
        effective_pool = pool if pool else global_pool

        alt = None
        alt_meta = None
        if devils_advocate:
            alt, alt_meta = _pick_alt_answer(pool=effective_pool, exclude=[wrong, ground_truth], rng=rng)

        user, claim_meta = render_authority_claim_prompt(
            question=question,
            claimed_answer=wrong,
            claim_style=claim_style,
            alternate_answer=alt if devils_advocate else None,
        )

        meta = {
            "prompt_renderer_version": PROMPT_RENDERER_VERSION,
            "condition_name": cond_name,
            "condition_type": ctype,
            "mechanism": "authority",
            "prompt_family": "zhu2024",
            "source": {"paper": "arXiv:2410.12428"},
            "system_style": system_style,
            "system_prompt_source": sys_src,
            "user_prompt_style": "claim",
            "claim_style": claim_meta.get("claim_style"),
            "devils_advocate": bool(devils_advocate),
            "wrong_answer": wrong,
            "alternate_answer": alt,
            "alternate_answer_meta": alt_meta,
            "ground_truth_text": (ground_truth if ground_truth else None),
            "run_seed": run_seed,
            "render_seed": render_seed,
        }
        return system, user, history, meta

    # Fallback: treat as control
    system, sys_src = _load_system_prompt("control")
    meta = {
        "prompt_renderer_version": PROMPT_RENDERER_VERSION,
        "condition_name": cond_name,
        "condition_type": ctype,
        "mechanism": "fallback_control",
        "system_prompt_source": sys_src,
        "user_prompt_style": "direct_question",
        "item_id": item_id,
        "dataset_id": dataset_id,
        "ground_truth_text": (ground_truth if ground_truth else None),
        "wrong_answer": (base_wrong if base_wrong else None),
        "run_seed": run_seed,
        "render_seed": render_seed,
    }
    return system, question, history, meta


def run_suite(
    *,
    suite_config_path: str,
    runs_dir: str,
    run_id: Optional[str],
    api_base: Optional[str],
    api_key: Optional[str],
    rate_limit_enabled: bool,
    rate_limit_rpm: Optional[int],
    rate_limit_tpm: Optional[int],
    rate_limit_max_concurrent: int,
    capture_activations: bool = False,
    capture_layers: Optional[List[int]] = None,
    capture_components: Optional[List[str]] = None,
    capture_dtype: str = "float16",
    use_judgeval: bool = False,
    judgeval_judge_model: str = "llama3.2",
    judgeval_ollama_base: str = "http://localhost:11434/v1",
) -> RunPaths:
    cfg = load_suite_config(suite_config_path)

    suite_path = Path(suite_config_path).resolve()
    suite_dir = suite_path.parent
    
    # Load paths config (models_dir, runs_dir) from shared config file
    paths_cfg = load_paths_config(suite_config_path, cfg)
    
    # Environment variables take precedence over config file
    # This allows the automation script to override HPC paths for local runs
    hf_cache_dir = (
        os.environ.get("VIVARIUM_HF_CACHE")
        or os.environ.get("AAM_HF_CACHE")
        or os.environ.get("VIVARIUM_MODEL_DIR")
        or os.environ.get("AAM_MODEL_DIR")
        or os.environ.get("AAM_MODELS_DIR")  # back-compat
        or paths_cfg.get("models_dir")
    )
    config_runs_dir = (
        os.environ.get("VIVARIUM_ARTIFACTS_DIR")
        or os.environ.get("VIVARIUM_RUNS_DIR")
        or os.environ.get("AAM_ARTIFACTS_DIR")
        or os.environ.get("AAM_RUNS_DIR")
        or paths_cfg.get("runs_dir")
    )
    
    # Use config runs_dir as default if CLI didn't specify a custom path
    effective_runs_dir = runs_dir
    if runs_dir == "./runs" and config_runs_dir:
        effective_runs_dir = config_runs_dir

    run_id_final = str(run_id or str(uuid.uuid4()))
    existing_run_dir = _find_existing_run_dir(effective_runs_dir, run_id_final) if run_id else None
    is_resume = existing_run_dir is not None

    if is_resume:
        run_dir = existing_run_dir
        print(f"[Runner] Resuming existing run: run_id={run_id_final}, run_dir={run_dir}")
    else:
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        run_dir = os.path.join(effective_runs_dir, f"{ts}_{run_id_final}")
    paths = _ensure_dirs(run_dir)

    trace_db = TraceDb(TraceDbConfig(db_path=paths.db_path))
    trace_db.connect()
    trace_db.init_schema()
    trace_db.init_conformity_schema()

    repo_root = str(settings.PROJECT_ROOT)

    # Record repo state for traceability (best-effort; do not fail runs if git is unavailable).
    repo_state: JsonDict = {}
    try:
        import subprocess  # noqa: S404

        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, timeout=5).strip()
        )
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True, timeout=5)
        repo_state = {"git_commit": commit, "git_dirty": bool(str(dirty).strip())}
    except Exception:
        repo_state = {}

    run_exists = (
        trace_db.conn.execute("SELECT 1 FROM runs WHERE run_id = ?;", (run_id_final,)).fetchone() is not None
    )
    if not run_exists:
        trace_db.insert_run(
            RunMetadata(
                run_id=run_id_final,
                seed=int(cfg.get("run", {}).get("seed", 42)),
                created_at=time.time(),
                config={
                    "mode": "olmo_conformity",
                    "suite_config": cfg,
                    "prompt_renderer_version": PROMPT_RENDERER_VERSION,
                    "repo_state": repo_state,
                },
            )
        )

    # Prefer prompts relative to the suite config (so the experiment can be relocated),
    # but tolerate nested config folders (e.g., configs/test_scripts/...).
    prompts_root = os.path.join(repo_root, "experiments", "olmo_conformity", "prompts")
    cursor = suite_dir
    while True:
        candidate = cursor.parent / "prompts"
        if candidate.is_dir():
            prompts_root = str(candidate)
            break
        if cursor == cursor.parent:
            break
        cursor = cursor.parent

    output_parse_cfg = OutputParsingConfig()

    def _resolve_data_path(path_str: str) -> str:
        p = Path(str(path_str))
        if p.is_absolute():
            return str(p)
        # Try suite-relative first, then project root.
        suite_rel = (suite_dir / p).resolve()
        if suite_rel.exists():
            return str(suite_rel)
        proj_rel = (Path(repo_root) / p).resolve()
        return str(proj_rel)

    # Register datasets + items (on resume: reuse existing dataset/condition IDs, skip existing items)
    dataset_ids: Dict[str, str] = {}
    for ds in cfg.get("datasets", []):
        name = str(ds["name"])
        version = str(ds.get("version", "v0"))
        rel_path = str(ds["path"])
        abs_path = _resolve_data_path(rel_path)
        if is_resume:
            row = trace_db.conn.execute(
                "SELECT dataset_id FROM conformity_datasets WHERE name = ? AND version = ?;",
                (name, version),
            ).fetchone()
            dataset_id = str(row["dataset_id"]) if row else str(uuid.uuid4())
        else:
            dataset_id = str(uuid.uuid4())
        dataset_ids[name] = dataset_id
        trace_db.upsert_conformity_dataset(
            dataset_id=dataset_id,
            name=name,
            version=version,
            path=rel_path,
            sha256=sha256_file(abs_path),
        )

        items = clamp_items(read_jsonl(abs_path), cfg.get("run", {}).get("max_items_per_dataset"))
        for it in items:
            item_id = str(it.get("item_id") or str(uuid.uuid4()))
            # Store wrong_answer in source_json for retrieval during prompt building
            source_data = it.get("source") if isinstance(it.get("source"), dict) else {}
            if it.get("wrong_answer"):
                source_data["wrong_answer"] = str(it["wrong_answer"])
            if is_resume:
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
                # IMPORTANT: Avoid `str(None) == "None"` which would incorrectly create
                # a fake ground truth for unlabeled items (e.g., social_conventions).
                ground_truth_text=(None if it.get("ground_truth_text") is None else str(it.get("ground_truth_text"))),
                ground_truth_json=(it.get("ground_truth_json") if isinstance(it.get("ground_truth_json"), dict) else None),
                source_json=source_data if source_data else None,
            )

    # Reverse map: dataset_id -> dataset_name (for enhanced scoring)
    dataset_name_by_id = {did: dname for dname, did in dataset_ids.items()}

    # Register conditions (on resume: reuse existing condition IDs)
    condition_ids: Dict[str, str] = {}
    for cond in cfg.get("conditions", []):
        cond_id = str(uuid.uuid4())
        name = str(cond.get("name") or cond_id)
        if is_resume:
            row = trace_db.conn.execute(
                "SELECT condition_id FROM conformity_conditions WHERE name = ?;", (name,)
            ).fetchone()
            if row:
                cond_id = str(row["condition_id"])
        condition_ids[name] = cond_id
        trace_db.upsert_conformity_condition(condition_id=cond_id, name=name, params=dict(cond.get("params") or {}))

    # Pre-load condition params (avoids N×M redundant SQL queries in the trial loop)
    condition_params: Dict[str, Dict] = {}
    for cname, cid in condition_ids.items():
        row = trace_db.conn.execute(
            "SELECT params_json FROM conformity_conditions WHERE condition_id = ?;", (cid,)
        ).fetchone()
        try:
            condition_params[cname] = json.loads(row["params_json"])
        except Exception:
            condition_params[cname] = {}

    # Execute trials (behavioral only). Interpretability/probes/interventions are separate steps.
    temperature = float(cfg.get("run", {}).get("temperature", 0.0))
    top_k_raw = cfg.get("run", {}).get("top_k", cfg.get("run", {}).get("top_n"))
    top_k: Optional[int]
    try:
        top_k = (None if top_k_raw is None else int(top_k_raw))
    except Exception:
        top_k = None
    if top_k is not None and top_k <= 0:
        top_k = None
    top_p_raw = cfg.get("run", {}).get("top_p", cfg.get("run", {}).get("nucleus_p"))
    top_p: Optional[float]
    try:
        top_p = (None if top_p_raw is None else float(top_p_raw))
    except Exception:
        top_p = None
    if top_p is not None and not (0.0 < top_p <= 1.0):
        top_p = None
    seed = int(cfg.get("run", {}).get("seed", 42))

    # Setup Judge Eval tracer if requested
    judgment_tracer = None
    if use_judgeval:
        try:
            from judgeval.tracer import Tracer
            judgment_tracer = Tracer(project_name="olmo_conformity")
            print("Judge Eval tracer initialized (local mode)")
        except ImportError:
            print("Warning: Judge Eval not installed, skipping tracer integration")
            use_judgeval = False
    
    # Pre-create JudgeVal scorers (reused across all trials)
    _jv_conformity_scorer = None
    _jv_truthfulness_scorer = None
    _jv_rationalization_scorer = None
    if use_judgeval and JUDGEVAL_AVAILABLE:
        if ConformityScorer is not None:
            try:
                _jv_conformity_scorer = ConformityScorer(judge_model=judgeval_judge_model, ollama_base=judgeval_ollama_base)  # type: ignore
            except Exception as e:
                print(f"Warning: Failed to create conformity scorer: {e}")
        if TruthfulnessScorer is not None:
            try:
                _jv_truthfulness_scorer = TruthfulnessScorer(judge_model=judgeval_judge_model, ollama_base=judgeval_ollama_base)  # type: ignore
            except Exception as e:
                print(f"Warning: Failed to create truthfulness scorer: {e}")
        if RationalizationScorer is not None:
            try:
                _jv_rationalization_scorer = RationalizationScorer(judge_model=judgeval_judge_model, ollama_base=judgeval_ollama_base)  # type: ignore
            except Exception as e:
                print(f"Warning: Failed to create rationalization scorer: {e}")

    # Setup activation capture if requested
    activations_dir = os.path.join(run_dir, "activations") if capture_activations else None
    if capture_activations and activations_dir:
        os.makedirs(activations_dir, exist_ok=True)
        default_layers = capture_layers or list(range(32))
        default_components = capture_components or ["resid_post"]
        cap_cfg = CaptureConfig(
            layers=default_layers,
            components=default_components,
            trigger_actions=["trial_execution"],  # Capture for all trials
            token_position=-1,  # Last token
        )
        cap_ctx = CaptureContext(
            output_dir=activations_dir,
            config=cap_cfg,
            dtype=capture_dtype,
            trace_db=trace_db,
        )
    else:
        cap_ctx = None

    for m in cfg.get("models", []):
        variant = str(m.get("variant") or "unknown")
        model_id = str(m.get("model_id") or "mock")
        max_tokens = int(m.get("max_new_tokens", 128))
        model_config = {
            "variant": variant,
            "model_id": model_id,
            "max_new_tokens": max_tokens,
            "has_think_tokens": bool(m.get("has_think_tokens", False)),
        }

        # Pre-flight: verify local HuggingFace models are available (OLMo-specific)
        if not api_base and model_id.startswith("allenai/"):
            try:
                models_dir_for_download = str(Path(str(hf_cache_dir)).parent) if hf_cache_dir else None
                ensure_olmo_model_downloaded(model_id=model_id, models_dir=models_dir_for_download, import_to_ollama=False)
            except Exception as e:
                print(f"ERROR: Failed to verify model {model_id}: {e}")
                raise

        print(f"\n{'='*60}")
        print(f"Setting up model: {model_id} (variant={variant})")
        print(f"{'='*60}")

        rl_cfg = None if not rate_limit_enabled else RateLimitConfig(
            max_concurrent_requests=int(rate_limit_max_concurrent),
            requests_per_minute=rate_limit_rpm,
            tokens_per_minute=rate_limit_tpm,
        )

        gateway, model_id_for_api = create_gateway(
            model_id=model_id,
            variant=variant,
            api_base=api_base,
            api_key=api_key,
            hf_cache_dir=hf_cache_dir,
            capture_context=cap_ctx if capture_activations else None,
            rate_limit_config=rl_cfg,
            max_new_tokens=max_tokens,
        )

        # Query items back from DB for this run's datasets
        print(f"\n[Runner] Querying items from database...")
        rows = trace_db.conn.execute(
            """
            SELECT item_id, dataset_id, domain, question, ground_truth_text, source_json
            FROM conformity_items
            WHERE dataset_id IN (SELECT dataset_id FROM conformity_datasets)
            ORDER BY dataset_id, item_id;
            """
        ).fetchall()
        
        # Build per-dataset distractor pools (wrong_answer strings) for Diverse/DA prompt variants.
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
        num_conditions = len(condition_ids)
        total_trials = len(rows) * num_conditions
        print(f"  [Runner] Found {len(rows)} items, {num_conditions} conditions = {total_trials} total trials")
        print(f"  [Runner] Starting trial execution...\n")

        trial_num = 0
        for row in rows:
            # Build item dict including wrong_answer from source_json if available
            item = {
                "item_id": row["item_id"],
                "dataset_id": row["dataset_id"],
                "domain": row["domain"],
                "question": row["question"],
                "ground_truth_text": row["ground_truth_text"],
                # Expose run seed to the prompt renderer so sampling is reproducible and logged.
                "_run_seed": seed,
                # Inject distractor pools for Diverse/DA variants (tracked via prompt metadata).
                "_distractor_pool": distractor_pool_by_dataset.get(str(row["dataset_id"]), []),
                "_global_distractor_pool": global_distractor_pool,
            }
            # Extract wrong_answer from source_json
            source_json_str = row["source_json"]
            if source_json_str:
                try:
                    source_data = json.loads(source_json_str)
                    if source_data.get("wrong_answer"):
                        item["wrong_answer"] = source_data["wrong_answer"]
                except Exception:
                    pass
            for cond_name, cond_id in condition_ids.items():
                condition = {"name": cond_name, "params": condition_params.get(cond_name, {})}

                trial_num += 1

                # On resume: skip item×condition pairs that already have a completed output
                if is_resume:
                    _existing = trace_db.conn.execute(
                        "SELECT 1 FROM conformity_trials t "
                        "JOIN conformity_outputs o ON t.trial_id = o.trial_id "
                        "WHERE t.run_id = ? AND t.item_id = ? AND t.condition_id = ?",
                        (run_id_final, str(item["item_id"]), cond_id),
                    ).fetchone()
                    if _existing:
                        continue

                trial_id = str(uuid.uuid4())
                print(f"  [Runner] Trial {trial_num}/{total_trials}: item={item['item_id']}, condition={cond_name}")
                trace_db.insert_conformity_trial(
                    trial_id=trial_id,
                    run_id=run_id_final,
                    model_id=model_id,
                    variant=variant,
                    item_id=str(item["item_id"]),
                    condition_id=cond_id,
                    seed=seed,
                    temperature=temperature,
                )
                
                # Trial-level metadata: generation config + gateway + model config (for full traceability)
                try:
                    trace_db.upsert_conformity_trial_metadata(
                        trial_id=trial_id,
                        metadata={
                            "prompt_renderer_version": PROMPT_RENDERER_VERSION,
                            "suite_name": str(cfg.get("suite_name") or ""),
                            "suite_version": str(cfg.get("suite_version") or ""),
                            "generation": {
                                "seed": int(seed),
                                "temperature": float(temperature),
                                "top_k": (None if top_k is None else int(top_k)),
                                "top_p": (None if top_p is None else float(top_p)),
                            },
                            "model": {
                                "model_id": str(model_id),
                                "variant": str(variant),
                                "model_config": (model_config if isinstance(model_config, dict) else {}),
                            },
                            "gateway": {
                                "class": gateway.__class__.__name__,
                                "api_base": (str(api_base) if api_base else None),
                            },
                        },
                    )
                except Exception as e:
                    print(f"Warning: failed to write trial metadata: {e}")

                print(f"    [Runner] Building prompt...")
                system, user, history, prompt_meta = _build_prompt_for_condition(
                    condition=condition, item=item, prompts_root=prompts_root
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
                
                # Prompt-level structured metadata for traceability (tone, consensus mode, DA/QD settings, etc.)
                try:
                    meta_to_store = dict(prompt_meta or {})
                    meta_to_store.update(
                        {
                            "prompt_id": prompt_id,
                            "trial_id": trial_id,
                            "rendered_prompt_hash": prompt_hash,
                        }
                    )
                    trace_db.upsert_conformity_prompt_metadata(prompt_id=prompt_id, metadata=meta_to_store)
                except Exception as e:
                    print(f"Warning: failed to write prompt metadata: {e}")

                messages = build_messages(system=system, user=user, history=history)

                # Use trial_num as time_step for activation alignment
                time_step = trial_num
                agent_id = f"trial_{trial_id[:8]}"
                
                # Register trial step for activation alignment if capturing
                if capture_activations and cap_ctx:
                    trace_db.upsert_conformity_trial_step(
                        trial_id=trial_id,
                        time_step=time_step,
                        agent_id=agent_id
                    )

                print(f"    [Runner] Calling gateway.chat() with seed={seed}...")
                t0 = time.time()
                resp = gateway.chat(
                    model=model_id_for_api,
                    messages=messages,
                    tools=None,
                    tool_choice=None,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    seed=seed,
                )
                latency_ms = (time.time() - t0) * 1000.0
                print(f"    [Runner] Gateway response received ({latency_ms:.1f}ms)")

                # Commit activations if capturing
                if capture_activations and cap_ctx and getattr(gateway, "capture_context", None) is cap_ctx:
                    print(f"    [Runner] Committing activations...")
                    cap_ctx.on_action_decided(
                        run_id=run_id_final,
                        time_step=time_step,
                        agent_id=agent_id,
                        model_id=model_id,
                        action_name="trial_execution"
                    )
                    cap_ctx.flush_step(time_step=time_step)
                    print(f"    [Runner] Activations committed")

                # Extract text best-effort
                raw_text = ""
                try:
                    raw_text = str(resp["choices"][0]["message"].get("content") or "")
                except Exception:
                    raw_text = str(resp)

                classified = classify_output(
                    raw_text=raw_text,
                    cfg=output_parse_cfg,
                    system_prompt=system,
                    user_prompt=user,
                    expected_answer_texts=(
                        [str(item.get("ground_truth_text"))] if item.get("ground_truth_text") is not None else []
                    ),
                    token_logprobs=None,
                )

                scoring_result = score_single_output(
                    raw_text=raw_text,
                    ground_truth_text=item.get("ground_truth_text"),
                    wrong_answer=item.get("wrong_answer"),
                    condition_name=cond_name,
                    dataset_name=dataset_name_by_id.get(item["dataset_id"], "unknown"),
                )
                parsed = scoring_result.parsed_answer_text
                refusal = scoring_result.refusal_flag
                is_correct = scoring_result.is_correct

                # Judge Eval evaluation (uses pre-created scorers)
                judgeval_scores = {}
                if use_judgeval and judgment_tracer and JUDGEVAL_AVAILABLE and ConformityExample is not None:
                    try:
                        import asyncio
                        example = ConformityExample(  # type: ignore
                            question=item.get("question", ""),
                            answer=raw_text,
                            ground_truth=item.get("ground_truth_text"),
                            condition=condition.get("name", "unknown"),
                        )
                        async def _run_judgeval():
                            scores = {}
                            if _jv_conformity_scorer is not None:
                                try:
                                    scores["conformity"] = await _jv_conformity_scorer.a_score_example(example)
                                except Exception as e:
                                    print(f"Warning: Conformity scorer failed: {e}")
                            if _jv_truthfulness_scorer is not None:
                                try:
                                    scores["truthfulness"] = await _jv_truthfulness_scorer.a_score_example(example)
                                except Exception as e:
                                    print(f"Warning: Truthfulness scorer failed: {e}")
                            if model_config.get("has_think_tokens", False) and _jv_rationalization_scorer is not None:
                                try:
                                    scores["rationalization"] = await _jv_rationalization_scorer.a_score_example(example)
                                except Exception as e:
                                    print(f"Warning: Rationalization scorer failed: {e}")
                            return scores
                        try:
                            judgeval_scores = asyncio.run(_run_judgeval())
                        except RuntimeError:
                            print("Warning: Cannot run async Judge Eval in sync context, skipping")
                    except Exception as e:
                        print(f"Warning: Judge Eval evaluation failed: {e}")

                output_id = str(uuid.uuid4())
                
                # Store enhanced scoring + Judge Eval scores in parsed_answer_json
                parsed_json = {
                    "endorsement": scoring_result.endorsement,
                    "endorsement_evidence": scoring_result.endorsement_evidence,
                    "candidates": scoring_result.candidates,
                    "winning_candidate": scoring_result.winning_candidate,
                }
                if judgeval_scores:
                    parsed_json["judgeval"] = judgeval_scores

                token_usage_json = {
                    "_output_quality": {
                        "label": classified.label.value,
                        "metadata": classified.metadata,
                    }
                }
                
                trace_db.insert_conformity_output(
                    output_id=output_id,
                    trial_id=trial_id,
                    raw_text=raw_text,
                    parsed_answer_text=parsed,
                    parsed_answer_json=parsed_json,
                    is_correct=is_correct,
                    refusal_flag=refusal,
                    latency_ms=latency_ms,
                    token_usage_json=token_usage_json,
                )
                print(f"    [Runner] Trial {trial_num}/{total_trials} complete (correct={is_correct}, refusal={refusal})\n")

                # Periodic memory cleanup to prevent MPS/CUDA fragmentation on long runs
                if trial_num % 20 == 0:
                    try:
                        import torch
                        gc.collect()
                        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                            torch.mps.empty_cache()
                        elif torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

        # Clean up model memory between iterations to prevent MPS device mismatch errors
        # This is critical when running multiple 7B models sequentially on Apple Silicon
        try:
            import torch
            if hasattr(gateway, "_model"):
                del gateway._model
            if hasattr(gateway, "_tokenizer"):
                del gateway._tokenizer
            del gateway
            gc.collect()
            # Clear MPS cache if using Apple Silicon
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            print(f"[Runner] Memory cleanup complete for model {model_id}")
        except Exception as e:
            print(f"[Runner] Warning: Memory cleanup failed: {e}")

    print(f"\n[Runner] All {total_trials} trials completed")
    trace_db.close()
    return paths
