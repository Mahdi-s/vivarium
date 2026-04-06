from __future__ import annotations

import asyncio
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
from vivarium.persistence import ConformityOutputRow, TraceDb, TraceDbConfig
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
    render_ngram_sequence_prompt,
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


@dataclass
class _TrialWork:
    """Pre-built trial unit: all data needed to issue an LLM call and write its result.

    Populated in the serial build phase (DB reads/writes + prompt building).
    Consumed in the async fan-out phase (achat calls) and result processing phase.
    """

    trial_num: int
    total_trials: int
    trial_id: str
    item: JsonDict
    condition: JsonDict
    cond_name: str
    messages: List[JsonDict]
    system: str
    user: str
    dataset_name: str
    model_has_think_tokens: bool


@dataclass
class _TrialResult:
    """LLM call result paired with its originating work unit."""

    work: _TrialWork
    resp: Optional[JsonDict]
    latency_ms: float
    error: Optional[Exception]


async def _fan_out_trials(
    *,
    works: List[_TrialWork],
    gateway: Any,
    model_id_for_api: str,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    seed: Optional[int],
    max_concurrent: int,
) -> List[_TrialResult]:
    """Fan out LLM calls concurrently with bounded concurrency; return ordered results.

    Uses a fresh asyncio.Semaphore per invocation. The caller should keep all
    batch fan-out work on a single event loop for the lifetime of the async
    phase (LiteLLM may keep loop-bound background workers internally).
    The gateway's own RateLimiter (if configured) provides additional RPM/TPM
    back-pressure on top of this.
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def _call_one(w: _TrialWork) -> _TrialResult:
        async with sem:
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            try:
                resp = await gateway.achat(
                    model=model_id_for_api,
                    messages=w.messages,
                    tools=None,
                    tool_choice=None,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    seed=seed,
                )
                return _TrialResult(
                    work=w, resp=resp,
                    latency_ms=(loop.time() - t0) * 1000.0,
                    error=None,
                )
            except Exception as exc:
                return _TrialResult(
                    work=w, resp=None,
                    latency_ms=(loop.time() - t0) * 1000.0,
                    error=exc,
                )

    tasks = [asyncio.create_task(_call_one(w)) for w in works]
    return list(await asyncio.gather(*tasks))


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


def _normalize_suite_identity(cfg: JsonDict) -> str:
    """Deterministic string capturing suite identity: models + conditions + datasets + run params."""
    identity = {
        "suite_name": cfg.get("suite_name", ""),
        "models": sorted(
            [{"model_id": m.get("model_id"), "variant": m.get("variant")} for m in cfg.get("models", [])],
            key=lambda x: str(x.get("model_id", "")),
        ),
        "conditions": sorted([c.get("name", "") for c in cfg.get("conditions", [])]),
        "datasets": sorted([d.get("name", "") for d in cfg.get("datasets", [])]),
        "run": {
            "temperature": cfg.get("run", {}).get("temperature"),
            "seed": cfg.get("run", {}).get("seed"),
        },
    }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _find_incomplete_run_for_suite(
    runs_dir: str,
    suite_config_sha256: str,
    current_cfg: JsonDict,
    model_ids: List[str],
    condition_names: List[str],
    expected_item_count: int,
) -> Optional[Tuple[str, str]]:
    """Scan runs_dir for an incomplete run matching the current suite config.

    Matching strategy:
      1. Exact match on suite_config_sha256 (new runs store this).
      2. Fallback: compare normalized suite identity (models, conditions, datasets,
         temperature, seed) extracted from the stored suite_config. This covers
         runs created before the sha256 field was introduced.

    Returns (run_id, run_dir_path) for the most recent incomplete match, or None.
    """
    if not os.path.isdir(runs_dir):
        return None

    current_identity = _normalize_suite_identity(current_cfg)
    candidates: List[Tuple[int, float, str, str]] = []  # (completed_count, mtime, run_id, run_dir)

    for name in os.listdir(runs_dir):
        run_path = os.path.join(runs_dir, name)
        db_path = os.path.join(run_path, "simulation.db")
        if not os.path.isdir(run_path) or not os.path.isfile(db_path):
            continue

        try:
            tmp_db = TraceDb(TraceDbConfig(db_path=db_path))
            tmp_db.connect()

            row = tmp_db.conn.execute("SELECT run_id, config_json FROM runs LIMIT 1;").fetchone()
            if not row:
                tmp_db.close()
                continue

            db_run_id = str(row["run_id"])
            config = json.loads(row["config_json"]) if row["config_json"] else {}

            # Match 1: exact SHA (fast path for runs created after this feature)
            stored_sha = config.get("suite_config_sha256")
            matched = (stored_sha is not None and stored_sha == suite_config_sha256)

            # Match 2: normalized identity comparison (covers pre-existing runs)
            if not matched:
                stored_suite_cfg = config.get("suite_config")
                if isinstance(stored_suite_cfg, dict):
                    stored_identity = _normalize_suite_identity(stored_suite_cfg)
                    matched = (stored_identity == current_identity)

            if not matched:
                tmp_db.close()
                continue

            expected_cells = expected_item_count * len(condition_names) * len(model_ids)
            completed = tmp_db.conn.execute(
                "SELECT COUNT(DISTINCT t.trial_id) AS n FROM conformity_trials t "
                "JOIN conformity_outputs o ON o.trial_id = t.trial_id "
                "WHERE t.run_id = ? AND o.raw_text NOT LIKE '<error>%';",
                (db_run_id,),
            ).fetchone()["n"]

            if completed >= expected_cells:
                tmp_db.close()
                continue

            mtime = os.path.getmtime(db_path)
            candidates.append((completed, mtime, db_run_id, run_path))
            tmp_db.close()
        except Exception:
            continue

    if not candidates:
        return None

    # Prefer the run with the most completed work (then newest mtime as tiebreaker)
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    best_completed, best_mtime, best_run_id, best_path = candidates[0]
    if len(candidates) > 1:
        print(f"[Runner] WARNING: found {len(candidates)} incomplete runs matching this suite config:")
        for _cnt, _mt, _rid, _rp in candidates:
            print(f"  - {_rp}  (run_id={_rid}, completed={_cnt})")
        print(f"  Using run with most progress ({best_completed} outputs): {best_path}")
    return (best_run_id, best_path)


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

    if ctype == "ngram_sequence":
        system_style = str(params.get("system_style", "control"))
        system, sys_src = _load_system_prompt(system_style)

        confederates = int(params.get("confederates", 5))
        label_prefix = str(params.get("label_prefix", "String"))

        wrong = _get_wrong_answer(item, ctype)
        injected_answers = [wrong for _ in range(max(0, confederates))]

        user = render_ngram_sequence_prompt(
            question=question,
            injected_answers=injected_answers,
            label_prefix=label_prefix,
        )

        meta = {
            "prompt_renderer_version": PROMPT_RENDERER_VERSION,
            "condition_name": cond_name,
            "condition_type": ctype,
            "mechanism": "ngram_sequence",
            "prompt_family": "ablation_construct_validity",
            "system_style": system_style,
            "system_prompt_source": sys_src,
            "user_prompt_style": "sequence",
            "confederates": int(confederates),
            "label_prefix": label_prefix,
            "wrong_answer": wrong,
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
    resume_auto: bool = False,
    # Throughput controls: override suite config values when set.
    execution_mode: Optional[str] = None,       # "serial" | "async"; None → read from config
    db_flush_batch_size: Optional[int] = None,  # async batch size; None → read from config
    openrouter_provider: Optional[Dict[str, Any]] = None,  # suite-level provider routing
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

    suite_model_ids = [str(m.get("model_id") or "mock") for m in cfg.get("models", [])]
    suite_condition_names = [str(c.get("name") or "") for c in cfg.get("conditions", [])]

    # Count expected items per dataset (respecting max_items_per_dataset)
    _expected_item_count = 0
    for ds in cfg.get("datasets", []):
        rel_path = str(ds["path"])
        p = Path(rel_path)
        if not p.is_absolute():
            suite_rel = (suite_dir / p).resolve()
            abs_path = str(suite_rel) if suite_rel.exists() else str((Path(str(settings.PROJECT_ROOT)) / p).resolve())
        else:
            abs_path = str(p)
        try:
            _expected_item_count += len(clamp_items(read_jsonl(abs_path), cfg.get("run", {}).get("max_items_per_dataset")))
        except Exception:
            pass

    run_id_final = str(run_id or str(uuid.uuid4()))
    existing_run_dir = _find_existing_run_dir(effective_runs_dir, run_id_final) if run_id else None
    is_resume = existing_run_dir is not None

    # --resume-auto: discover an incomplete run for the same suite config
    if not is_resume and not run_id and resume_auto:
        _auto_sha = sha256_file(str(suite_path))
        _auto_result = _find_incomplete_run_for_suite(
            runs_dir=effective_runs_dir,
            suite_config_sha256=_auto_sha,
            current_cfg=cfg,
            model_ids=suite_model_ids,
            condition_names=suite_condition_names,
            expected_item_count=_expected_item_count,
        )
        if _auto_result:
            run_id_final, existing_run_dir = _auto_result
            is_resume = True
            print(f"[Runner] --resume-auto: found incomplete run to continue")

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
    suite_config_sha256 = sha256_file(str(suite_path))
    if not run_exists:
        trace_db.insert_run(
            RunMetadata(
                run_id=run_id_final,
                seed=int(cfg.get("run", {}).get("seed", 42)),
                created_at=time.time(),
                config={
                    "mode": "olmo_conformity",
                    "suite_config": cfg,
                    "suite_config_sha256": suite_config_sha256,
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

    # Throughput controls: CLI args override suite config values.
    _execution_mode = execution_mode or str(cfg.get("run", {}).get("execution_mode", "serial"))
    _db_flush_batch_size = db_flush_batch_size or int(cfg.get("run", {}).get("db_flush_batch_size", 20))

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

        # Build OpenRouter extras: model spec takes precedence over suite-level CLI arg.
        # These are forwarded to LiteLLMGateway as extra_body on each request.
        _or_provider = m.get("openrouter_provider") or openrouter_provider
        _or_transforms = m.get("openrouter_transforms")
        _or_extras_dict: Dict[str, Any] = {}
        if _or_provider is not None:
            _or_extras_dict["provider"] = _or_provider
        if _or_transforms is not None:
            _or_extras_dict["transforms"] = _or_transforms
        _or_extras: Optional[Dict[str, Any]] = _or_extras_dict or None
        if _or_extras and api_base:
            _provider_label = (
                _or_extras_dict.get("provider", {}).get("order", ["?"])[0]
                if isinstance(_or_extras_dict.get("provider"), dict)
                else str(_or_extras_dict.get("provider", ""))
            )
            print(f"  [Runner] OpenRouter provider routing: {_or_extras_dict}")

        gateway, model_id_for_api = create_gateway(
            model_id=model_id,
            variant=variant,
            api_base=api_base,
            api_key=api_key,
            hf_cache_dir=hf_cache_dir,
            capture_context=cap_ctx if capture_activations else None,
            rate_limit_config=rl_cfg,
            max_new_tokens=max_tokens,
            openrouter_extras=_or_extras,
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

        # ── ASYNC EXECUTION PATH ─────────────────────────────────────────────
        # Active when execution_mode="async" and activation capture is off.
        # Three phases:
        #   1. Build: serial DB writes + prompt building (no LLM, fast).
        #   2. Fan-out: bounded concurrent achat() calls via asyncio.
        #   3. Write: batch INSERT via executemany (reduced transaction churn).
        # After this block completes, rows is set to [] so the serial loop below
        # does nothing. All existing serial code is preserved and unchanged.
        if _execution_mode == "async" and not capture_activations:
            if capture_activations:
                print("  [Runner] Warning: async mode incompatible with capture_activations; falling back to serial.")
            elif use_judgeval:
                print("  [Runner] Note: judgeval scoring is not available in async mode; skipping.")

            _async_works: List[_TrialWork] = []
            trial_num = 0
            print(f"  [Runner] Async build phase: registering {total_trials} trial/prompt records in DB...")
            for row in rows:
                item = {
                    "item_id": row["item_id"],
                    "dataset_id": row["dataset_id"],
                    "domain": row["domain"],
                    "question": row["question"],
                    "ground_truth_text": row["ground_truth_text"],
                    "_run_seed": seed,
                    "_distractor_pool": distractor_pool_by_dataset.get(str(row["dataset_id"]), []),
                    "_global_distractor_pool": global_distractor_pool,
                }
                _src_str = row["source_json"]
                if _src_str:
                    try:
                        _src_data = json.loads(_src_str)
                        if _src_data.get("wrong_answer"):
                            item["wrong_answer"] = _src_data["wrong_answer"]
                    except Exception:
                        pass
                for cond_name, cond_id in condition_ids.items():
                    condition = {"name": cond_name, "params": condition_params.get(cond_name, {})}
                    trial_num += 1

                    _reuse_trial_id: Optional[str] = None
                    if is_resume:
                        _ex_row = trace_db.conn.execute(
                            "SELECT t.trial_id, o.output_id FROM conformity_trials t "
                            "LEFT JOIN conformity_outputs o ON t.trial_id = o.trial_id "
                            "  AND o.raw_text NOT LIKE '<error>%' "
                            "WHERE t.run_id = ? AND t.model_id = ? AND t.item_id = ? AND t.condition_id = ?",
                            (run_id_final, model_id, str(item["item_id"]), cond_id),
                        ).fetchone()
                        if _ex_row:
                            if _ex_row["output_id"] is not None:
                                continue
                            _reuse_trial_id = str(_ex_row["trial_id"])

                    if _reuse_trial_id:
                        trial_id = _reuse_trial_id
                        print(f"  [Runner] Trial {trial_num}/{total_trials}: item={item['item_id']}, cond={cond_name} (orphaned, resuming)")
                    else:
                        trial_id = str(uuid.uuid4())
                        print(f"  [Runner] Trial {trial_num}/{total_trials}: item={item['item_id']}, cond={cond_name} [build]")
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
                                    "execution_mode": "async",
                                },
                            )
                        except Exception as _e:
                            print(f"Warning: failed to write trial metadata: {_e}")

                    _ex_prompt = None
                    if _reuse_trial_id:
                        _ex_prompt = trace_db.conn.execute(
                            "SELECT system_prompt, user_prompt, chat_history_json FROM conformity_prompts "
                            "WHERE trial_id = ? ORDER BY created_at ASC LIMIT 1;",
                            (trial_id,),
                        ).fetchone()

                    if _ex_prompt:
                        system = str(_ex_prompt["system_prompt"] or "")
                        user = str(_ex_prompt["user_prompt"] or "")
                        _hist: List[JsonDict] = []
                        try:
                            _rh = _ex_prompt["chat_history_json"]
                            if _rh:
                                _hist = json.loads(_rh)
                        except Exception:
                            _hist = []
                        history = _hist
                    else:
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
                        try:
                            _pm = dict(prompt_meta or {})
                            _pm.update({
                                "prompt_id": prompt_id,
                                "trial_id": trial_id,
                                "rendered_prompt_hash": prompt_hash,
                            })
                            trace_db.upsert_conformity_prompt_metadata(prompt_id=prompt_id, metadata=_pm)
                        except Exception as _e:
                            print(f"Warning: failed to write prompt metadata: {_e}")

                    messages = build_messages(system=system, user=user, history=history)
                    _async_works.append(_TrialWork(
                        trial_num=trial_num,
                        total_trials=total_trials,
                        trial_id=trial_id,
                        item=dict(item),
                        condition=condition,
                        cond_name=cond_name,
                        messages=messages,
                        system=system,
                        user=user,
                        dataset_name=dataset_name_by_id.get(str(item["dataset_id"]), "unknown"),
                        model_has_think_tokens=bool(model_config.get("has_think_tokens", False)),
                    ))

            # Fan-out + write phase: process in batches of _db_flush_batch_size
            _n_queued = len(_async_works)
            _total_batches = max(1, (_n_queued + _db_flush_batch_size - 1) // _db_flush_batch_size)
            print(f"\n  [Runner] Async fan-out: {_n_queued} trials queued in {_total_batches} batch(es), max_concurrent={rate_limit_max_concurrent}")
            _t_async_start = time.time()
            _async_latencies: List[float] = []

            async def _run_async_batches_single_loop() -> None:
                # Keep all batches on one event loop so LiteLLM async workers/queues
                # are not rebound across asyncio.run() boundaries.
                for _bi in range(0, _n_queued, _db_flush_batch_size):
                    _batch = _async_works[_bi:_bi + _db_flush_batch_size]
                    _batch_num = _bi // _db_flush_batch_size + 1
                    print(f"  [Runner] Batch {_batch_num}/{_total_batches}: issuing {len(_batch)} concurrent LLM calls...")
                    _t_batch = time.time()

                    _batch_results: List[_TrialResult] = await _fan_out_trials(
                        works=_batch,
                        gateway=gateway,
                        model_id_for_api=model_id_for_api,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        seed=seed,
                        max_concurrent=rate_limit_max_concurrent,
                    )

                    _output_rows: List[ConformityOutputRow] = []
                    for _result in _batch_results:
                        _w = _result.work
                        _async_latencies.append(_result.latency_ms)

                        if _result.error is not None:
                            print(f"    [Runner] Trial {_w.trial_num}/{total_trials}: item={_w.item['item_id']}, cond={_w.cond_name} ERROR: {_result.error}")
                            _raw_text = f"<error>{_result.error}</error>"
                            _is_correct: Optional[bool] = None
                            _refusal = False
                            _parsed: Optional[str] = None
                            _parsed_json: Dict[str, Any] = {"error": str(_result.error)}
                            _tok_usage: Dict[str, Any] = {}
                        else:
                            _resp = _result.resp
                            _raw_text = ""
                            try:
                                _msg = _resp["choices"][0]["message"]
                                _content = str(_msg.get("content") or "")
                                _reasoning = _msg.get("reasoning") or _msg.get("reasoning_content") or ""
                                if _reasoning:
                                    _raw_text = f"<think>{_reasoning}</think>{_content}"
                                else:
                                    _raw_text = _content
                            except Exception:
                                _raw_text = str(_resp)

                            _classified = classify_output(
                                raw_text=_raw_text,
                                cfg=output_parse_cfg,
                                system_prompt=_w.system,
                                user_prompt=_w.user,
                                expected_answer_texts=(
                                    [str(_w.item.get("ground_truth_text"))]
                                    if _w.item.get("ground_truth_text") is not None else []
                                ),
                                token_logprobs=None,
                            )
                            _sr = score_single_output(
                                raw_text=_raw_text,
                                ground_truth_text=_w.item.get("ground_truth_text"),
                                wrong_answer=_w.item.get("wrong_answer"),
                                condition_name=_w.cond_name,
                                dataset_name=_w.dataset_name,
                            )
                            _parsed = _sr.parsed_answer_text
                            _refusal = _sr.refusal_flag
                            _is_correct = _sr.is_correct
                            _parsed_json = {
                                "endorsement": _sr.endorsement,
                                "endorsement_evidence": _sr.endorsement_evidence,
                                "candidates": _sr.candidates,
                                "winning_candidate": _sr.winning_candidate,
                            }
                            _tok_usage = {
                                "_output_quality": {
                                    "label": _classified.label.value,
                                    "metadata": _classified.metadata,
                                }
                            }
                            print(
                                f"    [Runner] Trial {_w.trial_num}/{total_trials}: "
                                f"item={_w.item['item_id']}, cond={_w.cond_name}, "
                                f"correct={_is_correct} ({_result.latency_ms:.0f}ms)"
                            )

                        _output_rows.append(ConformityOutputRow(
                            output_id=str(uuid.uuid4()),
                            trial_id=_w.trial_id,
                            raw_text=_raw_text,
                            parsed_answer_text=_parsed,
                            parsed_answer_json=_parsed_json,
                            is_correct=_is_correct,
                            refusal_flag=_refusal,
                            latency_ms=_result.latency_ms,
                            token_usage_json=_tok_usage,
                        ))

                    trace_db.batch_write_conformity_trial_results(_output_rows)
                    _batch_wall = time.time() - _t_batch
                    print(f"  [Runner] Batch {_batch_num}/{_total_batches}: wrote {len(_output_rows)} outputs (wall={_batch_wall:.1f}s)")

                    try:
                        import torch as _torch
                        gc.collect()
                        if hasattr(_torch, "mps") and hasattr(_torch.mps, "empty_cache"):
                            _torch.mps.empty_cache()
                        elif _torch.cuda.is_available():
                            _torch.cuda.empty_cache()
                    except Exception:
                        pass

            asyncio.run(_run_async_batches_single_loop())

            _async_wall = time.time() - _t_async_start
            if _async_latencies:
                _mean_lat = sum(_async_latencies) / len(_async_latencies)
                _p95_lat = sorted(_async_latencies)[max(0, int(len(_async_latencies) * 0.95) - 1)]
                print(
                    f"\n  [Runner] Async timing: {len(_async_latencies)} trials, "
                    f"wall={_async_wall:.1f}s, mean_llm={_mean_lat:.0f}ms, p95_llm={_p95_lat:.0f}ms, "
                    f"throughput={len(_async_latencies) / max(_async_wall, 0.001):.2f} trials/s"
                )
            # Exhaust the serial loop — all work done above.
            rows = []

        # ── SERIAL EXECUTION PATH ────────────────────────────────────────────
        # Default mode; runs when execution_mode="serial" OR async was not
        # applicable (e.g., capture_activations=True). If the async path ran,
        # rows was set to [] above and this loop iterates zero times.
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

                # On resume: check per-(model, item, condition) whether work already exists.
                _reuse_trial_id: Optional[str] = None
                if is_resume:
                    _existing_row = trace_db.conn.execute(
                        "SELECT t.trial_id, o.output_id FROM conformity_trials t "
                        "LEFT JOIN conformity_outputs o ON t.trial_id = o.trial_id "
                        "  AND o.raw_text NOT LIKE '<error>%' "
                        "WHERE t.run_id = ? AND t.model_id = ? AND t.item_id = ? AND t.condition_id = ?",
                        (run_id_final, model_id, str(item["item_id"]), cond_id),
                    ).fetchone()
                    if _existing_row:
                        if _existing_row["output_id"] is not None:
                            continue
                        _reuse_trial_id = str(_existing_row["trial_id"])

                if _reuse_trial_id:
                    trial_id = _reuse_trial_id
                    print(f"  [Runner] Trial {trial_num}/{total_trials}: item={item['item_id']}, condition={cond_name} (resuming orphaned trial {trial_id[:8]})")
                else:
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

                # Build prompt (or reuse existing for orphaned trials)
                _existing_prompt = None
                if _reuse_trial_id:
                    _existing_prompt = trace_db.conn.execute(
                        "SELECT system_prompt, user_prompt, chat_history_json FROM conformity_prompts "
                        "WHERE trial_id = ? ORDER BY created_at ASC LIMIT 1;",
                        (trial_id,),
                    ).fetchone()

                if _existing_prompt:
                    system = str(_existing_prompt["system_prompt"] or "")
                    user = str(_existing_prompt["user_prompt"] or "")
                    history: List[JsonDict] = []
                    try:
                        raw_hist = _existing_prompt["chat_history_json"]
                        if raw_hist:
                            history = json.loads(raw_hist)
                    except Exception:
                        history = []
                    prompt_meta = None
                else:
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
                    msg = resp["choices"][0]["message"]
                    content = str(msg.get("content") or "")
                    # OpenRouter returns reasoning/thinking tokens in a separate
                    # 'reasoning' field (or 'reasoning_content') rather than inline
                    # <think> tags.  Reconstruct the full output so downstream
                    # think-token extraction works identically to local inference.
                    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
                    if reasoning:
                        raw_text = f"<think>{reasoning}</think>{content}"
                    else:
                        raw_text = content
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
