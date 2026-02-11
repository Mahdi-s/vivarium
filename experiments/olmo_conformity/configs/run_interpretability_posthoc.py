#!/usr/bin/env python3
"""
Posthoc interpretability pipeline for an *existing* Olmo conformity run.

Why this exists:
- The expanded temperature sweeps were originally run in behavioral-only mode
  (no activation capture), so interpretability tables (trial_steps, activations,
  probes, projections, interventions) are empty.
- This script backfills the missing pieces *in-place* under the existing
  run_dir in HPC scratch, keyed by the same run_id.

Pipeline (default):
1) Backfill `conformity_trial_steps` for behavioral trials (deterministic mapping)
2) Re-run the model on stored prompts to capture activations into
   `activation_metadata` + `activations/step_*.safetensors`
3) Train truth/social probes per (variant, model_id) and compute projections
4) Run a targeted intervention battery per variant (battery1 by default)
5) Regenerate paper-facing figures/tables and write a scientific report

NOTE: This is intentionally variant-aware because our expanded suites contain
multiple model variants inside one run. Any per-variant step MUST filter trials
by variant to avoid cross-model leakage / invalid interventions.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# --- Repo import setup (works whether or not the package is installed) ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]  # experiments/olmo_conformity/configs -> repo root
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


from aam.analytics.reporting import ScientificReportGenerator  # noqa: E402
from aam.experiments.olmo_conformity.analysis import generate_core_figures  # noqa: E402
from aam.experiments.olmo_conformity.intervention import run_intervention_sweep  # noqa: E402
from aam.experiments.olmo_conformity.olmo_utils import get_olmo_model_config  # noqa: E402
from aam.experiments.olmo_conformity.prompts import build_messages  # noqa: E402
from aam.experiments.olmo_conformity.vector_analysis import run_truth_social_vector_analysis  # noqa: E402
from aam.interpretability import CaptureConfig, CaptureContext  # noqa: E402
from aam.llm_gateway import select_local_gateway  # noqa: E402
from aam.persistence import TraceDb, TraceDbConfig  # noqa: E402


PATHS_CONFIG_FILE = SCRIPT_DIR / "paths.json"


def _load_paths_config() -> Dict[str, str]:
    if PATHS_CONFIG_FILE.exists():
        return json.loads(PATHS_CONFIG_FILE.read_text())
    return {}


def _find_run_dir(*, runs_dir: Path, run_id: str) -> Path:
    # Allow multiple timestamps; pick most recent lexicographically (YYYYMMDD_HHMMSS_...).
    matches = sorted(runs_dir.glob(f"*_{run_id}"), key=lambda p: p.name, reverse=True)
    if not matches:
        raise FileNotFoundError(f"Could not find run_dir under {runs_dir} ending with _{run_id}")
    return matches[0]


def _parse_csv_ints(s: str) -> List[int]:
    return [int(x) for x in str(s).split(",") if str(x).strip() != ""]


def _parse_csv_floats(s: str) -> List[float]:
    return [float(x) for x in str(s).split(",") if str(x).strip() != ""]


def _escape_sql_str(s: str) -> str:
    # This is only used for composing internal SQL snippets (trial_filter_sql).
    return str(s).replace("'", "''")


@dataclass(frozen=True)
class RunResolved:
    run_id: str
    run_dir: Path
    db_path: Path
    artifacts_dir: Path
    temperature: float


def _resolve_run(
    *,
    run_id: str,
    runs_dir: Path,
    run_dir_override: Optional[str] = None,
    db_override: Optional[str] = None,
) -> RunResolved:
    run_dir = Path(run_dir_override) if run_dir_override else _find_run_dir(runs_dir=runs_dir, run_id=run_id)
    db_path = Path(db_override) if db_override else (run_dir / "simulation.db")
    if not db_path.exists():
        raise FileNotFoundError(f"Missing simulation.db at {db_path}")
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Extract temperature from DB (should be single-valued for these runs).
    conn = None
    try:
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT DISTINCT temperature FROM conformity_trials WHERE run_id = ? ORDER BY temperature ASC;",
            (str(run_id),),
        ).fetchall()
        temps = [float(r[0]) for r in rows if r and r[0] is not None]
        temperature = temps[0] if temps else 0.0
    finally:
        if conn is not None:
            conn.close()

    return RunResolved(
        run_id=str(run_id),
        run_dir=run_dir,
        db_path=db_path,
        artifacts_dir=artifacts_dir,
        temperature=float(temperature),
    )


def _ensure_trial_steps(*, trace_db: TraceDb, run_id: str) -> int:
    """
    Ensure `conformity_trial_steps` exists for behavioral trials.

    The original behavioral-only runs didn't populate this table because activation
    capture was disabled. We create a deterministic mapping:
      time_step = row_number over trials ordered by created_at
      agent_id  = "trial_" + trial_id[:8]
    """
    existing = trace_db.conn.execute(
        """
        SELECT COUNT(*) AS c
        FROM conformity_trial_steps s
        JOIN conformity_trials t ON t.trial_id = s.trial_id
        WHERE t.run_id = ?;
        """,
        (str(run_id),),
    ).fetchone()
    if existing is not None and int(existing["c"]) > 0:
        return 0

    trials = trace_db.conn.execute(
        """
        SELECT trial_id
        FROM conformity_trials
        WHERE run_id = ?
        ORDER BY created_at ASC, trial_id ASC;
        """,
        (str(run_id),),
    ).fetchall()
    if not trials:
        return 0

    inserted = 0
    for i, r in enumerate(trials):
        trial_id = str(r["trial_id"])
        time_step = int(i)
        agent_id = f"trial_{trial_id[:8]}"
        trace_db.upsert_conformity_trial_step(trial_id=trial_id, time_step=time_step, agent_id=agent_id)
        inserted += 1
    return inserted


def _trial_prompt_messages(*, trace_db: TraceDb, trial_id: str) -> List[Dict[str, Any]]:
    row = trace_db.conn.execute(
        """
        SELECT system_prompt, user_prompt, chat_history_json
        FROM conformity_prompts
        WHERE trial_id = ?
        ORDER BY created_at ASC
        LIMIT 1;
        """,
        (str(trial_id),),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Missing prompt row for trial_id={trial_id}")

    system = str(row["system_prompt"] or "")
    user = str(row["user_prompt"] or "")
    history: List[Dict[str, Any]] = []
    try:
        raw = row["chat_history_json"] or "[]"
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            history = [m for m in parsed if isinstance(m, dict)]
    except Exception:
        history = []

    return build_messages(system=system, user=user, history=history)


def _activation_exists_for_trial(
    *,
    trace_db: TraceDb,
    run_id: str,
    model_id: str,
    trial_id: str,
    layer_index: int,
    component: str,
) -> bool:
    row = trace_db.conn.execute(
        """
        SELECT s.time_step, s.agent_id
        FROM conformity_trial_steps s
        WHERE s.trial_id = ?
        LIMIT 1;
        """,
        (str(trial_id),),
    ).fetchone()
    if row is None:
        return False
    ts = int(row["time_step"])
    agent_id = str(row["agent_id"])

    rec = trace_db.conn.execute(
        """
        SELECT 1
        FROM activation_metadata
        WHERE run_id = ? AND time_step = ? AND agent_id = ? AND model_id = ? AND layer_index = ? AND component = ?
        LIMIT 1;
        """,
        (str(run_id), int(ts), str(agent_id), str(model_id), int(layer_index), str(component)),
    ).fetchone()
    return rec is not None


def backfill_behavioral_activations(
    *,
    trace_db: TraceDb,
    run_id: str,
    run_dir: Path,
    capture_layers: List[int],
    capture_component: str = "resid_post",
    capture_dtype: str = "float16",
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """
    Replay stored prompts and capture activations for behavioral trials.

    This writes:
    - `activation_metadata` rows
    - `activations/step_<time_step>.safetensors` shards
    """
    activations_dir = run_dir / "activations"
    activations_dir.mkdir(parents=True, exist_ok=True)

    # CaptureContext expects short component names (e.g. "resid_post").
    cap_cfg = CaptureConfig(
        layers=list(capture_layers),
        components=[str(capture_component)],
        trigger_actions=["trial_execution"],
        token_position=-1,
    )
    cap_ctx = CaptureContext(output_dir=str(activations_dir), config=cap_cfg, dtype=str(capture_dtype), trace_db=trace_db)

    # Group by model_id to avoid re-loading weights for every trial.
    models = trace_db.conn.execute(
        """
        SELECT DISTINCT model_id
        FROM conformity_trials
        WHERE run_id = ?
        ORDER BY model_id ASC;
        """,
        (str(run_id),),
    ).fetchall()
    model_ids = [str(r["model_id"]) for r in models]

    stats: Dict[str, Any] = {"models": {}, "total_trials": 0, "captured_trials": 0, "skipped_trials": 0}

    for model_id in model_ids:
        model_cfg = get_olmo_model_config(model_id)
        max_new_tokens = int(model_cfg.get("max_new_tokens", 128))
        print("\n" + "=" * 70)
        print(f"[Activation Backfill] model_id={model_id}")
        print(f"  max_new_tokens={max_new_tokens}")
        print("=" * 70)

        gateway = select_local_gateway(
            model_id_or_path=str(model_id),
            capture_context=cap_ctx,
            max_new_tokens=max_new_tokens,
            scientific_mode=True,
        )

        trials = trace_db.conn.execute(
            """
            SELECT t.trial_id, t.temperature, t.seed, s.time_step, s.agent_id
            FROM conformity_trials t
            JOIN conformity_trial_steps s ON s.trial_id = t.trial_id
            WHERE t.run_id = ? AND t.model_id = ?
            ORDER BY s.time_step ASC;
            """,
            (str(run_id), str(model_id)),
        ).fetchall()

        captured = 0
        skipped = 0

        for tr in trials:
            trial_id = str(tr["trial_id"])
            temperature = float(tr["temperature"])
            seed = int(tr["seed"])
            time_step = int(tr["time_step"])
            agent_id = str(tr["agent_id"])

            stats["total_trials"] += 1

            if skip_existing:
                # Probe one representative layer at the component we expect in activation_metadata.
                # CaptureConfig("resid_post") expands to "hook_resid_post" in metadata.
                if _activation_exists_for_trial(
                    trace_db=trace_db,
                    run_id=run_id,
                    model_id=model_id,
                    trial_id=trial_id,
                    layer_index=int(capture_layers[0]) if capture_layers else 0,
                    component="hook_resid_post",
                ):
                    skipped += 1
                    stats["skipped_trials"] += 1
                    continue

            msgs = _trial_prompt_messages(trace_db=trace_db, trial_id=trial_id)

            _ = gateway.chat(model=str(model_id), messages=msgs, tools=None, tool_choice=None, temperature=temperature, seed=seed)
            # Commit + flush activations aligned to this trial's step.
            cap_ctx.on_action_decided(
                run_id=str(run_id),
                time_step=int(time_step),
                agent_id=str(agent_id),
                model_id=str(model_id),
                action_name="trial_execution",
            )
            cap_ctx.flush_step(time_step=int(time_step))
            captured += 1
            stats["captured_trials"] += 1

        stats["models"][model_id] = {"n_trials": len(trials), "captured": captured, "skipped": skipped}

        # Best-effort GPU memory hygiene between model variants.
        try:
            import torch  # type: ignore

            del gateway
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    return stats


def run_vector_analysis_all_variants(
    *,
    trace_db: TraceDb,
    run_id: str,
    artifacts_dir: Path,
    truth_probe_dataset_path: str,
    social_probe_dataset_path: str,
    probe_layers: List[int],
    component: str,
    token_position: int,
    dtype: str,
    temperature: float,
) -> Dict[str, Any]:
    # Distinct (model_id, variant) pairs
    rows = trace_db.conn.execute(
        """
        SELECT DISTINCT model_id, variant
        FROM conformity_trials
        WHERE run_id = ?
        ORDER BY variant ASC;
        """,
        (str(run_id),),
    ).fetchall()
    pairs = [(str(r["model_id"]), str(r["variant"])) for r in rows]

    out: Dict[str, Any] = {"by_variant": {}}
    for model_id, variant in pairs:
        print("\n" + "=" * 70)
        print(f"[Vector Analysis] variant={variant} model_id={model_id}")
        print("=" * 70)
        res = run_truth_social_vector_analysis(
            trace_db=trace_db,
            run_id=str(run_id),
            model_id=str(model_id),
            variant=str(variant),
            truth_probe_dataset_path=str(truth_probe_dataset_path),
            social_probe_dataset_path=str(social_probe_dataset_path),
            layers=list(probe_layers),
            component=str(component),
            token_position=int(token_position),
            dtype=str(dtype),
            artifacts_dir=str(artifacts_dir),
            temperature=float(temperature),
        )
        out["by_variant"][variant] = res
    return out


def run_interventions_battery1(
    *,
    trace_db: TraceDb,
    run_id: str,
    vector_results: Dict[str, Any],
    intervention_layers: List[int],
    intervention_alphas: List[float],
    temperature: float,
    component_hook: str,
) -> Dict[str, Any]:
    """
    Battery I (from protocol): immutable_facts_minimal only, pressure-only, all variants.

    This keeps the expanded temp sweep tractable while still providing causal evidence.
    """
    out: Dict[str, Any] = {"by_variant": {}, "total_inserted": 0}

    for variant, res in (vector_results.get("by_variant") or {}).items():
        social_probe_id = (res or {}).get("social_probe_id")
        if not social_probe_id:
            out["by_variant"][variant] = {"skipped": True, "reason": "missing social_probe_id"}
            continue

        # Vector analysis writes probes to: <run_dir>/artifacts/social_probe_<variant>.safetensors
        db_dir = Path(trace_db._config.db_path).parent  # type: ignore[attr-defined]
        artifact_path = str(db_dir / "artifacts" / f"social_probe_{variant}.safetensors")
        if not os.path.exists(artifact_path):
            out["by_variant"][variant] = {"skipped": True, "reason": f"missing probe artifact: {artifact_path}"}
            continue

        # Resolve model_id for this variant.
        row = trace_db.conn.execute(
            "SELECT model_id FROM conformity_trials WHERE run_id = ? AND variant = ? LIMIT 1;",
            (str(run_id), str(variant)),
        ).fetchone()
        if row is None:
            out["by_variant"][variant] = {"skipped": True, "reason": "missing model_id for variant"}
            continue
        model_id = str(row["model_id"])

        esc_variant = _escape_sql_str(variant)
        trial_filter_sql = (
            "t.variant = '" + esc_variant + "' "
            "AND i.ground_truth_text IS NOT NULL "
            "AND i.dataset_id IN (SELECT dataset_id FROM conformity_datasets WHERE name = 'immutable_facts_minimal') "
            "AND t.condition_id IN (SELECT condition_id FROM conformity_conditions "
            "WHERE name != 'control' AND name NOT LIKE '%probe_capture%'))"
        )

        print("\n" + "=" * 70)
        print(f"[Interventions] Battery I | variant={variant} model_id={model_id}")
        print(f"  probe_artifact={artifact_path}")
        print("=" * 70)

        inserted = run_intervention_sweep(
            trace_db=trace_db,
            run_id=str(run_id),
            model_id=str(model_id),
            probe_artifact_path=str(artifact_path),
            social_probe_id=str(social_probe_id),
            target_layers=list(intervention_layers),
            component_hook=str(component_hook),
            alpha_values=list(intervention_alphas),
            max_new_tokens=64,
            trial_filter_sql=trial_filter_sql,
            temperature=float(temperature),
        )
        out["by_variant"][variant] = {"inserted": int(inserted)}
        out["total_inserted"] += int(inserted)

    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Posthoc interpretability pipeline for an existing Olmo conformity run.")
    p.add_argument("--run-id", type=str, required=True, help="Existing run UUID (directory suffix).")
    p.add_argument("--hpc", action="store_true", help="Use experiments/olmo_conformity/configs/paths.json for scratch paths.")
    p.add_argument("--runs-dir", type=str, default=None, help="Override base runs directory.")
    p.add_argument("--run-dir", type=str, default=None, help="Override run directory (otherwise resolved from runs-dir + run-id).")
    p.add_argument("--db", type=str, default=None, help="Override simulation.db path.")

    # Activation capture
    default_layers_32 = ",".join(str(i) for i in range(32))
    p.add_argument("--capture-layers", type=str, default=default_layers_32, help="Comma-separated layer indices.")
    p.add_argument("--capture-component", type=str, default="resid_post", help="Capture component (CaptureContext form).")
    p.add_argument("--capture-dtype", type=str, default="float16", choices=["float16", "float32"])
    p.add_argument("--skip-activations", action="store_true", help="Skip behavioral activation backfill step.")

    # Probes / projections
    p.add_argument(
        "--truth-probe-dataset",
        type=str,
        default="experiments/olmo_conformity/datasets/candidates/truth_probe_train.jsonl",
    )
    p.add_argument(
        "--social-probe-dataset",
        type=str,
        default="experiments/olmo_conformity/datasets/candidates/social_probe_train.jsonl",
    )
    p.add_argument("--probe-layers", type=str, default=default_layers_32, help="Comma-separated probe layer indices.")
    p.add_argument("--probe-component", type=str, default="hook_resid_post")
    p.add_argument("--probe-token-position", type=int, default=-1)
    p.add_argument("--probe-dtype", type=str, default="float16", choices=["float16", "float32"])
    p.add_argument("--skip-vector-analysis", action="store_true")

    # Interventions
    p.add_argument("--skip-interventions", action="store_true")
    p.add_argument("--intervention-battery", type=str, default="battery1", choices=["battery1", "none"])
    p.add_argument("--intervention-layers", type=str, default="15,16,17,18,19,20")
    p.add_argument("--intervention-alphas", type=str, default="0.5,1.0,2.0")
    p.add_argument("--intervention-component-hook", type=str, default="hook_resid_post")

    # Reporting
    p.add_argument("--skip-reports", action="store_true")

    args = p.parse_args(argv)

    paths_cfg = _load_paths_config() if bool(args.hpc) else {}
    runs_dir = Path(str(args.runs_dir) if args.runs_dir else (paths_cfg.get("runs_dir") or str(REPO_ROOT / "runs")))

    # HPC convenience: ensure HF/transformers cache points at scratch, not $HOME.
    if bool(args.hpc):
        models_dir = paths_cfg.get("models_dir")
        if models_dir:
            try:
                hf_cache = Path(str(models_dir))
                os.environ.setdefault("HF_HOME", str(hf_cache.parent))
                os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_cache))
                os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache))
                # Prefer offline loads on HPC (avoid accidental re-downloads / hub calls).
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                print(f"[env] HF_HOME={os.environ.get('HF_HOME')}")
                print(f"[env] HUGGINGFACE_HUB_CACHE={os.environ.get('HUGGINGFACE_HUB_CACHE')}")
            except Exception:
                pass

    resolved = _resolve_run(
        run_id=str(args.run_id),
        runs_dir=runs_dir,
        run_dir_override=str(args.run_dir) if args.run_dir else None,
        db_override=str(args.db) if args.db else None,
    )

    print("=" * 70)
    print("AAM | Olmo Conformity | Posthoc Interpretability Pipeline")
    print("=" * 70)
    print(f"run_id={resolved.run_id}")
    print(f"run_dir={resolved.run_dir}")
    print(f"db={resolved.db_path}")
    print(f"temperature={resolved.temperature}")

    trace_db = TraceDb(TraceDbConfig(db_path=str(resolved.db_path)))
    trace_db.connect()
    trace_db.init_schema()

    # 0) Ensure trial_steps exist
    trial_steps_inserted = _ensure_trial_steps(trace_db=trace_db, run_id=resolved.run_id)
    if trial_steps_inserted:
        print(f"[OK] Inserted conformity_trial_steps rows: {trial_steps_inserted}")
    else:
        print("[OK] conformity_trial_steps already present (or no trials found)")

    # 1) Capture activations for behavioral trials
    capture_layers = _parse_csv_ints(str(args.capture_layers))
    if not bool(args.skip_activations):
        print("\n[Step 1/4] Backfilling behavioral activations...")
        cap_stats = backfill_behavioral_activations(
            trace_db=trace_db,
            run_id=resolved.run_id,
            run_dir=resolved.run_dir,
            capture_layers=capture_layers,
            capture_component=str(args.capture_component),
            capture_dtype=str(args.capture_dtype),
            skip_existing=True,
        )
        print(f"[OK] Activation backfill summary: captured={cap_stats['captured_trials']} skipped={cap_stats['skipped_trials']}")
    else:
        cap_stats = {"skipped": True}
        print("\n[Step 1/4] Skipping activation backfill (--skip-activations)")

    # 2) Vector analysis (truth/social probes + projections) per variant
    if not bool(args.skip_vector_analysis):
        print("\n[Step 2/4] Running vector analysis (per variant)...")
        probe_layers = _parse_csv_ints(str(args.probe_layers))
        vector_results = run_vector_analysis_all_variants(
            trace_db=trace_db,
            run_id=resolved.run_id,
            artifacts_dir=resolved.artifacts_dir,
            truth_probe_dataset_path=str(args.truth_probe_dataset),
            social_probe_dataset_path=str(args.social_probe_dataset),
            probe_layers=probe_layers,
            component=str(args.probe_component),
            token_position=int(args.probe_token_position),
            dtype=str(args.probe_dtype),
            temperature=float(resolved.temperature),
        )
        print("[OK] Vector analysis complete")
    else:
        vector_results = {"skipped": True, "by_variant": {}}
        print("\n[Step 2/4] Skipping vector analysis (--skip-vector-analysis)")

    # 3) Interventions (targeted battery)
    if bool(args.skip_interventions) or str(args.intervention_battery) == "none":
        intervention_results = {"skipped": True}
        print("\n[Step 3/4] Skipping interventions")
    else:
        print("\n[Step 3/4] Running interventions...")
        intervention_layers = _parse_csv_ints(str(args.intervention_layers))
        intervention_alphas = _parse_csv_floats(str(args.intervention_alphas))
        if str(args.intervention_battery) == "battery1":
            intervention_results = run_interventions_battery1(
                trace_db=trace_db,
                run_id=resolved.run_id,
                vector_results=vector_results,
                intervention_layers=intervention_layers,
                intervention_alphas=intervention_alphas,
                temperature=float(resolved.temperature),
                component_hook=str(args.intervention_component_hook),
            )
        else:
            intervention_results = {"skipped": True, "reason": f"unknown battery: {args.intervention_battery}"}

        print("[OK] Interventions complete")

    # 4) Reports + scientific report
    if bool(args.skip_reports):
        report_paths: Dict[str, str] = {}
        scientific_report_path: Optional[str] = None
        print("\n[Step 4/4] Skipping reports (--skip-reports)")
    else:
        print("\n[Step 4/4] Regenerating report figures/tables...")
        report_paths = generate_core_figures(trace_db=trace_db, run_id=resolved.run_id, run_dir=str(resolved.run_dir))
        print(f"[OK] Generated {len(report_paths)} core figure(s)")

        print("\n[Step 4/4] Generating scientific report...")
        scientific_report_path = None
        try:
            reporter = ScientificReportGenerator(Path(resolved.run_dir))
            report = reporter.generate()
            report_path = Path(resolved.artifacts_dir) / "scientific_report.json"
            report.save(str(report_path))
            scientific_report_path = str(report_path)
            reporter.close()
            print(f"[OK] Scientific report saved: {scientific_report_path}")
            if report.dual_stack_risk:
                print("WARNING: dual-stack risk detected (different weights for inference vs probing)")
        except Exception as e:
            print(f"WARNING: scientific report generation failed: {e}")

    trace_db.close()

    # Write a compact run-level summary JSON for orchestration / auditing.
    summary = {
        "run_id": resolved.run_id,
        "run_dir": str(resolved.run_dir),
        "db": str(resolved.db_path),
        "temperature": float(resolved.temperature),
        "trial_steps_inserted": int(trial_steps_inserted),
        "activation_backfill": cap_stats,
        "vector_analysis": vector_results,
        "interventions": intervention_results,
        "report_paths": report_paths,
        "scientific_report_path": scientific_report_path,
        "completed_at": time.time(),
    }
    summary_path = Path(resolved.artifacts_dir) / "posthoc_interpretability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 70)
    print("Posthoc interpretability pipeline complete")
    print("=" * 70)
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
