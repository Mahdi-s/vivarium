#!/usr/bin/env python3
"""
Backfill probe projections for existing runs without re-running model inference.

Context:
Some HPC posthoc interpretability jobs completed activation capture but failed
before finishing probe training / projection computation. This script "replays"
probe projections by applying a set of *canonical* per-variant probe weights to
captured activations and writing the resulting projection rows into each run DB.

Important:
- This does NOT train probes (no GPU/inference required).
- It assumes each run has `activations/step_XXXXXX.safetensors` shards containing
  per-layer `hook_resid_post` vectors for each trial.
- It inserts new `conformity_probes` rows (truth + social per variant) plus
  `conformity_probe_projections` for behavioral trials only.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


from aam.persistence import TraceDb, TraceDbConfig  # noqa: E402


BEHAVIORAL_CONDITIONS = ("control", "asch_history_5", "authoritative_bias")
VARIANTS: Tuple[str, ...] = ("base", "instruct", "instruct_sft", "think", "think_sft", "rl_zero")
VARIANT_TO_MODEL_ID: Dict[str, str] = {
    "base": "allenai/Olmo-3-1025-7B",
    "instruct": "allenai/Olmo-3-7B-Instruct",
    "instruct_sft": "allenai/Olmo-3-7B-Instruct-SFT",
    "think": "allenai/Olmo-3-7B-Think",
    "think_sft": "allenai/Olmo-3-7B-Think-SFT",
    "rl_zero": "allenai/Olmo-3-7B-RL-Zero-Math",
}


@dataclass(frozen=True)
class RunRef:
    run_id: str
    run_dir: Path
    db_path: Path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_run_dir(*, runs_dir: Path, run_id: str) -> Path:
    matches = sorted(runs_dir.glob(f"*_{run_id}"), key=lambda p: p.name, reverse=True)
    if not matches:
        raise FileNotFoundError(f"Could not find run_dir under {runs_dir} ending with _{run_id}")
    return matches[0]


def _is_sqlite_db(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(16)
        return head == b"SQLite format 3\x00"
    except Exception:
        return False


def _sqlite_quick_check_ok(path: Path) -> bool:
    if not _is_sqlite_db(path):
        return False
    try:
        conn = sqlite3.connect(str(path))
        row = conn.execute("PRAGMA quick_check;").fetchone()
        conn.close()
        return bool(row) and str(row[0]).strip().lower() == "ok"
    except Exception:
        return False


def _repair_db_if_needed(*, target_db: Path, fallback_db: Path) -> None:
    if _sqlite_quick_check_ok(target_db):
        return

    if not fallback_db.exists() or not _is_sqlite_db(fallback_db):
        raise RuntimeError(f"Target DB is corrupt and fallback DB is missing/invalid: {fallback_db}")

    backup_path = target_db.with_suffix(target_db.suffix + ".corrupt")
    if target_db.exists():
        target_db.replace(backup_path)

    shutil.copy2(fallback_db, target_db)


def _load_trials_index(*, conn: sqlite3.Connection, run_id: str) -> Dict[str, Dict[str, str]]:
    """
    Build prefix->trial mapping (prefix is first 8 chars of trial_id) for fast lookup.
    Returns dict[prefix] = {trial_id, variant, model_id, condition_name}
    """
    rows = conn.execute(
        """
        SELECT
          t.trial_id,
          t.variant,
          t.model_id,
          c.name AS condition_name
        FROM conformity_trials t
        JOIN conformity_conditions c ON c.condition_id = t.condition_id
        WHERE t.run_id = ?;
        """,
        (str(run_id),),
    ).fetchall()

    out: Dict[str, Dict[str, str]] = {}
    for r in rows:
        trial_id = str(r[0])
        prefix = trial_id[:8]
        if prefix in out and out[prefix]["trial_id"] != trial_id:
            raise RuntimeError(f"Non-unique trial_id prefix collision: {prefix}")
        out[prefix] = {
            "trial_id": trial_id,
            "variant": str(r[1]),
            "model_id": str(r[2]),
            "condition_name": str(r[3]),
        }
    return out


def _activation_files(run_dir: Path) -> List[Path]:
    act_dir = run_dir / "activations"
    if not act_dir.exists():
        return []
    return sorted(act_dir.glob("step_*.safetensors"))


def _extract_agent_id_from_keys(keys: Sequence[str]) -> str:
    # Keys look like: "<agent_id>.blocks.<layer>.hook_resid_post"
    # Behavioral trials use agent_id like "trial_<8-hex-prefix>" while probe-capture
    # trials often use a fixed agent_id (e.g. "probe_agent"). We only need the
    # behavioral ones, so the caller can filter on agent_id prefix.
    for k in keys:
        if ".blocks." in k:
            return k.split(".blocks.", 1)[0]
    raise RuntimeError("Could not find expected activation tensor key format (missing '*.blocks.*').")


def _delete_existing_probe_rows(*, conn: sqlite3.Connection, run_id: str) -> None:
    # projections table doesn't have run_id, so delete by trial_id set
    with conn:
        conn.execute(
            """
            DELETE FROM conformity_probe_projections
            WHERE trial_id IN (SELECT trial_id FROM conformity_trials WHERE run_id = ?);
            """,
            (str(run_id),),
        )
        conn.execute("DELETE FROM conformity_probes WHERE run_id = ?;", (str(run_id),))


def _upsert_dataset_row(
    *, conn: sqlite3.Connection, dataset_id: str, name: str, version: str, path: str, sha256: str
) -> None:
    ts = time.time()
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO conformity_datasets(dataset_id, name, version, path, sha256, created_at)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (str(dataset_id), str(name), str(version), str(path), str(sha256), float(ts)),
        )


@dataclass(frozen=True)
class CanonicalProbe:
    probe_kind: str  # "truth" | "social"
    variant: str
    model_id: str
    layers: List[int]
    component: str
    token_position: int
    artifact_path: Path
    metrics: Dict[str, object]


def _load_canonical_probes(*, source_run_dir: Path, variants: Iterable[str]) -> Dict[Tuple[str, str], CanonicalProbe]:
    """
    Load probe metadata + artifact paths from a *completed* source run directory.

    Returns dict[(variant, kind)] -> CanonicalProbe
    """
    src_db = source_run_dir / "simulation.db"
    if not src_db.exists():
        raise FileNotFoundError(f"Missing source simulation.db: {src_db}")

    conn = sqlite3.connect(str(src_db))
    conn.row_factory = sqlite3.Row
    probes = conn.execute(
        "SELECT probe_kind, model_id, layers_json, component, token_position, artifact_path, metrics_json "
        "FROM conformity_probes WHERE run_id = ?;",
        (source_run_dir.name.split("_", 2)[-1],),  # not reliable; we'll match by file existence below
    ).fetchall()
    conn.close()

    # The run_id lookup above is fragile; prefer filesystem canonical artifacts.
    # We still want metrics; if we can't read them via DB, we'll use empty dicts.
    metrics_by_model_kind: Dict[Tuple[str, str], Dict[str, object]] = {}
    try:
        conn = sqlite3.connect(str(src_db))
        conn.row_factory = sqlite3.Row
        src_run_id = conn.execute("SELECT run_id FROM runs LIMIT 1;").fetchone()["run_id"]
        rows = conn.execute(
            "SELECT probe_kind, model_id, metrics_json FROM conformity_probes WHERE run_id = ?;", (str(src_run_id),)
        ).fetchall()
        import json

        for r in rows:
            try:
                metrics_by_model_kind[(str(r["model_id"]), str(r["probe_kind"]))] = json.loads(str(r["metrics_json"]))
            except Exception:
                metrics_by_model_kind[(str(r["model_id"]), str(r["probe_kind"]))] = {}
        conn.close()
    except Exception:
        metrics_by_model_kind = {}

    out: Dict[Tuple[str, str], CanonicalProbe] = {}
    layers = list(range(32))
    for variant in variants:
        model_id = VARIANT_TO_MODEL_ID[variant]
        for kind in ("truth", "social"):
            artifact = source_run_dir / "artifacts" / f"{kind}_probe_{variant}.safetensors"
            if not artifact.exists():
                raise FileNotFoundError(f"Missing canonical probe artifact: {artifact}")
            out[(variant, kind)] = CanonicalProbe(
                probe_kind=kind,
                variant=variant,
                model_id=model_id,
                layers=layers,
                component="hook_resid_post",
                token_position=-1,
                artifact_path=artifact,
                metrics=metrics_by_model_kind.get((model_id, kind), {}),
            )
    return out


def _copy_canonical_artifacts(*, canonical: Dict[Tuple[str, str], CanonicalProbe], target_artifacts_dir: Path) -> None:
    target_artifacts_dir.mkdir(parents=True, exist_ok=True)
    for (variant, kind), p in canonical.items():
        dst = target_artifacts_dir / f"{kind}_probe_{variant}.safetensors"
        if not dst.exists():
            shutil.copy2(p.artifact_path, dst)


def _insert_probes(
    *,
    trace_db: TraceDb,
    run_id: str,
    canonical: Dict[Tuple[str, str], CanonicalProbe],
    truth_dataset_id: str,
    social_dataset_id: str,
    target_artifacts_dir: Path,
) -> Dict[Tuple[str, str], str]:
    probe_ids: Dict[Tuple[str, str], str] = {}
    for variant in VARIANTS:
        for kind in ("truth", "social"):
            cp = canonical[(variant, kind)]
            probe_id = str(uuid.uuid4())
            artifact_path = str(target_artifacts_dir / f"{kind}_probe_{variant}.safetensors")
            train_dataset_id = truth_dataset_id if kind == "truth" else social_dataset_id
            trace_db.insert_conformity_probe(
                probe_id=probe_id,
                run_id=str(run_id),
                probe_kind=str(kind),
                train_dataset_id=str(train_dataset_id),
                model_id=str(cp.model_id),
                layers=list(cp.layers),
                component=str(cp.component),
                token_position=int(cp.token_position),
                artifact_path=str(artifact_path),
                metrics=dict(cp.metrics or {}),
            )
            probe_ids[(variant, kind)] = probe_id
    return probe_ids


def _load_weights_matrix(*, path: Path, layers: Sequence[int]) -> Tuple["torch.Tensor", "torch.Tensor"]:
    import torch
    from safetensors.torch import load_file

    weights = load_file(str(path))
    w_list = []
    b_list = []
    for layer in layers:
        w = weights.get(f"layer_{layer}.weight")
        b = weights.get(f"layer_{layer}.bias")
        if w is None or b is None:
            raise RuntimeError(f"Missing layer weights in {path} for layer={layer}")
        w_list.append(w.detach().to(torch.float32))
        b_list.append(b.detach().to(torch.float32).reshape(()))
    W = torch.stack(w_list, dim=0)  # [L, d]
    B = torch.stack(b_list, dim=0)  # [L]
    return W, B


def _compute_and_insert_projections(
    *,
    trace_db: TraceDb,
    run_id: str,
    run_dir: Path,
    probe_ids: Dict[Tuple[str, str], str],
    weight_cache: Dict[Tuple[str, str], Tuple["torch.Tensor", "torch.Tensor"]],
    layers: Sequence[int],
    behavioral_conditions: Sequence[str],
    batch_size: int = 20_000,
) -> int:
    import torch
    from safetensors.torch import load_file

    # Preload trial metadata for filtering / variant routing.
    conn = trace_db.conn
    trial_rows = conn.execute(
        """
        SELECT
          t.trial_id,
          t.variant,
          c.name AS condition_name
        FROM conformity_trials t
        JOIN conformity_conditions c ON c.condition_id = t.condition_id
        WHERE t.run_id = ?;
        """,
        (str(run_id),),
    ).fetchall()
    trial_meta = {str(r["trial_id"]): (str(r["variant"]), str(r["condition_name"])) for r in trial_rows}

    # Fast prefix lookup for mapping activation keys -> trial_id
    prefix_index: Dict[str, str] = {}
    for trial_id, (variant, cond) in trial_meta.items():
        if cond not in behavioral_conditions:
            continue
        prefix = trial_id[:8]
        prefix_index[prefix] = trial_id

    total_inserted = 0
    pending: List[Tuple[str, str, str, int, Optional[int], float]] = []

    for act_path in _activation_files(run_dir):
        buf = load_file(str(act_path))
        agent_id = _extract_agent_id_from_keys(list(buf.keys()))
        if not str(agent_id).startswith("trial_"):
            continue
        prefix = agent_id.split("_", 1)[-1]
        trial_id = prefix_index.get(prefix)
        if not trial_id:
            continue  # either non-behavioral trial or unknown

        variant, _cond = trial_meta[trial_id]
        if variant not in VARIANT_TO_MODEL_ID:
            continue

        # Assemble activation matrix [L, d] once per trial
        A_list: List[torch.Tensor] = []
        for layer in layers:
            k = f"{agent_id}.blocks.{layer}.hook_resid_post"
            v = buf.get(k)
            if v is None:
                raise RuntimeError(f"Missing activation key {k} in {act_path}")
            A_list.append(v.detach().to(torch.float32))
        A = torch.stack(A_list, dim=0)  # [L, d]

        for kind in ("truth", "social"):
            probe_id = probe_ids[(variant, kind)]
            W, B = weight_cache[(variant, kind)]
            scores = (A * W).sum(dim=1) + B  # [L]
            for layer, s in zip(layers, scores.tolist()):
                pending.append((str(uuid.uuid4()), str(trial_id), str(probe_id), int(layer), None, float(s)))

        if len(pending) >= batch_size:
            trace_db.insert_conformity_projection_rows(rows=pending)
            total_inserted += len(pending)
            pending.clear()

    if pending:
        trace_db.insert_conformity_projection_rows(rows=pending)
        total_inserted += len(pending)
        pending.clear()

    return total_inserted


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Backfill probe projections from canonical probe weights.")
    p.add_argument("--runs-dir", type=str, default="runs-hpc-full/probe/runs", help="Runs directory containing run folders.")
    p.add_argument("--fallback-runs-dir", type=str, default="runs-hpc-full/runs", help="Fallback runs directory for DB repair.")
    p.add_argument(
        "--canonical-run-id",
        type=str,
        default="dda9d6b3-a516-41b3-a85a-b424de8f15d3",
        help="Run ID that contains completed canonical probe artifacts.",
    )
    p.add_argument(
        "--target-run-ids",
        type=str,
        default="56478e99-7607-4957-9f53-a53b73a7e9d4,99127619-fcc7-4fd4-ba3a-cc810610249f,271bb5b2-572d-4ecd-8577-b07a7cd10846,eb777acc-3ab5-4f87-b073-249a50d25863,fa0b1d4f-d547-4094-b07c-4f9efc20f771",
        help="Comma-separated run IDs to backfill.",
    )
    p.add_argument("--skip-db-repair", action="store_true", help="Do not attempt to repair corrupt DBs from fallback runs.")
    p.add_argument("--batch-size", type=int, default=20_000)
    args = p.parse_args(argv)

    runs_dir = Path(str(args.runs_dir))
    fallback_runs_dir = Path(str(args.fallback_runs_dir))

    canonical_run_dir = _find_run_dir(runs_dir=runs_dir, run_id=str(args.canonical_run_id))
    canonical = _load_canonical_probes(source_run_dir=canonical_run_dir, variants=VARIANTS)

    # Load weights once.
    weight_cache: Dict[Tuple[str, str], Tuple["torch.Tensor", "torch.Tensor"]] = {}
    layers = list(range(32))
    for key, cp in canonical.items():
        weight_cache[key] = _load_weights_matrix(path=cp.artifact_path, layers=layers)

    # Compute dataset hashes once.
    truth_dataset_path = REPO_ROOT / "experiments" / "olmo_conformity" / "datasets" / "candidates" / "truth_probe_train.jsonl"
    social_dataset_path = REPO_ROOT / "experiments" / "olmo_conformity" / "datasets" / "candidates" / "social_probe_train.jsonl"
    truth_sha = _sha256_file(truth_dataset_path)
    social_sha = _sha256_file(social_dataset_path)

    target_ids = [s.strip() for s in str(args.target_run_ids).split(",") if s.strip()]
    for run_id in target_ids:
        run_dir = _find_run_dir(runs_dir=runs_dir, run_id=run_id)
        db_path = run_dir / "simulation.db"

        if not bool(args.skip_db_repair):
            fallback_db = _find_run_dir(runs_dir=fallback_runs_dir, run_id=run_id) / "simulation.db"
            _repair_db_if_needed(target_db=db_path, fallback_db=fallback_db)

        if not db_path.exists() or not _is_sqlite_db(db_path):
            raise RuntimeError(f"Missing/invalid sqlite DB after repair: {db_path}")

        print("=" * 70)
        print(f"[Backfill] run_id={run_id}")
        print(f"  run_dir={run_dir}")
        print(f"  db={db_path}")

        # Clear old probe rows for this run (safe for runs without interventions).
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA foreign_keys = ON;")
        _delete_existing_probe_rows(conn=conn, run_id=run_id)
        conn.close()

        # Copy canonical artifacts into this run.
        target_artifacts_dir = run_dir / "artifacts"
        _copy_canonical_artifacts(canonical=canonical, target_artifacts_dir=target_artifacts_dir)

        # Insert dataset + probes + projections.
        trace_db = TraceDb(TraceDbConfig(db_path=str(db_path)))
        trace_db.connect()
        trace_db.init_schema()

        # Insert datasets used as provenance for imported probes
        truth_dataset_id = str(uuid.uuid4())
        social_dataset_id = str(uuid.uuid4())
        _upsert_dataset_row(
            conn=trace_db.conn,
            dataset_id=truth_dataset_id,
            name="truth_probe_train_imported",
            version="v1",
            path=str(truth_dataset_path.relative_to(REPO_ROOT)),
            sha256=str(truth_sha),
        )
        _upsert_dataset_row(
            conn=trace_db.conn,
            dataset_id=social_dataset_id,
            name="social_probe_train_imported",
            version="v1",
            path=str(social_dataset_path.relative_to(REPO_ROOT)),
            sha256=str(social_sha),
        )

        probe_ids = _insert_probes(
            trace_db=trace_db,
            run_id=run_id,
            canonical=canonical,
            truth_dataset_id=truth_dataset_id,
            social_dataset_id=social_dataset_id,
            target_artifacts_dir=target_artifacts_dir,
        )

        inserted = _compute_and_insert_projections(
            trace_db=trace_db,
            run_id=run_id,
            run_dir=run_dir,
            probe_ids=probe_ids,
            weight_cache=weight_cache,
            layers=layers,
            behavioral_conditions=BEHAVIORAL_CONDITIONS,
            batch_size=int(args.batch_size),
        )
        trace_db.close()

        print(f"[OK] Inserted probe projections: {inserted}")

    print("=" * 70)
    print("Backfill complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
