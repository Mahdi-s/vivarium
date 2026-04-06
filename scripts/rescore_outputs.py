#!/usr/bin/env python3
"""
Re-score conformity_outputs in-place using the enhanced 4-phase extraction cascade.

Architecture: Decoupled Producer-Consumer with dynamic GPU batching.

    Main Thread (Producer)        — reads rows in 50k chunks from each simulation.db
    14 × CPU Workers (Process)    — run Phase A-D parsing (pure CPU)
    1 × GPU Batcher (Thread)      — accumulates embedding requests, fires at batch_size or timeout
    1 × DB Writer (Thread)        — drains output queue, writes in executemany(500) transactions

Usage:
    # Original metadata-driven mode (runs_latest/runs/)
    python scripts/rescore_outputs.py --metadata Comparing_Experiments/runs_metadata_v6.json

    # Restrict to a single run via metadata
    python scripts/rescore_outputs.py --run-id 9f240f89-... --force

    # Auto-discover all simulation.db files under one or more directories
    python scripts/rescore_outputs.py --discover --runs-dir runs --runs-dir runs/think

    # Discover + only rescore specific variants (e.g. skip think variants)
    python scripts/rescore_outputs.py \\
        --discover --runs-dir runs_latest/runs \\
        --variant-filter base,instruct,instruct_sft,instruct_dpo,rl_zero

    # Full sweep: runs_latest (non-think) + runs/ + runs/think/ (all)
    python scripts/rescore_outputs.py \\
        --discover --runs-dir runs_latest/runs \\
        --variant-filter base,instruct,instruct_sft,instruct_dpo,rl_zero
    python scripts/rescore_outputs.py --discover --runs-dir runs --runs-dir runs/think
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import re
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SENTINEL = None  # poison pill for queues
_DB_CHUNK_SIZE = 50_000
_WRITE_BATCH_SIZE = 500
_GPU_BATCH_SIZE = 1024
_GPU_TIMEOUT_S = 0.2

# ---------------------------------------------------------------------------
# Worker function (runs in child process)
# ---------------------------------------------------------------------------

def _init_worker():
    """Per-process init: import the scoring module once."""
    global _score_fn
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from vivarium.experiments.olmo_conformity.enhanced_scoring import score_single_output
    _score_fn = score_single_output


def _process_row(item: Tuple) -> Tuple[str, str, Optional[int], int]:
    """
    Score one row and return (output_id, parsed_answer_text, is_correct, refusal_flag).

    Item tuple layout:
      (output_id, raw_text, ground_truth_text, wrong_answer, condition_name, dataset_name, source_json_str)
    """
    (output_id, raw_text, ground_truth_text, wrong_answer,
     condition_name, dataset_name, source_json_str) = item

    source_json = None
    if source_json_str:
        try:
            source_json = json.loads(source_json_str)
        except (json.JSONDecodeError, TypeError):
            pass

    result = _score_fn(
        raw_text=raw_text or "",
        ground_truth_text=ground_truth_text,
        wrong_answer=wrong_answer,
        condition_name=condition_name or "control",
        dataset_name=dataset_name or "",
        source_json=source_json,
    )

    return (output_id, result.parsed_answer_text, result.is_correct, result.refusal_flag)


def _worker_loop(input_q: mp.Queue, output_q: mp.Queue):
    """Drain input_q, score each row, push results to output_q."""
    _init_worker()
    while True:
        item = input_q.get()
        if item is _SENTINEL:
            break
        try:
            result = _process_row(item)
            output_q.put(result)
        except Exception as exc:
            output_id = item[0] if item else "unknown"
            print(f"  [worker error] output_id={output_id}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# DB Writer thread
# ---------------------------------------------------------------------------

class DBWriter(threading.Thread):
    def __init__(self, output_q: mp.Queue, db_path: str, total_expected: int):
        super().__init__(daemon=True)
        self.output_q = output_q
        self.db_path = db_path
        self.total_expected = total_expected
        self.written = 0
        self._stop_event = threading.Event()

    def run(self):
        conn = sqlite3.connect(self.db_path)
        batch: List[Tuple[str, Optional[int], int, str]] = []
        last_report = time.monotonic()

        while not self._stop_event.is_set() or not self.output_q.empty():
            try:
                item = self.output_q.get(timeout=0.5)
            except queue.Empty:
                if batch:
                    self._flush(conn, batch)
                    batch.clear()
                continue

            if item is _SENTINEL:
                break

            output_id, parsed_text, is_correct, refusal_flag = item
            batch.append((parsed_text, is_correct, refusal_flag, output_id))

            if len(batch) >= _WRITE_BATCH_SIZE:
                self._flush(conn, batch)
                batch.clear()

            now = time.monotonic()
            if now - last_report >= 5.0:
                pct = (self.written / self.total_expected * 100) if self.total_expected else 0
                print(f"  [writer] {self.written:,}/{self.total_expected:,} ({pct:.1f}%)")
                last_report = now

        if batch:
            self._flush(conn, batch)

        conn.close()

    def _flush(self, conn: sqlite3.Connection, batch: List[Tuple]):
        conn.execute("BEGIN TRANSACTION")
        conn.executemany(
            "UPDATE conformity_outputs SET parsed_answer_text=?, is_correct=?, refusal_flag=? WHERE output_id=?",
            batch,
        )
        conn.execute("COMMIT")
        self.written += len(batch)

    def signal_stop(self):
        self._stop_event.set()


# ---------------------------------------------------------------------------
# GPU Batcher thread (optional)
# ---------------------------------------------------------------------------

class GPUBatcher(threading.Thread):
    """
    Accumulates embedding requests and fires in batches.

    For the current dataset this is rarely needed — the rule-based pipeline
    resolves most items. The batcher is activated only with --use-embeddings.
    """
    def __init__(self, gpu_q: mp.Queue, output_q: mp.Queue, model_name: str):
        super().__init__(daemon=True)
        self.gpu_q = gpu_q
        self.output_q = output_q
        self.model_name = model_name
        self._stop_event = threading.Event()
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            if hasattr(self._model, "start_multi_process_pool"):
                pass  # DataParallel handled internally by encode
            print(f"  [gpu] Loaded embedding model: {self.model_name}")
        except ImportError:
            print("  [gpu] sentence-transformers not installed; GPU batcher disabled", file=sys.stderr)

    def run(self):
        self._load_model()
        batch: list = []
        last_fire = time.monotonic()

        while not self._stop_event.is_set() or not self.gpu_q.empty():
            try:
                item = self.gpu_q.get(timeout=_GPU_TIMEOUT_S)
            except queue.Empty:
                if batch and (time.monotonic() - last_fire) >= _GPU_TIMEOUT_S:
                    self._fire_batch(batch)
                    batch.clear()
                    last_fire = time.monotonic()
                continue

            if item is _SENTINEL:
                break
            batch.append(item)
            if len(batch) >= _GPU_BATCH_SIZE:
                self._fire_batch(batch)
                batch.clear()
                last_fire = time.monotonic()

        if batch:
            self._fire_batch(batch)

    def _fire_batch(self, batch: list):
        if not self._model or not batch:
            for item in batch:
                self.output_q.put(item)
            return
        # Placeholder: actual embedding comparison would go here
        for item in batch:
            self.output_q.put(item)

    def signal_stop(self):
        self._stop_event.set()


# ---------------------------------------------------------------------------
# Metadata & DB loading
# ---------------------------------------------------------------------------

def load_runs_metadata(metadata_path: Path) -> Dict[str, Dict[str, Any]]:
    meta = json.loads(metadata_path.read_text())
    out: Dict[str, Dict[str, Any]] = {}
    for temp_str, info in meta.get("experiments", {}).items():
        if info.get("status") != "completed":
            continue
        out[temp_str] = {
            "temperature": float(temp_str),
            "run_id": str(info["run_id"]),
            "run_dir": str(info["run_dir"]),
        }
    return dict(sorted(out.items(), key=lambda kv: float(kv[0])))


_ROW_QUERY = """
SELECT
    o.output_id,
    o.raw_text,
    i.ground_truth_text,
    json_extract(i.source_json, '$.wrong_answer') AS wrong_answer,
    c.name AS condition_name,
    d.name AS dataset_name,
    i.source_json
FROM conformity_outputs o
JOIN conformity_trials t ON t.trial_id = o.trial_id
JOIN conformity_items i ON i.item_id = t.item_id
JOIN conformity_datasets d ON d.dataset_id = i.dataset_id
JOIN conformity_conditions c ON c.condition_id = t.condition_id
WHERE t.run_id = ?{variant_clause}
"""

def _count_rows(db_path: str, run_id: str, allowed_variants: Optional[List[str]] = None) -> int:
    conn = sqlite3.connect(db_path)
    variant_clause = ""
    params: list = [run_id]
    if allowed_variants:
        placeholders = ",".join("?" * len(allowed_variants))
        variant_clause = f" AND t.variant IN ({placeholders})"
        params.extend(allowed_variants)
    count = conn.execute(
        "SELECT COUNT(*) FROM conformity_outputs o "
        f"JOIN conformity_trials t ON t.trial_id=o.trial_id WHERE t.run_id=?{variant_clause}",
        params,
    ).fetchone()[0]
    conn.close()
    return count


def _iter_rows(db_path: str, run_id: str, allowed_variants: Optional[List[str]] = None):
    """Yield tuples of (output_id, raw_text, gt, wa, cond, dataset, source_json) in chunks."""
    variant_clause = ""
    params: list = [run_id]
    if allowed_variants:
        placeholders = ",".join("?" * len(allowed_variants))
        variant_clause = f" AND t.variant IN ({placeholders})"
        params.extend(allowed_variants)
    query = _ROW_QUERY.format(variant_clause=variant_clause)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(query, params)

    while True:
        rows = cursor.fetchmany(_DB_CHUNK_SIZE)
        if not rows:
            break
        for row in rows:
            yield (
                row["output_id"],
                row["raw_text"],
                row["ground_truth_text"],
                row["wrong_answer"],
                row["condition_name"],
                row["dataset_name"],
                row["source_json"],
            )

    conn.close()


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def rescore_db(
    db_path: str,
    run_id: str,
    *,
    n_workers: int = 14,
    use_embeddings: bool = False,
    embedding_model: str = "all-MiniLM-L6-v2",
    allowed_variants: Optional[List[str]] = None,
) -> int:
    """Rescore a single simulation.db. Returns number of rows updated."""
    total = _count_rows(db_path, run_id, allowed_variants)
    if total == 0:
        print(f"  No rows to rescore")
        return 0

    variant_label = f" (variants: {','.join(allowed_variants)})" if allowed_variants else ""
    print(f"  Rescoring {total:,} rows with {n_workers} workers{variant_label} ...")

    ctx = mp.get_context("spawn")
    input_q: mp.Queue = ctx.Queue(maxsize=n_workers * 200)
    output_q: mp.Queue = ctx.Queue(maxsize=n_workers * 200)
    gpu_q: mp.Queue = ctx.Queue() if use_embeddings else ctx.Queue(maxsize=1)

    workers = []
    for _ in range(n_workers):
        p = ctx.Process(target=_worker_loop, args=(input_q, output_q))
        p.start()
        workers.append(p)

    writer = DBWriter(output_q, db_path, total)
    writer.start()

    gpu_batcher: Optional[GPUBatcher] = None
    if use_embeddings:
        gpu_batcher = GPUBatcher(gpu_q, output_q, embedding_model)
        gpu_batcher.start()

    t0 = time.monotonic()
    fed = 0
    try:
        for item in _iter_rows(db_path, run_id, allowed_variants):
            input_q.put(item)
            fed += 1
    except KeyboardInterrupt:
        print("\n  [interrupted] Draining queues ...")

    for _ in workers:
        input_q.put(_SENTINEL)
    for p in workers:
        p.join(timeout=60)

    if gpu_batcher:
        gpu_q.put(_SENTINEL)
        gpu_batcher.signal_stop()
        gpu_batcher.join(timeout=30)

    # Wait for writer to drain
    while not output_q.empty():
        time.sleep(0.2)
    output_q.put(_SENTINEL)
    writer.signal_stop()
    writer.join(timeout=60)

    elapsed = time.monotonic() - t0
    rate = writer.written / elapsed if elapsed > 0 else 0
    print(f"  Done: {writer.written:,} rows in {elapsed:.1f}s ({rate:,.0f} rows/s)")
    return writer.written


# ---------------------------------------------------------------------------
# Discovery (no-metadata mode)
# ---------------------------------------------------------------------------

_RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}_[0-9a-f\-]{36}$")


def discover_runs(runs_dirs: List[Path]) -> List[Tuple[str, Path]]:
    """Return deduplicated [(run_id, db_path), ...] from all supplied roots.

    Iterates each directory one level deep for folders matching
    YYYYMMDD_HHMMSS_<uuid> that contain a simulation.db.
    The same run_id appearing under multiple roots is de-duplicated;
    first occurrence wins.
    """
    seen: Dict[str, Path] = {}
    results: List[Tuple[str, Path]] = []

    for runs_dir in runs_dirs:
        if not runs_dir.exists():
            print(f"  [warn] runs-dir not found, skipping: {runs_dir}")
            continue

        # The directory itself might be a single run folder
        if _RUN_DIR_RE.match(runs_dir.name):
            db = runs_dir / "simulation.db"
            if db.exists():
                run_id = runs_dir.name.split("_", 2)[-1]
                if run_id not in seen:
                    seen[run_id] = db
                    results.append((run_id, db))
            continue

        for entry in sorted(runs_dir.iterdir()):
            if not entry.is_dir() or not _RUN_DIR_RE.match(entry.name):
                continue
            db = entry / "simulation.db"
            if not db.exists():
                print(f"  [skip] {entry.name} — simulation.db not found")
                continue
            run_id = entry.name.split("_", 2)[-1]
            if run_id in seen:
                print(f"  [dedup] run_id={run_id[:8]}… already seen, skipping {entry}")
                continue
            seen[run_id] = db
            results.append((run_id, db))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Re-score conformity outputs with enhanced cascade",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Source selection
    ap.add_argument(
        "--runs-dir",
        dest="runs_dirs",
        action="append",
        metavar="DIR",
        help=(
            "Directory containing run folders (YYYYMMDD_HHMMSS_<uuid>). "
            "May be repeated to process multiple trees. "
            "Used in --discover mode; ignored when --metadata is used without --discover."
        ),
    )
    ap.add_argument(
        "--discover",
        action="store_true",
        help=(
            "Auto-discover run folders under --runs-dir trees instead of reading "
            "a metadata JSON. Required when targeting runs/ or runs/think/."
        ),
    )
    ap.add_argument(
        "--metadata",
        type=str,
        default="Comparing_Experiments/runs_metadata_v6.json",
        help="Path to runs_metadata JSON (used in legacy metadata mode, default: %(default)s).",
    )

    # Filtering
    ap.add_argument(
        "--variant-filter",
        type=str,
        default=None,
        metavar="VARIANTS",
        help=(
            "Comma-separated list of variant names to rescore "
            "(e.g. 'base,instruct,instruct_sft,instruct_dpo,rl_zero'). "
            "When set, only matching rows are rescored. Default: all variants."
        ),
    )
    ap.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Restrict to a single run_id (metadata mode only).",
    )

    # Performance
    ap.add_argument("--n-workers", type=int, default=14)
    ap.add_argument("--use-embeddings", action="store_true")
    ap.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2")

    args = ap.parse_args()

    allowed_variants: Optional[List[str]] = None
    if args.variant_filter:
        allowed_variants = [v.strip() for v in args.variant_filter.split(",") if v.strip()]

    total_updated = 0

    if args.discover:
        # ---- Discovery mode --------------------------------------------------
        runs_dirs_raw: List[str] = args.runs_dirs or ["runs"]
        runs_dirs = [Path(d).expanduser().resolve() for d in runs_dirs_raw]

        print(f"Discovering runs under: {[str(d) for d in runs_dirs]}")
        runs = discover_runs(runs_dirs)
        if not runs:
            print("No valid run directories found. Exiting.")
            return 1

        variant_label = f"  variant_filter : {','.join(allowed_variants)}\n" if allowed_variants else ""
        print(
            f"\n{len(runs)} run(s) found.\n"
            f"{variant_label}"
            f"  n_workers      : {args.n_workers}\n"
        )

        for run_id, db_path in runs:
            print(f"\n=== run_id={run_id[:12]}…  db={db_path} ===")
            updated = rescore_db(
                str(db_path),
                run_id,
                n_workers=args.n_workers,
                use_embeddings=args.use_embeddings,
                embedding_model=args.embedding_model,
                allowed_variants=allowed_variants,
            )
            total_updated += updated

        print(f"\nTotal updated: {total_updated:,} rows across {len(runs)} run(s)")

    else:
        # ---- Legacy metadata mode --------------------------------------------
        metadata = load_runs_metadata(Path(args.metadata))
        runs_dir = Path(args.runs_dirs[0] if args.runs_dirs else "runs_latest/runs")

        if args.run_id:
            targets = {k: v for k, v in metadata.items() if v["run_id"] == args.run_id}
            if not targets:
                print(f"Run ID {args.run_id} not found in metadata", file=sys.stderr)
                return 1
        else:
            targets = metadata

        variant_label = f"  variant_filter : {','.join(allowed_variants)}\n" if allowed_variants else ""
        print(
            f"\n{len(targets)} run(s) from metadata.\n"
            f"{variant_label}"
            f"  n_workers      : {args.n_workers}\n"
        )

        for temp_str, info in targets.items():
            db_path = str(runs_dir / info["run_dir"] / "simulation.db")
            if not Path(db_path).exists():
                print(f"[SKIP] Missing DB: {db_path}")
                continue

            print(f"\n=== T={info['temperature']} run={info['run_id'][:12]}... ===")
            updated = rescore_db(
                db_path,
                info["run_id"],
                n_workers=args.n_workers,
                use_embeddings=args.use_embeddings,
                embedding_model=args.embedding_model,
                allowed_variants=allowed_variants,
            )
            total_updated += updated

        print(f"\nTotal updated: {total_updated:,} rows across {len(targets)} run(s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
