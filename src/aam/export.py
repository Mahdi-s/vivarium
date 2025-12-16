from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from aam.persistence import TraceDb
from aam.types import TraceEvent


def export_trace_to_parquet(trace_db: TraceDb, *, run_id: str, output_path: str) -> None:
    """
    Export trace events to Parquet format for efficient analysis.

    PRD Section 9: Parquet is columnar, compressed, and strictly typed.
    It is 10-100x faster for analytical queries (Pandas/DuckDB).
    """
    try:
        import pandas as pd  # type: ignore
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Parquet export requires pandas and pyarrow. Install: pip install pandas pyarrow"
        ) from e

    # Fetch all trace events
    events = trace_db.fetch_trace_events(run_id=run_id)

    if not events:
        raise ValueError(f"No trace events found for run_id={run_id}")

    # Convert to DataFrame
    rows = []
    for event in events:
        row = {
            "trace_id": event.trace_id,
            "run_id": event.run_id,
            "time_step": event.time_step,
            "timestamp": event.timestamp,
            "agent_id": event.agent_id,
            "action_type": event.action_type,
            "info_json": json.dumps(event.info, sort_keys=True),
            "outcome_json": json.dumps(event.outcome, sort_keys=True),
            "environment_state_hash": event.environment_state_hash,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Write to Parquet
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")

    print(f"Exported {len(rows)} trace events to {output_path}")


def export_messages_to_parquet(trace_db: TraceDb, *, run_id: str, output_path: str) -> None:
    """Export messages table to Parquet format."""
    try:
        import pandas as pd  # type: ignore
    except ImportError as e:
        raise RuntimeError("Parquet export requires pandas. Install: pip install pandas pyarrow") from e

    rows = trace_db.conn.execute(
        """
        SELECT message_id, run_id, time_step, author_id, content, created_at
        FROM messages
        WHERE run_id = ?
        ORDER BY time_step ASC, created_at ASC;
        """,
        (run_id,),
    ).fetchall()

    if not rows:
        raise ValueError(f"No messages found for run_id={run_id}")

    df = pd.DataFrame([dict(r) for r in rows])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")

    print(f"Exported {len(rows)} messages to {output_path}")


def export_table_to_parquet(trace_db: TraceDb, *, query: str, params: tuple, output_path: str) -> None:
    """
    Generic Parquet export helper for conformity_* and other analytic tables.
    """
    try:
        import pandas as pd  # type: ignore
    except ImportError as e:
        raise RuntimeError("Parquet export requires pandas and pyarrow. Install: pip install pandas pyarrow") from e

    rows = trace_db.conn.execute(query, params).fetchall()
    if not rows:
        raise ValueError("Query returned no rows")

    df = pd.DataFrame([dict(r) for r in rows])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")

    print(f"Exported {len(df)} rows to {output_path}")

