from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from vivarium.types import RunMetadata, TraceEvent


def _json_dumps_deterministic(value: Dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def _json_dumps_any(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class TraceDbConfig:
    db_path: str


@dataclass
class ConformityOutputRow:
    """Buffered output row for batch insertion into conformity_outputs.

    Collect these during async trial fan-out, then flush with
    TraceDb.batch_write_conformity_trial_results() to reduce per-trial
    transaction overhead from O(N) to O(N/batch_size).
    """

    output_id: str
    trial_id: str
    raw_text: str
    parsed_answer_text: Optional[str]
    parsed_answer_json: Optional[Dict[str, Any]]
    is_correct: Optional[bool]
    refusal_flag: bool
    latency_ms: Optional[float]
    token_usage_json: Optional[Dict[str, Any]]
    created_at: Optional[float] = None


class TraceDb:
    def __init__(self, config: TraceDbConfig):
        self._config = config
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        if self._conn is not None:
            return
        conn = sqlite3.connect(self._config.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        self._conn = conn

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("TraceDb is not connected. Call connect() first.")
        return self._conn

    def init_schema(self) -> None:
        """
        Initialize the core Vivarium schema.

        This MUST remain experiment-agnostic. Experiment-specific tables should be
        created by separate init_*_schema() methods (e.g. init_conformity_schema()).
        """
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY,
              seed INTEGER NOT NULL,
              created_at REAL NOT NULL,
              config_json TEXT NOT NULL
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trace (
              trace_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              time_step INTEGER NOT NULL,
              agent_id TEXT NOT NULL,
              action_type TEXT NOT NULL,
              info_json TEXT NOT NULL,
              outcome_json TEXT NOT NULL,
              created_at REAL NOT NULL,
              environment_state_hash TEXT,
              FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_trace_run_step ON trace(run_id, time_step);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_trace_agent ON trace(run_id, agent_id);")

        # Phase 2 minimal domain state: shared message feed
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
              message_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              time_step INTEGER NOT NULL,
              author_id TEXT NOT NULL,
              content TEXT NOT NULL,
              created_at REAL NOT NULL,
              FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_run_step ON messages(run_id, time_step);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_run_author ON messages(run_id, author_id);")

        # Activation metadata table (for Phase 3 interpretability)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS activation_metadata (
              record_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              time_step INTEGER NOT NULL,
              agent_id TEXT NOT NULL,
              model_id TEXT NOT NULL,
              layer_index INTEGER NOT NULL,
              component TEXT NOT NULL,
              token_position INTEGER NOT NULL,
              shard_file_path TEXT NOT NULL,
              tensor_key TEXT NOT NULL,
              shape_json TEXT NOT NULL,
              dtype TEXT NOT NULL,
              created_at REAL NOT NULL,
              FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_activation_run_step ON activation_metadata(run_id, time_step);"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_activation_agent ON activation_metadata(run_id, agent_id);"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_activation_layer ON activation_metadata(run_id, layer_index, component);"
        )

        # Cryptographic provenance log (Merkle accumulator snapshots)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS merkle_log (
              merkle_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              time_step INTEGER NOT NULL,
              agent_id TEXT NOT NULL,
              prompt_hash TEXT NOT NULL,
              activation_hash TEXT NOT NULL,
              leaf_hash TEXT NOT NULL,
              merkle_root TEXT NOT NULL,
              created_at REAL NOT NULL,
              FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_merkle_run_step_agent ON merkle_log(run_id, time_step, agent_id);")

        # Agent state snapshots (optional; used when AgentStateSpace is provided)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_states (
              run_id TEXT NOT NULL,
              time_step INTEGER NOT NULL,
              agent_id TEXT NOT NULL,
              state_json TEXT NOT NULL,
              created_at REAL NOT NULL,
              PRIMARY KEY (run_id, time_step, agent_id),
              FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_states_run_step ON agent_states(run_id, time_step);")

        # Initialize the generic mechanistic-interpretability tables.
        self.init_interpretability_schema()

    def init_interpretability_schema(self) -> None:
        """
        Initialize the generic mechanistic-interpretability schema (vivarium_* tables).

        These tables are experiment-agnostic and keyed off the core trace table,
        allowing any experiment to record probes, interventions, steering, and
        activation-patching results without coupling to a specific experiment schema.
        """
        # Probe registry: probe weights stored on disk (safetensors) and indexed here
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vivarium_probes (
              probe_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              probe_kind TEXT NOT NULL,
              train_dataset_id TEXT,
              model_id TEXT NOT NULL,
              layers_json TEXT NOT NULL,
              component TEXT NOT NULL,
              token_position INTEGER NOT NULL,
              artifact_path TEXT NOT NULL,
              metrics_json TEXT NOT NULL,
              created_at REAL NOT NULL,
              FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vivarium_probes_run ON vivarium_probes(run_id, probe_kind, model_id);"
        )

        # Layerwise projections (scalar) against a probe
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vivarium_probe_projections (
              projection_id TEXT PRIMARY KEY,
              trace_id TEXT NOT NULL,
              probe_id TEXT NOT NULL,
              layer_index INTEGER NOT NULL,
              token_index INTEGER,
              value_float REAL NOT NULL,
              created_at REAL NOT NULL,
              FOREIGN KEY(trace_id) REFERENCES trace(trace_id),
              FOREIGN KEY(probe_id) REFERENCES vivarium_probes(probe_id)
            );
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vivarium_proj_trace ON vivarium_probe_projections(trace_id, probe_id);"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vivarium_proj_layer ON vivarium_probe_projections(probe_id, layer_index);"
        )

        # Think token traces
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vivarium_think_tokens (
              think_id TEXT PRIMARY KEY,
              trace_id TEXT NOT NULL,
              token_index INTEGER NOT NULL,
              token_text TEXT NOT NULL,
              token_id INTEGER,
              created_at REAL NOT NULL,
              FOREIGN KEY(trace_id) REFERENCES trace(trace_id)
            );
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vivarium_think_trace ON vivarium_think_tokens(trace_id, token_index);"
        )

        # Answer-level logprob probes (posthoc)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vivarium_answer_logprobs (
              trace_id TEXT NOT NULL,
              context_kind TEXT NOT NULL,
              candidate_kind TEXT NOT NULL,
              candidate_text TEXT NOT NULL,
              token_count INTEGER NOT NULL,
              logprob_sum REAL NOT NULL,
              logprob_mean REAL NOT NULL,
              first_token_id INTEGER,
              first_token_logprob REAL,
              metadata_json TEXT NOT NULL,
              created_at REAL NOT NULL,
              PRIMARY KEY(trace_id, context_kind, candidate_kind),
              FOREIGN KEY(trace_id) REFERENCES trace(trace_id)
            );
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vivarium_answerlog_trace ON vivarium_answer_logprobs(trace_id, context_kind);"
        )

        # Intervention definitions (activation steering)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vivarium_interventions (
              intervention_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              name TEXT NOT NULL,
              alpha REAL NOT NULL,
              target_layers_json TEXT NOT NULL,
              component TEXT NOT NULL,
              vector_probe_id TEXT NOT NULL,
              notes TEXT,
              created_at REAL NOT NULL,
              FOREIGN KEY(run_id) REFERENCES runs(run_id),
              FOREIGN KEY(vector_probe_id) REFERENCES vivarium_probes(probe_id)
            );
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vivarium_interventions_run ON vivarium_interventions(run_id, name);"
        )

        # Intervention results compare before/after outputs for the same trace event
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vivarium_intervention_results (
              result_id TEXT PRIMARY KEY,
              trace_id TEXT NOT NULL,
              intervention_id TEXT NOT NULL,
              output_id_before TEXT NOT NULL,
              output_id_after TEXT NOT NULL,
              flipped_to_truth INTEGER,
              created_at REAL NOT NULL,
              FOREIGN KEY(trace_id) REFERENCES trace(trace_id),
              FOREIGN KEY(intervention_id) REFERENCES vivarium_interventions(intervention_id)
            );
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vivarium_intervention_trace ON vivarium_intervention_results(trace_id, intervention_id);"
        )

        # Logit lens tug-of-war: P(truth) vs P(sycophantic) per (trace, layer)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vivarium_logit_lens_tug_of_war (
              tow_id TEXT PRIMARY KEY,
              trace_id TEXT NOT NULL,
              layer_index INTEGER NOT NULL,
              truth_token TEXT NOT NULL,
              truth_token_id INTEGER,
              truth_prob REAL NOT NULL,
              sycophantic_token TEXT NOT NULL,
              sycophantic_token_id INTEGER,
              sycophantic_prob REAL NOT NULL,
              crossing_flag INTEGER NOT NULL DEFAULT 0,
              created_at REAL NOT NULL,
              FOREIGN KEY(trace_id) REFERENCES trace(trace_id)
            );
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vivarium_tow_trace ON vivarium_logit_lens_tug_of_war(trace_id, layer_index);"
        )

        # Contrastive steering results (Contrastive Activation Addition / RepE)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vivarium_contrastive_steering (
              steering_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              layer_index INTEGER NOT NULL,
              alpha REAL NOT NULL,
              trace_id TEXT NOT NULL,
              output_id_before TEXT NOT NULL,
              output_id_after TEXT NOT NULL,
              flipped_to_sycophantic INTEGER,
              deference_vector_path TEXT,
              created_at REAL NOT NULL,
              FOREIGN KEY(run_id) REFERENCES runs(run_id),
              FOREIGN KEY(trace_id) REFERENCES trace(trace_id)
            );
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vivarium_steering_run ON vivarium_contrastive_steering(run_id, layer_index, alpha);"
        )

        # Activation patching (causal tracing) results
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vivarium_activation_patching (
              patch_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              source_trace_id TEXT NOT NULL,
              target_trace_id TEXT NOT NULL,
              layer_index INTEGER NOT NULL,
              output_id_before_patch TEXT NOT NULL,
              output_id_after_patch TEXT NOT NULL,
              rescued_truth INTEGER,
              logit_diff_recovery REAL,
              created_at REAL NOT NULL,
              FOREIGN KEY(run_id) REFERENCES runs(run_id),
              FOREIGN KEY(source_trace_id) REFERENCES trace(trace_id),
              FOREIGN KEY(target_trace_id) REFERENCES trace(trace_id)
            );
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vivarium_patching_run ON vivarium_activation_patching(run_id, layer_index);"
        )

    def insert_agent_state(
        self,
        *,
        run_id: str,
        time_step: int,
        agent_id: str,
        state_json: str,
        created_at: float,
    ) -> None:
        """Insert or replace agent state snapshot for a given run/step/agent."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO agent_states (run_id, time_step, agent_id, state_json, created_at)
            VALUES (?, ?, ?, ?, ?);
            """,
            (run_id, time_step, agent_id, state_json, created_at),
        )

    def init_conformity_schema(self) -> None:
        """
        Initialize the Olmo conformity experiment schema (conformity_* tables).

        This is intentionally separated from init_schema() so the simulation kernel
        can run without coupling to any particular experiment.
        """
        # Shared datasets (immutable facts, social conventions, probe training sets)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conformity_datasets (
              dataset_id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              version TEXT NOT NULL,
              path TEXT NOT NULL,
              sha256 TEXT NOT NULL,
              created_at REAL NOT NULL
            );
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_conformity_datasets_name ON conformity_datasets(name, version);")

        # Individual items/questions within a dataset
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conformity_items (
              item_id TEXT PRIMARY KEY,
              dataset_id TEXT NOT NULL,
              domain TEXT NOT NULL,
              question TEXT NOT NULL,
              ground_truth_text TEXT,
              ground_truth_json TEXT,
              source_json TEXT,
              created_at REAL NOT NULL,
              FOREIGN KEY(dataset_id) REFERENCES conformity_datasets(dataset_id)
            );
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_conformity_items_dataset ON conformity_items(dataset_id, domain);")

        # Experimental conditions (control vs synthetic confederates vs authoritative bias etc.)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conformity_conditions (
              condition_id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              params_json TEXT NOT NULL,
              created_at REAL NOT NULL
            );
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_conformity_conditions_name ON conformity_conditions(name);")

        # Trial = one model variant answering one item under one condition
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conformity_trials (
              trial_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              model_id TEXT NOT NULL,
              variant TEXT NOT NULL,
              item_id TEXT NOT NULL,
              condition_id TEXT NOT NULL,
              seed INTEGER NOT NULL,
              temperature REAL NOT NULL,
              created_at REAL NOT NULL,
              FOREIGN KEY(run_id) REFERENCES runs(run_id),
              FOREIGN KEY(item_id) REFERENCES conformity_items(item_id),
              FOREIGN KEY(condition_id) REFERENCES conformity_conditions(condition_id)
            );
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conformity_trials_run ON conformity_trials(run_id, variant, model_id);"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conformity_trials_item ON conformity_trials(item_id, condition_id);"
        )

        # Prompt record (rendered components + hash)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conformity_prompts (
              prompt_id TEXT PRIMARY KEY,
              trial_id TEXT NOT NULL,
              system_prompt TEXT NOT NULL,
              user_prompt TEXT NOT NULL,
              chat_history_json TEXT NOT NULL,
              rendered_prompt_hash TEXT NOT NULL,
              created_at REAL NOT NULL,
              FOREIGN KEY(trial_id) REFERENCES conformity_trials(trial_id)
            );
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_conformity_prompts_trial ON conformity_prompts(trial_id);")

        # Prompt rendering metadata (structured; enables tracing prompt construction decisions)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conformity_prompt_metadata (
              prompt_id TEXT PRIMARY KEY,
              metadata_json TEXT NOT NULL,
              created_at REAL NOT NULL,
              FOREIGN KEY(prompt_id) REFERENCES conformity_prompts(prompt_id)
            );
            """
        )

        # Trial metadata (generation config, gateway info, etc.)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conformity_trial_metadata (
              trial_id TEXT PRIMARY KEY,
              metadata_json TEXT NOT NULL,
              created_at REAL NOT NULL,
              FOREIGN KEY(trial_id) REFERENCES conformity_trials(trial_id)
            );
            """
        )

        # Map each trial to a deterministic capture step (for activation_metadata alignment)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conformity_trial_steps (
              trial_id TEXT PRIMARY KEY,
              time_step INTEGER NOT NULL,
              agent_id TEXT NOT NULL,
              created_at REAL NOT NULL,
              FOREIGN KEY(trial_id) REFERENCES conformity_trials(trial_id)
            );
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_conformity_trial_steps_step ON conformity_trial_steps(time_step, agent_id);")

        # Output record (raw + parsed + evaluation signals)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conformity_outputs (
              output_id TEXT PRIMARY KEY,
              trial_id TEXT NOT NULL,
              raw_text TEXT NOT NULL,
              parsed_answer_text TEXT,
              parsed_answer_json TEXT,
              is_correct INTEGER,
              refusal_flag INTEGER NOT NULL,
              latency_ms REAL,
              token_usage_json TEXT,
              created_at REAL NOT NULL,
              FOREIGN KEY(trial_id) REFERENCES conformity_trials(trial_id)
            );
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_conformity_outputs_trial ON conformity_outputs(trial_id);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_conformity_outputs_correct ON conformity_outputs(is_correct);")

        # Logit lens outputs (optional, stored as JSON for compactness)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conformity_logit_lens (
              logit_id TEXT PRIMARY KEY,
              trial_id TEXT NOT NULL,
              layer_index INTEGER NOT NULL,
              token_index INTEGER NOT NULL,
              topk_json TEXT NOT NULL,
              created_at REAL NOT NULL,
              FOREIGN KEY(trial_id) REFERENCES conformity_trials(trial_id)
            );
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conformity_logit_trial ON conformity_logit_lens(trial_id, layer_index, token_index);"
        )

    def insert_run(self, meta: RunMetadata) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO runs(run_id, seed, created_at, config_json)
                VALUES (?, ?, ?, ?);
                """,
                (meta.run_id, meta.seed, meta.created_at, _json_dumps_deterministic(meta.config)),
            )

    def append_trace(self, event: TraceEvent) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO trace(
                  trace_id, run_id, time_step, agent_id, action_type,
                  info_json, outcome_json, created_at, environment_state_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    event.trace_id,
                    event.run_id,
                    event.time_step,
                    event.agent_id,
                    event.action_type,
                    _json_dumps_deterministic(event.info),
                    _json_dumps_deterministic(event.outcome),
                    event.timestamp,
                    event.environment_state_hash,
                ),
            )

    def insert_message(
        self,
        *,
        message_id: str,
        run_id: str,
        time_step: int,
        author_id: str,
        content: str,
        created_at: float,
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO messages(message_id, run_id, time_step, author_id, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (message_id, run_id, time_step, author_id, content, created_at),
            )

    def fetch_recent_messages(
        self, *, run_id: str, up_to_time_step: int, limit: int = 20
    ) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT message_id, run_id, time_step, author_id, content, created_at
            FROM messages
            WHERE run_id = ? AND time_step <= ?
            ORDER BY time_step DESC, created_at DESC
            LIMIT ?;
            """,
            (run_id, up_to_time_step, limit),
        ).fetchall()
        # Return oldest-to-newest for easier prompting
        return [dict(r) for r in reversed(rows)]

    def fetch_trace_events(
        self, *, run_id: str, from_time_step: int = 0, to_time_step: Optional[int] = None
    ) -> List[TraceEvent]:
        """
        Fetch trace events for replay. Returns events ordered by time_step, then created_at.
        """
        if to_time_step is None:
            rows = self.conn.execute(
                """
                SELECT trace_id, run_id, time_step, agent_id, action_type,
                       info_json, outcome_json, created_at, environment_state_hash
                FROM trace
                WHERE run_id = ? AND time_step >= ?
                ORDER BY time_step ASC, created_at ASC;
                """,
                (run_id, from_time_step),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT trace_id, run_id, time_step, agent_id, action_type,
                       info_json, outcome_json, created_at, environment_state_hash
                FROM trace
                WHERE run_id = ? AND time_step >= ? AND time_step <= ?
                ORDER BY time_step ASC, created_at ASC;
                """,
                (run_id, from_time_step, to_time_step),
            ).fetchall()

        events = []
        for row in rows:
            events.append(
                TraceEvent(
                    trace_id=row["trace_id"],
                    run_id=row["run_id"],
                    time_step=row["time_step"],
                    timestamp=row["created_at"],
                    agent_id=row["agent_id"],
                    action_type=row["action_type"],
                    info=json.loads(row["info_json"]),
                    outcome=json.loads(row["outcome_json"]),
                    environment_state_hash=row["environment_state_hash"],
                )
            )
        return events

    def get_run_metadata(self, *, run_id: str) -> Optional[RunMetadata]:
        """Fetch run metadata for a given run_id."""
        row = self.conn.execute(
            "SELECT run_id, seed, created_at, config_json FROM runs WHERE run_id = ?;", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return RunMetadata(
            run_id=row["run_id"],
            seed=row["seed"],
            created_at=row["created_at"],
            config=json.loads(row["config_json"]),
        )

    def insert_activation_metadata(self, record: "ActivationRecordRef") -> None:
        """Insert activation metadata record."""
        import uuid

        record_id = str(uuid.uuid4())
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO activation_metadata(
                  record_id, run_id, time_step, agent_id, model_id,
                  layer_index, component, token_position, shard_file_path,
                  tensor_key, shape_json, dtype, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    record_id,
                    record.run_id,
                    record.time_step,
                    record.agent_id,
                    record.model_id,
                    record.layer_index,
                    record.component,
                    record.token_position,
                    record.shard_file_path,
                    record.tensor_key,
                    _json_dumps_deterministic({"shape": list(record.shape)}),
                    record.dtype,
                    time.time(),
                ),
            )

    def insert_merkle_log(
        self,
        *,
        run_id: str,
        time_step: int,
        agent_id: str,
        prompt_hash: str,
        activation_hash: str,
        leaf_hash: str,
        merkle_root: str,
        created_at: Optional[float] = None,
    ) -> None:
        """Insert a Merkle log record for provenance verification."""
        import uuid

        ts = float(time.time() if created_at is None else created_at)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO merkle_log(
                  merkle_id, run_id, time_step, agent_id,
                  prompt_hash, activation_hash, leaf_hash, merkle_root, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    str(uuid.uuid4()),
                    str(run_id),
                    int(time_step),
                    str(agent_id),
                    str(prompt_hash),
                    str(activation_hash),
                    str(leaf_hash),
                    str(merkle_root),
                    ts,
                ),
            )

    def fetch_activation_metadata(
        self, *, run_id: str, time_step: Optional[int] = None, agent_id: Optional[str] = None
    ) -> List["ActivationRecordRef"]:
        """Fetch activation metadata records."""
        query = "SELECT * FROM activation_metadata WHERE run_id = ?"
        params: List[Any] = [run_id]

        if time_step is not None:
            query += " AND time_step = ?"
            params.append(time_step)
        if agent_id is not None:
            query += " AND agent_id = ?"
            params.append(agent_id)

        query += " ORDER BY time_step ASC, layer_index ASC;"

        rows = self.conn.execute(query, tuple(params)).fetchall()
        from vivarium.interpretability import ActivationRecordRef

        records = []
        for row in rows:
            shape_data = json.loads(row["shape_json"])
            records.append(
                ActivationRecordRef(
                    run_id=row["run_id"],
                    time_step=row["time_step"],
                    agent_id=row["agent_id"],
                    model_id=row["model_id"],
                    layer_index=row["layer_index"],
                    component=row["component"],
                    token_position=row["token_position"],
                    shard_file_path=row["shard_file_path"],
                    tensor_key=row["tensor_key"],
                    shape=tuple(shape_data["shape"]),
                    dtype=row["dtype"],
                )
            )
        return records

    # -----------------------------
    # Conformity experiment helpers
    # -----------------------------
    def upsert_conformity_dataset(
        self, *, dataset_id: str, name: str, version: str, path: str, sha256: str, created_at: Optional[float] = None
    ) -> None:
        ts = float(time.time() if created_at is None else created_at)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO conformity_datasets(dataset_id, name, version, path, sha256, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_id) DO UPDATE SET
                  name=excluded.name,
                  version=excluded.version,
                  path=excluded.path,
                  sha256=excluded.sha256;
                """,
                (dataset_id, name, version, path, sha256, ts),
            )

    def upsert_conformity_condition(
        self, *, condition_id: str, name: str, params: Dict[str, Any], created_at: Optional[float] = None
    ) -> None:
        ts = float(time.time() if created_at is None else created_at)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO conformity_conditions(condition_id, name, params_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(condition_id) DO UPDATE SET
                  name=excluded.name,
                  params_json=excluded.params_json;
                """,
                (condition_id, name, _json_dumps_any(params), ts),
            )

    def insert_conformity_item(
        self,
        *,
        item_id: str,
        dataset_id: str,
        domain: str,
        question: str,
        ground_truth_text: Optional[str],
        ground_truth_json: Optional[Dict[str, Any]],
        source_json: Optional[Dict[str, Any]],
        created_at: Optional[float] = None,
    ) -> None:
        ts = float(time.time() if created_at is None else created_at)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO conformity_items(
                  item_id, dataset_id, domain, question,
                  ground_truth_text, ground_truth_json, source_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    item_id,
                    dataset_id,
                    domain,
                    question,
                    ground_truth_text,
                    (_json_dumps_any(ground_truth_json) if ground_truth_json is not None else None),
                    (_json_dumps_any(source_json) if source_json is not None else None),
                    ts,
                ),
            )

    def insert_conformity_trial(
        self,
        *,
        trial_id: str,
        run_id: str,
        model_id: str,
        variant: str,
        item_id: str,
        condition_id: str,
        seed: int,
        temperature: float,
        created_at: Optional[float] = None,
    ) -> None:
        ts = float(time.time() if created_at is None else created_at)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO conformity_trials(
                  trial_id, run_id, model_id, variant, item_id, condition_id, seed, temperature, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (trial_id, run_id, model_id, variant, item_id, condition_id, int(seed), float(temperature), ts),
            )

    def insert_conformity_prompt(
        self,
        *,
        prompt_id: str,
        trial_id: str,
        system_prompt: str,
        user_prompt: str,
        chat_history: List[Dict[str, Any]],
        rendered_prompt_hash: str,
        created_at: Optional[float] = None,
    ) -> None:
        ts = float(time.time() if created_at is None else created_at)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO conformity_prompts(
                  prompt_id, trial_id, system_prompt, user_prompt, chat_history_json, rendered_prompt_hash, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (prompt_id, trial_id, system_prompt, user_prompt, _json_dumps_any(chat_history), rendered_prompt_hash, ts),
            )

    def upsert_conformity_prompt_metadata(
        self,
        *,
        prompt_id: str,
        metadata: Dict[str, Any],
        created_at: Optional[float] = None,
    ) -> None:
        ts = float(time.time() if created_at is None else created_at)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO conformity_prompt_metadata(prompt_id, metadata_json, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(prompt_id) DO UPDATE SET
                  metadata_json=excluded.metadata_json;
                """,
                (prompt_id, _json_dumps_any(metadata), ts),
            )

    def upsert_conformity_trial_metadata(
        self,
        *,
        trial_id: str,
        metadata: Dict[str, Any],
        created_at: Optional[float] = None,
    ) -> None:
        ts = float(time.time() if created_at is None else created_at)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO conformity_trial_metadata(trial_id, metadata_json, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(trial_id) DO UPDATE SET
                  metadata_json=excluded.metadata_json;
                """,
                (trial_id, _json_dumps_any(metadata), ts),
            )

    def upsert_conformity_trial_step(
        self, *, trial_id: str, time_step: int, agent_id: str, created_at: Optional[float] = None
    ) -> None:
        ts = float(time.time() if created_at is None else created_at)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO conformity_trial_steps(trial_id, time_step, agent_id, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(trial_id) DO UPDATE SET
                  time_step=excluded.time_step,
                  agent_id=excluded.agent_id;
                """,
                (trial_id, int(time_step), str(agent_id), ts),
            )

    def insert_conformity_output(
        self,
        *,
        output_id: str,
        trial_id: str,
        raw_text: str,
        parsed_answer_text: Optional[str],
        parsed_answer_json: Optional[Dict[str, Any]],
        is_correct: Optional[bool],
        refusal_flag: bool,
        latency_ms: Optional[float],
        token_usage_json: Optional[Dict[str, Any]],
        created_at: Optional[float] = None,
    ) -> None:
        ts = float(time.time() if created_at is None else created_at)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO conformity_outputs(
                  output_id, trial_id, raw_text, parsed_answer_text, parsed_answer_json,
                  is_correct, refusal_flag, latency_ms, token_usage_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    output_id,
                    trial_id,
                    raw_text,
                    parsed_answer_text,
                    (_json_dumps_any(parsed_answer_json) if parsed_answer_json is not None else None),
                    (None if is_correct is None else (1 if bool(is_correct) else 0)),
                    (1 if bool(refusal_flag) else 0),
                    latency_ms,
                    (_json_dumps_any(token_usage_json) if token_usage_json is not None else None),
                    ts,
                ),
            )

    def batch_write_conformity_trial_results(
        self,
        output_rows: "List[ConformityOutputRow]",
    ) -> None:
        """Write a batch of conformity outputs in a single transaction.

        Reduces per-trial transaction churn from O(N) commits to O(N/batch_size)
        by grouping multiple outputs into one executemany call. Safe to call with
        an empty list (no-op). Compatible with --resume-auto: trial/prompt rows
        are written individually in the build phase; only outputs are batched here.
        """
        if not output_rows:
            return
        ts = float(time.time())
        with self.conn:
            self.conn.executemany(
                """
                INSERT INTO conformity_outputs(
                  output_id, trial_id, raw_text, parsed_answer_text, parsed_answer_json,
                  is_correct, refusal_flag, latency_ms, token_usage_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                [
                    (
                        row.output_id,
                        row.trial_id,
                        row.raw_text,
                        row.parsed_answer_text,
                        (_json_dumps_any(row.parsed_answer_json) if row.parsed_answer_json is not None else None),
                        (None if row.is_correct is None else (1 if bool(row.is_correct) else 0)),
                        (1 if bool(row.refusal_flag) else 0),
                        row.latency_ms,
                        (_json_dumps_any(row.token_usage_json) if row.token_usage_json is not None else None),
                        float(row.created_at or ts),
                    )
                    for row in output_rows
                ],
            )

    # -----------------------------------------
    # Generic interpretability helpers
    # -----------------------------------------
    def upsert_answer_logprob(
        self,
        *,
        trace_id: str,
        context_kind: str,
        candidate_kind: str,
        candidate_text: str,
        token_count: int,
        logprob_sum: float,
        logprob_mean: float,
        first_token_id: Optional[int],
        first_token_logprob: Optional[float],
        metadata: Dict[str, Any],
        created_at: Optional[float] = None,
    ) -> None:
        ts = float(time.time() if created_at is None else created_at)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO vivarium_answer_logprobs(
                  trace_id, context_kind, candidate_kind, candidate_text,
                  token_count, logprob_sum, logprob_mean,
                  first_token_id, first_token_logprob,
                  metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trace_id, context_kind, candidate_kind) DO UPDATE SET
                  candidate_text=excluded.candidate_text,
                  token_count=excluded.token_count,
                  logprob_sum=excluded.logprob_sum,
                  logprob_mean=excluded.logprob_mean,
                  first_token_id=excluded.first_token_id,
                  first_token_logprob=excluded.first_token_logprob,
                  metadata_json=excluded.metadata_json;
                """,
                (
                    str(trace_id),
                    str(context_kind),
                    str(candidate_kind),
                    str(candidate_text),
                    int(token_count),
                    float(logprob_sum),
                    float(logprob_mean),
                    (None if first_token_id is None else int(first_token_id)),
                    (None if first_token_logprob is None else float(first_token_logprob)),
                    _json_dumps_any(metadata or {}),
                    ts,
                ),
            )

    def insert_probe(
        self,
        *,
        probe_id: str,
        run_id: str,
        probe_kind: str,
        train_dataset_id: str,
        model_id: str,
        layers: List[int],
        component: str,
        token_position: int,
        artifact_path: str,
        metrics: Dict[str, Any],
        created_at: Optional[float] = None,
    ) -> None:
        ts = float(time.time() if created_at is None else created_at)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO vivarium_probes(
                  probe_id, run_id, probe_kind, train_dataset_id, model_id,
                  layers_json, component, token_position, artifact_path, metrics_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    probe_id,
                    run_id,
                    probe_kind,
                    train_dataset_id,
                    model_id,
                    _json_dumps_any({"layers": list(layers)}),
                    component,
                    int(token_position),
                    artifact_path,
                    _json_dumps_any(metrics),
                    ts,
                ),
            )

    def insert_projection_rows(
        self, *, rows: List[Tuple[str, str, str, int, Optional[int], float]], created_at: Optional[float] = None
    ) -> None:
        """
        Bulk insert projections into vivarium_probe_projections.
        rows: [(projection_id, trace_id, probe_id, layer_index, token_index, value_float), ...]
        """
        ts = float(time.time() if created_at is None else created_at)
        with self.conn:
            self.conn.executemany(
                """
                INSERT INTO vivarium_probe_projections(
                  projection_id, trace_id, probe_id, layer_index, token_index, value_float, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                [(pid, tid, prid, int(layer), tok, float(val), ts) for (pid, tid, prid, layer, tok, val) in rows],
            )

    def insert_intervention(
        self,
        *,
        intervention_id: str,
        run_id: str,
        name: str,
        alpha: float,
        target_layers: List[int],
        component: str,
        vector_probe_id: str,
        notes: Optional[str],
        created_at: Optional[float] = None,
    ) -> None:
        ts = float(time.time() if created_at is None else created_at)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO vivarium_interventions(
                  intervention_id, run_id, name, alpha, target_layers_json, component, vector_probe_id, notes, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    intervention_id,
                    run_id,
                    name,
                    float(alpha),
                    _json_dumps_any({"layers": list(target_layers)}),
                    component,
                    vector_probe_id,
                    notes,
                    ts,
                ),
            )

    def insert_intervention_result(
        self,
        *,
        result_id: str,
        trace_id: str,
        intervention_id: str,
        output_id_before: str,
        output_id_after: str,
        flipped_to_truth: Optional[bool],
        created_at: Optional[float] = None,
    ) -> None:
        ts = float(time.time() if created_at is None else created_at)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO vivarium_intervention_results(
                  result_id, trace_id, intervention_id, output_id_before, output_id_after, flipped_to_truth, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    result_id,
                    trace_id,
                    intervention_id,
                    output_id_before,
                    output_id_after,
                    (None if flipped_to_truth is None else (1 if bool(flipped_to_truth) else 0)),
                    ts,
                ),
            )

    def close(self) -> None:
        if self._conn is None:
            return
        self._conn.close()
        self._conn = None
