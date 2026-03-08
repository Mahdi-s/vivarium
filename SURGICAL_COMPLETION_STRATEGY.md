# Surgical Completion Strategy: Targeted Trial Completion

## Current State

Database audit shows:
- **T=0.0, 0.2, 0.8, 1.0**: 9 gaps each (only `rl_zero` × Zhu-style conditions missing)
- **T=0.4, 0.6**: Complete (no gaps)
- **Total gaps**: 36 cells (out of ~1,200 possible variant × condition combinations)
- **Why**: rl_zero is intentionally excluded from analysis, but the gaps still exist in the DB schema

## Strategic Options

### Option 1: MINIMAL - Accept Gaps (Recommended for Your Use Case)

**Status**: ✓ No action needed

Since rl_zero is excluded from your publication analysis (`EXCLUDED_VARIANTS = ("rl_zero",)` in existing scripts), the gaps are irrelevant. Your publication pipeline already handles this.

- **Pros**: Zero computation, analysis is already complete
- **Cons**: Database schema has gaps (cosmetic issue only)
- **Action**: Continue with current analysis using `generate_publication_item_set.py`

---

### Option 2: SURGICAL COMPLETION - Fill Only the 36 Gaps (Medium Effort)

If you want a **complete database for archival/reproducibility**, you can surgically fill just the rl_zero gaps.

#### Infrastructure Needed

**A. Create a Minimal Suite Config per Temperature**

This config specifies ONLY rl_zero × Zhu-style conditions:

```json
{
  "paths_config": "paths.json",
  "suite_name": "olmo_conformity_surgical_rl_zero_gaps",
  "suite_version": "v6_gaps",
  "description": "Surgical completion: only rl_zero variant × Zhu-style conditions (9 conditions) across all datasets",
  "datasets": [
    { "name": "immutable_facts_minimal", "version": "v2", "path": "..." },
    { "name": "social_conventions_minimal", "version": "v2", "path": "..." },
    { "name": "gsm8k", "version": "v1", "path": "..." },
    { "name": "mmlu_math", "version": "v1", "path": "..." },
    { "name": "mmlu_science", "version": "v1", "path": "..." },
    { "name": "mmlu_knowledge", "version": "v1", "path": "..." },
    { "name": "truthfulqa", "version": "v1", "path": "..." },
    { "name": "arc", "version": "v1", "path": "..." }
  ],
  "conditions": [
    { "name": "asch_zhu_unbiased_unanimous_plain", "params": {...} },
    { "name": "asch_zhu_unbiased_unanimous_neutral", "params": {...} },
    { "name": "asch_zhu_unbiased_unanimous_confident", "params": {...} },
    { "name": "asch_zhu_unbiased_unanimous_uncertain", "params": {...} },
    { "name": "asch_zhu_unbiased_diverse_plain", "params": {...} },
    { "name": "asch_zhu_unbiased_da", "params": {...} },
    { "name": "asch_zhu_unbiased_qd", "params": {...} },
    { "name": "authority_zhu_unbiased_trust", "params": {...} },
    { "name": "authority_zhu_unbiased_trust_da", "params": {...} }
  ],
  "models": [
    {
      "variant": "rl_zero",
      "model_id": "allenai/OLMo-3-7B",
      "temperatures": [0.0, 0.2, 0.8, 1.0]
    }
  ]
}
```

#### Execution Commands

**Per-Temperature:**
```bash
# Create each minimal suite config (names: suite_surgical_gap_fill_t0.0.json, etc.)
# Then run:

python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs \
  --suite experiments/olmo_conformity/configs/suite_surgical_gap_fill_t0.0.json \
  --phase trials \
  --runs-only  # Just trials, skip analysis

python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs \
  --suite experiments/olmo_conformity/configs/suite_surgical_gap_fill_t0.2.json \
  --phase trials \
  --runs-only

python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs \
  --suite experiments/olmo_conformity/configs/suite_surgical_gap_fill_t0.8.json \
  --phase trials \
  --runs-only

python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs \
  --suite experiments/olmo_conformity/configs/suite_surgical_gap_fill_t1.0.json \
  --phase trials \
  --runs-only
```

#### Why This Works

1. **Vivarium's incremental behavior**: When you run against an existing `simulation.db`, it:
   - Reads the current schema (conditions, items, etc.)
   - Inserts only NEW (variant, condition, item) tuples
   - Does NOT duplicate existing trials (uses compound keys)

2. **Verification**: After each run, query to confirm:
```sql
SELECT variant, condition_id, COUNT(*) as n_items
FROM conformity_trials
WHERE variant = 'rl_zero'
GROUP BY variant, condition_id
ORDER BY condition_id;
```

3. **Cost**: ~36 trials total (1 variant × 9 conditions × 4 items/dataset × 8 datasets ≈ negligible compute)

#### Database Safety

Vivarium uses atomic inserts with conflict detection:
- If a trial already exists (same run_id, item_id, condition_id), it's skipped
- No updates to existing output records
- Safe to re-run without data corruption

---

### Option 3: FULL COMPLETION - Re-run All Temperatures (Overkill)

Run complete fresh suite for all temperatures, then merge:

```bash
python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs/complete_v2 \
  --phase trials
```

Then merge the new data into existing databases. This is expensive but gives you a completely fresh run.

- **Pros**: Fresh data, no dependency on old runs
- **Cons**: ~200K trials to recompute, wastes RL0 resources
- **Not recommended** unless you suspect data corruption

---

## Recommended Path Forward

### **Option 1 (MINIMAL) - Suggested**

- ✓ Do nothing
- ✓ Your analysis already excludes rl_zero
- ✓ Publication datasets are complete
- ✓ Cost: $0, Time: 0 minutes

### **Option 2 (SURGICAL) - If You Want Full Schema Completeness**

- Create 4 minimal suite configs (one per temperature)
- Run each with `--phase trials --runs-only`
- Takes ~10 minutes total compute
- Databases will be schema-complete
- Good for archival/reproducibility

---

## How to Implement Option 2

### Step 1: Generate Minimal Suite Configs

```bash
python << 'EOF'
import json
from pathlib import Path

BASE_SUITE = "experiments/olmo_conformity/configs/suite_expanded_temp0.0.json"

with open(BASE_SUITE) as f:
    base = json.load(f)

# Only keep rl_zero variant
rl_zero_model = {
    "variant": "rl_zero",
    "model_id": "allenai/OLMo-3-7B",
    "temperatures": [0.0, 0.2, 0.8, 1.0]
}

# Only keep Zhu-style conditions (skip core behavioral)
zhu_conditions = [
    "asch_zhu_unbiased_unanimous_plain",
    "asch_zhu_unbiased_unanimous_neutral",
    "asch_zhu_unbiased_unanimous_confident",
    "asch_zhu_unbiased_unanimous_uncertain",
    "asch_zhu_unbiased_diverse_plain",
    "asch_zhu_unbiased_da",
    "asch_zhu_unbiased_qd",
    "authority_zhu_unbiased_trust",
    "authority_zhu_unbiased_trust_da",
]

base["suite_name"] = "olmo_conformity_surgical_rl_zero_gaps"
base["suite_version"] = "v6_gaps"
base["description"] = "Surgical completion: rl_zero × Zhu conditions only"
base["models"] = [rl_zero_model]
base["conditions"] = [
    c for c in base.get("conditions", [])
    if c.get("name") in zhu_conditions
]

# Save per-temperature
for temp in [0.0, 0.2, 0.8, 1.0]:
    temp_str = str(temp).replace(".", "p")
    config = base.copy()
    config["models"] = [{
        "variant": "rl_zero",
        "model_id": "allenai/OLMo-3-7B",
        "temperatures": [temp]
    }]

    with open(f"experiments/olmo_conformity/configs/suite_surgical_gap_fill_t{temp_str}.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Created suite_surgical_gap_fill_t{temp_str}.json")
EOF
```

### Step 2: Run Surgical Fills (In Parallel on HPC)

```bash
# Terminal 1
python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs \
  --suite experiments/olmo_conformity/configs/suite_surgical_gap_fill_t0p0.json \
  --phase trials --runs-only

# Terminal 2
python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs \
  --suite experiments/olmo_conformity/configs/suite_surgical_gap_fill_t0p2.json \
  --phase trials --runs-only

# Terminal 3
python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs \
  --suite experiments/olmo_conformity/configs/suite_surgical_gap_fill_t0p8.json \
  --phase trials --runs-only

# Terminal 4
python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs \
  --suite experiments/olmo_conformity/configs/suite_surgical_gap_fill_t1p0.json \
  --phase trials --runs-only
```

### Step 3: Verify Completeness

```bash
python3 << 'EOF'
import sqlite3
from pathlib import Path
import json

with open("Comparing_Experiments/runs_metadata_v6.json") as f:
    metadata = json.load(f)

print("POST-COMPLETION VERIFICATION\n")

for temp_key, temp_entry in sorted(metadata["experiments"].items()):
    temp = temp_entry["temperature"]
    run_dir = Path("runs_latest/runs") / temp_entry["run_dir"]
    db_path = run_dir / "simulation.db"

    db = sqlite3.connect(db_path)
    cursor = db.execute("""
        SELECT COUNT(DISTINCT (c.name, t.variant))
        FROM conformity_conditions c
        CROSS JOIN conformity_trials t
        WHERE EXISTS (
            SELECT 1 FROM conformity_trials t2
            WHERE t2.condition_id = c.condition_id AND t2.variant = t.variant
        )
    """)
    complete_cells = cursor.fetchone()[0]
    db.close()

    print(f"T={temp}: {complete_cells:3d}/84 cells populated")
EOF
```

---

## Database Schema Notes

The conformity_trials table has these key columns:
```sql
trial_id (PK)
run_id    (FK to runs)
model_id  (e.g., "allenai/OLMo-3-7B")
variant   (e.g., "rl_zero", "think_dpo")
item_id   (FK to conformity_items)
condition_id  (FK to conformity_conditions)
temperature
```

Vivarium enforces uniqueness via:
- (run_id, item_id, condition_id, variant, temperature) must be unique
- If you try to insert a duplicate, it's silently skipped (or logged)

This means surgical fills are **completely safe** — old trials are never overwritten.

---

## Cost-Benefit Analysis

| Option | Computation | Time | Benefit | Risk |
|--------|-----------|------|---------|------|
| **1 (Minimal)** | 0 | 0 min | Status quo, keep analysis as-is | None |
| **2 (Surgical)** | ~36 trials | ~10 min | Complete DB schema | None (atomic appends) |
| **3 (Full)** | ~200K trials | ~8 hours | Fresh run, full reproducibility | Low (compute waste) |

**Recommendation**: Use **Option 1**. Your analysis is complete. The rl_zero gaps are intentional and irrelevant to publication.

If auditors ask for schema completeness, use **Option 2** (~10 minutes).
