# Infrastructure for Surgical Database Completion

## TL;DR

**You have all the infrastructure you need.** Here's what's been created:

### 1. **Audit Tool** (Newly Created)
```bash
python Analysis\ Scripts/audit_and_generate_gap_configs.py --both
```
- Automatically audits all databases
- Identifies exact missing cells
- **Generates** minimal suite configs to fill only the gaps
- Produces commands you can copy/paste to run fills

### 2. **Completion Tracker** (Newly Created)
```json
Comparing_Experiments/completion_tracker.json
```
- Documents current completion status
- Lists exact gaps (36 rl_zero cells)
- Confirms publication readiness
- Provides recovery commands

### 3. **Strategy Document** (Newly Created)
```markdown
SURGICAL_COMPLETION_STRATEGY.md
```
- Explains 3 options (minimal, surgical, full)
- Cost-benefit analysis
- Step-by-step implementation guide

---

## Current State

| Temperature | Status | Gaps | Reason |
|---|---|---|---|
| T=0.0 | 75/84 complete | rl_zero × 9 Zhu conditions | Intentionally excluded |
| T=0.2 | 75/84 complete | rl_zero × 9 Zhu conditions | Intentionally excluded |
| T=0.4 | 84/84 complete | NONE | ✓ |
| T=0.6 | 84/84 complete | NONE | ✓ |
| T=0.8 | 75/84 complete | rl_zero × 9 Zhu conditions | Intentionally excluded |
| T=1.0 | 75/84 complete | rl_zero × 9 Zhu conditions | Intentionally excluded |

**Publication readiness**: ✓ READY (all non-excluded variants complete)

---

## What You Can Do Now

### Option A: Verify Current State (0 minutes)

```bash
# Audit databases
python Analysis\ Scripts/audit_and_generate_gap_configs.py --audit

# View completion tracker
cat Comparing_Experiments/completion_tracker.json | jq .summary
```

### Option B: Generate Gap-Fill Configs (2 minutes)

```bash
# Automatically creates suite_surgical_gap_fill_t*.json files
python Analysis\ Scripts/audit_and_generate_gap_configs.py --generate-gaps
```

### Option C: Run Surgical Fills (10 minutes total)

After running Option B:

```bash
# Fill T=0.0
python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs \
  --suite experiments/olmo_conformity/configs/suite_surgical_gap_fill_t0p0.json \
  --phase trials --runs-only

# Fill T=0.2 (parallel in another terminal)
python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs \
  --suite experiments/olmo_conformity/configs/suite_surgical_gap_fill_t0p2.json \
  --phase trials --runs-only

# Fill T=0.8 (parallel)
python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs \
  --suite experiments/olmo_conformity/configs/suite_surgical_gap_fill_t0p8.json \
  --phase trials --runs-only

# Fill T=1.0 (parallel)
python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs \
  --suite experiments/olmo_conformity/configs/suite_surgical_gap_fill_t1p0.json \
  --phase trials --runs-only
```

Then verify:
```bash
python Analysis\ Scripts/audit_and_generate_gap_configs.py --audit
```

---

## How It Works

### Database Append Logic

Vivarium has built-in incremental run support:

1. **Detects existing schema** – Reads conditions, items, variants from existing `simulation.db`
2. **Appends only new tuples** – Inserts trials where (run_id, item_id, condition_id, variant) is new
3. **Skips duplicates** – If a trial already exists, it's silently ignored
4. **Safe atomicity** – SQLite transactions ensure no corruption

This means:
- ✓ Filling gaps won't duplicate existing data
- ✓ Old trials are never overwritten
- ✓ Can re-run fills without harm

### Metadata Awareness

The completion tracker tracks:
- Which cells are complete (variant × condition × temperature)
- Which are gaps (and why)
- Judge label status per temperature
- Publication readiness

Enables decisions like:
- "Should I fill rl_zero? (Answer: No, it's excluded)"
- "Is my data publication-ready? (Answer: Yes)"
- "What's my estimated cost to fill everything? (Answer: ~72 trials)"

---

## Files Created

| File | Purpose |
|------|---------|
| `SURGICAL_COMPLETION_STRATEGY.md` | Strategic guide (3 options, cost-benefit, implementation steps) |
| `INFRASTRUCTURE_FOR_SURGICAL_COMPLETION.md` | This file — infrastructure overview |
| `Analysis Scripts/audit_and_generate_gap_configs.py` | Audit tool + config generator |
| `Comparing_Experiments/completion_tracker.json` | Metadata tracking for completion status |

---

## Future Use Cases

### 1. Adding a New Temperature (e.g., T=0.3)

```bash
# 1. Run full suite at T=0.3
python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs \
  --temps 0.3 \
  --phase all

# 2. Audit to confirm completeness
python Analysis\ Scripts/audit_and_generate_gap_configs.py --audit

# 3. Optionally fill any remaining gaps
python Analysis\ Scripts/audit_and_generate_gap_configs.py --generate-gaps
```

### 2. Adding a New Variant (e.g., future_model_v2)

```bash
# 1. Create suite config specifying only the new variant
# 2. Run it against existing databases
python experiments/olmo_conformity/configs/run_expanded_experiments.py \
  --runs-dir ./runs_latest/runs \
  --suite experiments/olmo_conformity/configs/suite_new_variant.json \
  --phase all

# 3. Audit shows new variant added to all temperatures
python Analysis\ Scripts/audit_and_generate_gap_configs.py --audit
```

### 3. Recovery from Partial Run

If a run is interrupted mid-way:

```bash
# 1. Audit to see what's missing
python Analysis\ Scripts/audit_and_generate_gap_configs.py --audit

# 2. Generate configs for incomplete cells only
python Analysis\ Scripts/audit_and_generate_gap_configs.py --generate-gaps

# 3. Resume with surgical fills
# (Re-runs only generate missing (variant, condition) combos)
```

---

## Design Principles

This infrastructure embodies:

1. **Surgical Completion** – Fill only what's missing, waste no computation
2. **Metadata-Aware** – Track what's complete, what's needed, what's excluded
3. **Atomic Safety** – Database appends don't corrupt existing data
4. **Auditability** – Complete record of what's in each database
5. **Reproducibility** – Generated configs are version-controlled and auditable
6. **Incrementality** – Can add data without full re-runs

---

## Recommendation

### For Your Current Analysis

**Do nothing.** Your publication analysis is complete:
- ✓ All required variants are complete
- ✓ All conditions are covered
- ✓ All temperatures have data
- ✓ Figures are publication-quality
- ✓ Database gaps are only excluded models (rl_zero)

### If Reviewers Ask for "Complete Database"

Run the surgical fill:
1. `python Analysis Scripts/audit_and_generate_gap_configs.py --both` (2 min)
2. Run the 4 generated commands in parallel (10 min total)
3. Commit the completion_tracker.json update

Total time: 12 minutes, trivial compute cost.

---

## API Reference

### audit_and_generate_gap_configs.py

```bash
# Audit only
python Analysis\ Scripts/audit_and_generate_gap_configs.py --audit

# Generate gap-fill configs only (requires existing audit)
python Analysis\ Scripts/audit_and_generate_gap_configs.py --generate-gaps

# Both (recommended)
python Analysis\ Scripts/audit_and_generate_gap_configs.py --both

# Custom directories
python Analysis\ Scripts/audit_and_generate_gap_configs.py \
  --audit \
  --runs-dir /custom/runs \
  --metadata /custom/metadata.json \
  --output-dir /custom/configs \
  --base-suite /custom/suite.json
```

**Output**: Formatted audit report + generated suite config files

---

## Questions?

Check:
1. **How do I know what's missing?** → `audit_and_generate_gap_configs.py --audit`
2. **How do I fill specific gaps?** → `audit_and_generate_gap_configs.py --generate-gaps`
3. **Is my publication data complete?** → `completion_tracker.json` (status: READY)
4. **What's the cost to be perfect?** → ~72 trials (10 min, negligible)
5. **Why are some gaps there?** → rl_zero excluded by design
