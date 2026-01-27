# Temperature Configuration Verification Report

## Summary

This report verifies that temperature values from JSON suite configs are correctly propagated through the entire experiment pipeline and applied to model sampling.

## Issues Found

### ✅ **CORRECT: Behavioral Trials (runner.py)**
- **Location**: `src/aam/experiments/olmo_conformity/runner.py:317`
- **Status**: ✅ Working correctly
- **Code**: `temperature = float(cfg.get("run", {}).get("temperature", 0.0))`
- **Usage**: Passed to `gateway.chat(..., temperature=temperature)` at line 551
- **Verification**: Temperature is correctly extracted from suite config and applied to all behavioral trials

### ✅ **CORRECT: Probe Training (orchestration.py)**
- **Location**: `src/aam/experiments/olmo_conformity/orchestration.py:179`
- **Status**: ✅ Working correctly
- **Code**: `temperature=config.temperature` passed to `run_truth_social_vector_analysis()`
- **Verification**: Temperature flows from ExperimentConfig to probe training

### ✅ **FIXED: Temperature Extraction in run.py**
- **Location**: `src/aam/run.py:1157`
- **Status**: ✅ **FIXED**
- **Previous Issue**: Was using `load_experiment_config()` which expects a different config structure
- **Fix Applied**: Now uses `load_suite_config()` from `io.py`:
  ```python
  from aam.experiments.olmo_conformity.io import load_suite_config
  suite_cfg = load_suite_config(str(args.suite_config))
  temperature = float(suite_cfg.get("run", {}).get("temperature", 0.0))
  ```
- **Verification**: Tested with `suite_complete_temp1.json` - correctly extracts temperature=1.0

### ✅ **FIXED: Intervention Hardcoded Temperature**
- **Location**: `src/aam/experiments/olmo_conformity/intervention.py:261, 309`
- **Status**: ✅ **FIXED**
- **Previous Issue**: Interventions always used temperature 0.0, regardless of suite config
- **Fixes Applied**: 
  1. ✅ Added `temperature: float = 0.0` parameter to `run_intervention_sweep()` function signature
  2. ✅ Updated orchestration.py to pass `temperature=config.temperature` to both intervention calls
  3. ✅ Updated both `resp_before` and `resp_after` calls to use `temperature=temperature` parameter
- **Verification**: Temperature now flows from config → orchestration → intervention → gateway

### ✅ **CORRECT: LLM Gateway Implementation**
- **Location**: `src/aam/llm_gateway.py:1080-1087`
- **Status**: ✅ Working correctly
- **Code**: 
  ```python
  do_sample = float(temperature) > 0.0
  if do_sample:
      gen_kwargs["temperature"] = float(temperature)
  ```
- **Verification**: HuggingFace gateway correctly uses temperature parameter:
  - Sets `do_sample=True` when temperature > 0.0
  - Passes temperature to `model.generate()` when sampling
  - Uses greedy decoding (no temperature) when temperature = 0.0

## JSON Config Structure Verification

### ✅ **All Suite Configs Have Correct Structure**

Verified all three temperature configs:
- `suite_complete_temp0.json`: `"temperature": 0.0` ✅
- `suite_complete_temp0.5.json`: `"temperature": 0.5` ✅
- `suite_complete_temp1.json`: `"temperature": 1.0` ✅

All configs have temperature nested under `"run": {"temperature": <value>}` as expected.

## Flow Verification

### Behavioral Trials Flow (✅ Working)
```
suite_complete_temp*.json
  → runner.py:317 (extracts temperature)
  → runner.py:513 (stores in trial record)
  → runner.py:551 (passes to gateway.chat())
  → llm_gateway.py:1080-1087 (applies to model.generate())
```

### Probe Training Flow (✅ Working)
```
suite_complete_temp*.json
  → run.py:1157 (extracts - BUT USING WRONG FUNCTION)
  → orchestration.py:68 (ExperimentConfig.temperature)
  → orchestration.py:179 (passes to vector_analysis)
  → vector_analysis.py:42 (function parameter)
  → probes.py:97 (function parameter)
  → probes.py:213 (passes to gateway.chat())
```

### Intervention Flow (✅ Fixed)
```
suite_complete_temp*.json
  → run.py:1157 (extracts using load_suite_config - FIXED)
  → orchestration.py:68 (ExperimentConfig.temperature)
  → orchestration.py:220,269 (passes temperature to run_intervention_sweep - FIXED)
  → intervention.py:261,309 (uses temperature parameter - FIXED)
  → llm_gateway.py:1080-1087 (applies to model.generate())
```

## Fixes Applied

### ✅ **Fixed: Temperature Extraction in run.py**
- **File**: `src/aam/run.py`
- **Change**: Replaced `load_experiment_config()` with `load_suite_config()` from `io.py`
- **Impact**: Temperature now correctly extracted from suite configs for probe training

### ✅ **Fixed: Intervention Temperature**
- **File**: `src/aam/experiments/olmo_conformity/intervention.py`
- **Changes**: 
  - Added `temperature: float = 0.0` parameter to `run_intervention_sweep()`
  - Updated both `resp_before` and `resp_after` to use `temperature=temperature`
- **File**: `src/aam/experiments/olmo_conformity/orchestration.py`
- **Changes**: 
  - Updated both intervention calls to pass `temperature=config.temperature`
- **Impact**: Interventions now use the same temperature as behavioral trials

### Verification Steps

1. ✅ Tested temperature extraction with `suite_complete_temp1.json` - correctly extracts 1.0
2. Run a full test with `suite_complete_temp1.json`
3. Check database: `SELECT DISTINCT temperature FROM conformity_trials WHERE run_id = ?`
4. Verify all trials have temperature = 1.0
5. Check intervention results use same temperature
6. Verify probe training uses same temperature

## Additional Notes

### ✅ **FIXED: Resume.py Temperature**
- **Location**: `src/aam/experiments/olmo_conformity/resume.py:122`
- **Status**: ✅ **FIXED**
- **Previous Issue**: Hardcoded `temperature=0.0` when repairing/replaying trial activations
- **Fix Applied**: 
  - Updated SQL query to include `t.temperature` from `conformity_trials` table
  - Extract temperature: `temperature = float(tr["temperature"])`
  - Use original temperature: `gateway.chat(..., temperature=temperature)`
- **Rationale**: When repairing activations for existing trials, we should use the same temperature that was originally used, not hardcode to 0.0. This ensures activation captures match the original trial conditions.
- **Impact**: Resume/repair operations now correctly preserve the original experiment temperature

### ✅ **Intentional: JudgeEval Temperature**
- **Location**: `src/aam/experiments/olmo_conformity/judgeval_scorers.py`
- **Status**: ✅ **Intentional - No fix needed**
- **Rationale**: Judge models (LLMs that evaluate other LLM outputs) should use deterministic temperature=0.0 for consistent evaluation. This is correct behavior.
