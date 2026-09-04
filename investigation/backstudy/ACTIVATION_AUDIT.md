# Audit: activation capture, probing, steering and the rest of the chain (2026-09-04)

Scope: `tools/belief_probe.py`, `probe_analysis.py`, `steer_analysis.py`, `analysis_cross_checkpoint.py`, the HPC job, and the
older pipeline the study inherits. Each item states the practice the literature has converged on, what the code did, and what
changed.

## A. Activation methodology vs. current practice

| # | Practice (source) | Before this audit | Now |
|---|---|---|---|
| A1 | **Capture at a defined token position in the residual stream**, document the layer indexing (`hidden_states[0]` = embeddings, `[k]` = stream entering decoder layer *k*, `[-1]` = post-final-norm). | Answer slot only (last context token before "The answer is" completion). Indexing undocumented. | Three positions per cell: answer slot, last token of *context+GT*, last token of *context+wrong* (statement representations, as in Marks & Tegmark 2023 "Geometry of Truth", Azaria & Mitchell 2023, and the erasure test of [Joswin et al. 2026](https://arxiv.org/abs/2607.00415)). Indexing documented in code; capture context recorded per row. |
| A2 | **Probe type**: mass-mean (difference-in-means) probes generalise better than logistic probes and are the standard readout (Marks & Tegmark 2023; [Im & Li 2025](https://arxiv.org/abs/2502.02716) for the steering analogue). | LR only, C=0.1, no baseline. | Mass-mean probe primary, L2 logistic secondary; both reported per layer. |
| A3 | **Control tasks**: shuffled-label probe (Hewitt & Liang 2019) and chance line must be reported next to every probe accuracy. | None. | Shuffled-label AUC per layer for both the truth and the belief-flip probe; chance = 0.5; below-chance on flipped items reported as *displacement* (the Joswin reading). |
| A4 | **Train on baseline, test under intervention** to distinguish suppression from erasure ([Joswin et al. 2026](https://arxiv.org/abs/2607.00415); Pandey 2026 [2604.19117](https://arxiv.org/abs/2604.19117); Wang et al. 2025 [2508.02087](https://arxiv.org/abs/2508.02087)). | Not possible (no statement capture). | Truth probe trained on control-context statements only; evaluated per pressure condition and on flipped-only items (same-item protocol, as in Joswin et al.) **and** on the pressure statements of held-out items (fold-trained probe never saw those items — removes item-specific leakage); leave-one-dataset-out generalisation (Marks & Tegmark). |
| A5 | **Cross-validation must respect the grouping** (items appear in 32 conditions). | GroupKFold by item ✓ | Unchanged; plus leave-one-dataset-out. |
| A6 | **Layer selection on held-out data, never on the test condition.** | Not applicable. | `best_layer` chosen on held-out *control* AUC; the pressure results are read at that layer and across all layers. |
| A7 | **Steering direction = difference-in-means; report directional agreement and separability, which predict steerability; always include a random-direction control; evaluate with likelihoods, not only sampled tokens** ([Im & Li 2025](https://arxiv.org/abs/2502.02716); [Braun et al. 2025](https://arxiv.org/abs/2505.22637); Tan et al. 2024; Rimsky et al. 2024 CAA). | No causal validation at all — decodable ≠ steerable. | `probe_analysis.py` saves per-layer raw mean-difference `pressure_dir` and `truth_dir`, and reports separability (Cohen's *d*) and directional agreement (mean cosine of per-item difference vectors with the mean direction, bootstrap CI). `belief_probe.py --steer-from` adds α·v (α ∈ {−1, +1} by default) at chosen layers on every position with an **equal-norm random-direction control**, re-scoring the forced-answer log-odds (likelihood metric) and a 16-token greedy readout; `steer_analysis.py` gives item-paired Δmargin with bootstrap CIs, steered vs unsteered flip rates, and the random control alongside. |
| A8 | **Cross-checkpoint comparisons need representation-similarity tools, not raw cosine** — centre activations; CKA / SVCCA across checkpoints ([Representation Collapse in Sequential Post-Training, 2026](https://arxiv.org/html/2605.30524); Kornblith et al. 2019). | Nothing. | `analysis_cross_checkpoint.py`: linear CKA per layer on identical cells, centred-mean cosine, and cosine between checkpoints' pressure/truth directions. Residual norms per layer reported (they grow with depth; all projections are centred per layer). |
| A9 | **Numerics**: store what you can reload; check ranges. | fp16 store, unverified. | Verified: max |x| < 2 at captured layers, no inf; fp16 kept (3× smaller than fp32). |
| A10 | **Reasoning models**: measure with and without the model's own reasoning; never score inside an open `<think>` block. | Raw context would have scored inside `<think>`. | Think checkpoints score after an *empty* think block and after their own budget-forced reasoning; natural-closure flag stored. |

What the literature does **not** settle and the study should therefore report as a robustness set: probe transfer after post-training is known to fail in some settings ([2602.20273](https://arxiv.org/abs/2602.20273)) — every checkpoint gets its own probes, and cross-checkpoint claims rest on CKA/direction cosines plus behaviour, not on transferring one checkpoint's probe to another.

## B. Code-chain findings (fixed in this pass unless marked)

| # | Finding | Severity | Status |
|---|---|---|---|
| B1 | `generate_batch` passed only `<|endoftext|>` (100257) as EOS, overriding OLMo's generation config `[<|im_end|>, <|endoftext|>]`; batched answers would run past the end of turn into fabricated follow-up turns. | High (would have corrupted every HPC greedy readout) | Fixed: stop set = tokenizer EOS ∪ `<|im_end|>` ∪ generation_config EOS; trimmed at the first stop id. Verified at 64 tokens: zero leaked turn markers. |
| B2 | `_contains` used substring matching on normalised text, so GT "8" matched an output containing "18". | High for numeric datasets (gsm8k, mmlu_math) | Fixed: whole-token match. Note: the *original* judge pipeline is unaffected (LLM judge), but any heuristic `parsed_answer_text` comparisons in `enhanced_scoring.py` should be checked for the same pattern. |
| B3 | `</think>` detection compared token ids only; a differently merged tokenisation would mark a closed trace as unclosed. | Medium | Text-based fallback with prefix-decoding search. |
| B4 | Auto-resubmit passed the GPU count from `SLURM_GPUS_ON_NODE` (unset under `--gpus-per-task`), so a 32B resume could land on one GPU. | Medium | GPU count/constraint/memory derived from the variant name. |
| B5 | Bundle step assumed `pyarrow` in the HPC venv. | Low | CSV.gz fallback. |
| B6 | Capture size: three positions × 10 layers × 12,800 rows ≈ 4.5 GB fp16 per 7B checkpoint (≈15 GB at all layers). | Info | Stays on scratch; `collect_bundle.sh` excludes it; goes to the external drive. |

Inherited-pipeline concerns that the new study must not repeat (not code changes here):

| # | Concern | Where | Consequence / mitigation |
|---|---|---|---|
| C1 | Judge (GPT-OSS-20B) is also an evaluated model. | cross-family runs | Exclude the judge from any ranking it labels, or use a second judge for that row. |
| C2 | Control system prompt licenses refusal ("If you are unsure, say you are unsure"); the warning prompt does not. | all `runs_latest` conditions | Refusal destinations are partly instruction-following; the factorial now crosses the clause explicitly (`*_nolicense` cells). |
| C3 | `max_new_tokens=128` for Instruct-path trials (`suite_7b_expanded.json`); 9–18% judge-null outputs are consistent with truncated verbose answers. | `runs_latest` | Treat judge-null as its own state (done in `analysis_judge_null.py`); the belief probe does not depend on free-text completion. |
| C4 | Reasoning-model outputs truncated (Think 256 tokens; API reasoning models 6–26% clean endings). | `runs_latest`, cross-family | Documented in FINDINGS §6; superseded by budget-forced measurement. |
| C5 | Injected wrong answers vary in plausibility (e.g. gsm8k "70000" vs "61256"); plausibility is uncontrolled in every prior comparison. | datasets | The belief probe records `lp_first_wrong` under control = a per-item plausibility covariate; include it in the mixed models. |
| C6 | OpenRouter T=0 is not guaranteed deterministic; no repeat-run check exists for the cross-family panel. | cross-family | Either re-run one cell 3× or bound ranking claims accordingly. |
| C7 | Dolci-DPO pairs are strong-vs-weak model pairs (no same-model pairs). | corpus audit | Restrict to `llm_judged` pairs; report both. |
| C8 | Corpus-audit `.py` sources were never committed; only `__pycache__` survives on T7. | dataset_analysis | Regexes are documented in the paper; re-implementations in `analysis_dolci_*.py`. |
| C9 | Three run folders (April matched-instruction runs, April Think-7B) were never committed. | `runs/` | Keep on the external drive; they are the source of FINDINGS §5. |
| C10 | The git metadata of this working copy had been wiped (Aug 8); re-attached to `vvm_rework`. | repo | Back up `.git` with the drive copy. |

## C. What the per-checkpoint job now produces

`<variant>.jsonl/.parquet` (belief + greedy readouts), `activations/<variant>_*.npz` (three positions × layers), `bundle/probe_<variant>.csv`
(per layer: held-out/shuffled/LODO truth AUC, truth AUC/accuracy under pressure, belief-flip AUC ± controls, separability, directional
agreement, best layer), `probe_<variant>_truth_by_condition.csv` (erasure test per condition, incl. flipped-only), `probe_<variant>_projections.csv`,
`directions_<variant>.npz`, `<variant>.steer.jsonl` + `steer_<variant>.csv` (causal validation with random control). Cross-checkpoint CKA is run
offline once two checkpoints exist.
