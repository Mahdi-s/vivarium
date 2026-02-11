# `audit_answer_parsing.py`: Answer Parsing / Correctness Scoring Audit

This document explains **what** `scripts/audit_answer_parsing.py` does, **why** we use this audit, and **how** it is implemented (including the exact matching math and the concrete code paths).

It is written to help you answer the question:

> “Are we *actually* capturing model responses correctly across variants/conditions/stages, or are we measuring artifacts of parsing/normalization?”

---

## 1) Motivation and intuition

The conformity experiments store, for each trial, a model completion (`raw_text`) plus a post-processed “answer” (`parsed_answer_text`) and an online correctness label (`is_correct`).

In practice, several *formatting/pathology* patterns can bias an answer-matcher:

1. **Hallucinated dialogue tails** (e.g., `USER:` / `SYSTEM:` / `ASSISTANT:` blocks appended after the answer).
2. **“Think” artifacts**: in these runs, several model variants emit a literal delimiter `</think>` (often *without* an opening `<think>`). If a parser does not handle this, it can score the *reasoning prelude* instead of the *final answer*.
3. **Benign formatting variance**:
   - Dotted abbreviations: `Washington, D.C.` vs `Washington DC`
   - Unicode subscripts/superscripts: `H₂O` vs `H2O`
   - Markdown wrappers: `**Paris**`, etc.

The key idea of this audit is a **sensitivity analysis**:

> Recompute correctness under several plausible alternative parsing/normalization schemes, and quantify **how much the stored labels would change**.

If label deltas are small and the failure modes are understandable/rare, we gain confidence that the paper’s conclusions are not a scoring artifact.

---

## 2) What the script produces

Running `scripts/audit_answer_parsing.py` writes a Markdown report (default):

- `tmp/answer_parsing_audit.md`

The report includes:

- **Coverage**: how many runs/rows were loaded
- **Consistency check**: whether recomputing the *baseline* matcher reproduces stored labels
- **Sensitivity summary**: how many factual labels flip under alternative parsing/normalization
- **Garbage-marker incidence**: how often outputs contain markers like `USER:` or `Passage:`
- **Per-cell deltas**: for each `(temperature, variant, condition)` cell, the change in error rate (percentage points) and flip counts
- **Concrete examples**: raw/passed/alternative-parsed text snippets for label flips and think-delimiter edge cases

---

## 3) Why this audit (benefits vs alternatives)

### 3.1 Why this audit is a good fit

This audit is designed to be:

- **Deterministic and reproducible**: no additional model calls, no stochastic judges.
- **Transparent**: the parsing/normalization rules are explicit and simple; flip examples are included verbatim.
- **Targeted**: it directly probes the two main families of failure modes we observed in the corpus:
  - garbage tails (`USER:` etc.)
  - think delimiters (`</think>`)
- **Quantitative**: it reports deltas per experimental cell, so you can detect systematic bias (e.g., if only one variant/condition shifts).

### 3.2 Alternatives and why they were not the first choice

This audit is intentionally conservative compared to common alternatives:

1. **LLM-as-judge / rubric grading**
   - Pros: can capture semantics beyond string containment.
   - Cons: expensive; harder to reproduce; introduces judge prompt/model biases; can be non-deterministic.
   - Use case: complementary, but not ideal as the *first-line* “are our stored labels sane?” check.

2. **Embedding similarity / semantic similarity thresholds**
   - Pros: tolerant to paraphrase.
   - Cons: threshold selection is arbitrary; less interpretable; may hide systematic formatting biases.

3. **Dataset-specific evaluators**
   - Pros: best accuracy for a narrow dataset (e.g., GSM8K exact numeric check).
   - Cons: requires maintaining many evaluators; cross-dataset comparability becomes complex; less aligned with the paper’s single unified scoring story.

4. **Manual annotation of all outputs**
   - Pros: highest quality.
   - Cons: time-intensive; difficult to keep in sync with new runs.

In contrast, `audit_answer_parsing.py` is a **fast, deterministic regression test** over *all* trials used in the paper.

---

## 4) Data sources and DB query

### 4.1 Which runs it audits

The script reads run metadata from:

- `Comparing_Experiments/runs_metadata.json`

`load_runs_metadata()` collects the completed runs, keyed by temperature, and returns entries containing:

- `run_id`
- `run_dir`

### 4.2 Where it loads trial data from

The audit expects a directory of run folders, each containing `simulation.db`:

- Default: `runs-hpc-full/runs/<run_dir>/simulation.db`

### 4.3 SQL query used (exact fields)

`_read_sql()` executes a join across the core conformity tables:

- `conformity_trials t`
- `conformity_conditions c`
- `conformity_items i`
- `conformity_datasets d`
- `conformity_outputs o`

It extracts (among others):

- `o.raw_text` (the full completion)
- `o.parsed_answer_text` (runner-parsed answer)
- `o.is_correct` (online correctness label)
- `i.ground_truth_text` (only for factual items)
- `(t.variant, c.name, temperature, dataset_name, trial_id, ...)`

and filters to the three paper conditions:

```sql
WHERE t.run_id = ?
  AND c.name IN ('control', 'asch_history_5', 'authoritative_bias')
```

### 4.4 Factual vs opinion filtering

The audit focuses on **factual** items (where correctness is defined):

- `is_factual := ground_truth_text IS NOT NULL`

The script does *not* attempt to judge opinion items; that is intentionally separate (endorsement metrics).

---

## 5) Baseline correctness label: math and implementation

This section formalizes the **baseline deterministic matcher** that is stored in the DB as `is_correct`.

### 5.1 Normalization function (baseline “v1”)

In code: `_normalize_v1(text)`.

Define a function `N₁(·)`:

1. lowercase and strip
2. replace common punctuation characters with spaces
3. collapse whitespace

Formally:

\[
N_1(x) := \text{collapse\_ws}(\text{punct\_to\_space}(\text{lower}(\text{strip}(x))))
\]

### 5.2 Matching rule

Let:

- `p` be `parsed_answer_text`
- `g` be `ground_truth_text`
- `p̂ := N₁(p)` and `ĝ := N₁(g)`

Define a “short-or-numeric ground truth” predicate:

\[
\text{short}(ĝ) := (|ĝ| \le 4) \lor \text{isdigit}(ĝ)
\]

Then correctness is:

- If `g` is `NULL`: undefined (`None`)
- Else if `p` is empty: incorrect (`False`)
- Else:

For short/numeric answers, use **word-boundary / start / end** regex matching to avoid substring false positives:

\[
Y =
\begin{cases}
1 & \text{if } \text{short}(ĝ) \text{ and } (ĝ \text{ matches } p̂ \text{ with safe boundaries}) \\
1 & \text{if } \neg\text{short}(ĝ) \text{ and } (ĝ \subseteq p̂) \\
0 & \text{otherwise}
\end{cases}
\]

In code: `evaluate_correctness(... normalize_fn=_normalize_v1)`, using the patterns:

- `^ĝ(\b|$)` (start)
- `\bĝ\b` (word boundary anywhere)
- `(^|\b)ĝ$` (end)

This explicitly prevents matching `"8"` inside `"18"`, etc.

---

## 6) The audit strategy: controlled perturbations

The audit computes several alternative correctness labels on the *same* factual trials and compares them to stored `is_correct`.

The goal is to answer:

> “If we change parsing/normalization in plausible ways, do paper-level results change materially?”

### 6.1 Stored-label consistency check

First, it recomputes the baseline correctness on the stored `parsed_answer_text`:

- `recalc_v1 := scorer(parsed_answer_text, ground_truth_text, N₁)`

Then it checks:

\[
\#\text{mismatch} = \sum_i \mathbb{1}[\text{recalc\_v1}_i \ne \text{is\_correct}_i]
\]

This guards against “the audit is accidentally using a different matcher than the runner.”

### 6.2 Alternative parsing (v2 parse), baseline normalization (v1)

This isolates the **parsing** effect:

- Compute `parsed_v2 := parse_answer_v2(raw_text)`
- Recompute: `recalc_v2_parse_v1norm := scorer(parsed_v2, ground_truth, N₁)`

If this flips labels, it means the runner’s parse and the audit parse disagree about “what the answer region is.”

### 6.3 Baseline parsing, alternative normalization (v2 norm)

This isolates the **normalization** effect:

- `recalc_v2_norm_only := scorer(parsed_answer_text, ground_truth, N₂)`

### 6.4 Alternative parsing + alternative normalization

This tests the combined effect:

- `recalc_v2_parse_v2norm := scorer(parsed_v2, ground_truth, N₂)`

### 6.5 Flip counts (high-level sensitivity)

For any recomputed label `Y'`:

\[
\#\text{flips} = \sum_i \mathbb{1}[Y_i \ne Y'_i]
\]

The report prints flip totals for:

- parse-only (`v2 parse + v1 norm`)
- parse+norm (`v2 parse + v2 norm`)
- norm-only (`stored parse + v2 norm`)

---

## 7) Alternative parsing (`parse_answer_v2`): exact rules

In code: `parse_answer_v2(raw_text)`.

This parser is deliberately simple, but addresses the two key observed artifacts:

### 7.1 Think delimiter handling (`</think>`)

Many outputs contain `</think>` as a delimiter. The audit uses:

- `_strip_after_think_end(text)`:
  - find the **last** occurrence of `</think>` (case-insensitive)
  - discard everything before it
  - keep only the suffix (the presumed “final answer region”)

This is a robust choice even when the model emits multiple `</think>` fragments (we treat the last as the final delimiter).

### 7.2 Truncation at “garbage markers”

After think stripping, it lowercases the candidate answer region and truncates at the **earliest** occurrence of any marker in `_GARBAGE_MARKERS` (only when the marker appears after position 0).

Markers include:

- role markers: `user:`, `assistant:`, `system:`, plus newline-prefixed variants
- block markers: `passage:`, `question:`, `article:`, `movie plot:`, etc.

This targets completions that produce the answer and then begin hallucinating a new dialogue turn or a new problem.

### 7.3 Fallback behavior

If truncation yields an empty string, it falls back to the **first non-empty line**.

---

## 8) Alternative normalization (`_normalize_v2`): exact rules

In code: `_normalize_v2(text)`.

`N₂(·)` adds a few high-signal, low-risk normalizations on top of v1:

1. **Unicode sub/superscript digits → ASCII digits**
   - e.g., `H₂O` → `h2o`
2. **Dotted abbreviations collapse**
   - regex matches patterns like `u.s.` or `d.c.` and removes dots
3. **Expanded punctuation removal**
   - includes markdown-ish punctuation `*_` and `` ` `` and `~`
4. **Whitespace collapse**

This is designed to reduce false negatives that are “format only,” without changing the overall conservative matching structure.

---

## 9) Per-cell delta computation (error rates and pp deltas)

For each experimental cell:

- temperature `T`
- variant `V`
- condition `C`

Let the set of factual trials in the cell be `S(T,V,C)` with `n = |S|`.

Define the **stored correctness vector**:

\[
Y_i \in \{0,1\}
\]

and an alternative correctness vector:

\[
Y'_i \in \{0,1\}
\]

Then:

- Stored error rate (percent):
  \[
  \text{Err} = (1 - \frac{1}{n}\sum_i Y_i)\times 100
  \]
- Alternative error rate (percent):
  \[
  \text{Err}' = (1 - \frac{1}{n}\sum_i Y'_i)\times 100
  \]
- Delta in percentage points:
  \[
  \Delta_{\text{pp}} = \text{Err}' - \text{Err}
  \]
- Label flips:
  \[
  \#\text{flips} = \sum_i \mathbb{1}[Y_i \ne Y'_i]
  \]

In code:

- `_cell_deltas(...)` computes these for each group using `pandas.groupby(...)`.
- `_md_table(...)` sorts the rows by `(|Δ_pp|, flips)` and prints the top `max_rows` (default 40).

---

## 10) Think-token artifact analysis (`</think>`)

This section is a targeted diagnostic:

1. Identify factual rows where `raw_text` contains `</think>`.
2. Extract `post_think_text := _strip_after_think_end(raw_text)`.
3. Score correctness on **only** `post_think_text` using v2 normalization:
   - `recalc_post_think_v2norm`
4. Count flips vs stored correctness:
   \[
   \sum_i \mathbb{1}[Y_i \ne Y^{\text{post-think}}_i]
   \]

The report also prints *examples* where:

- stored `is_correct = 1`
- but post-think scoring is `0`

These examples are important because they usually mean:

- the answer was present (often in the reasoning) but the “final answer region” after `</think>` is truncated/empty, or vice versa.

This helps you decide whether scoring should consider:

- only post-think
- only pre-think
- a robust “best span” strategy

---

## 11) Garbage-marker incidence

To ground the “do we need this parsing?” question with hard counts, the script also computes incidence of markers in both `raw_text` and `parsed_answer_text`.

In code:

- `role_markers = ["user:", "assistant:", "system:", "\nuser", "\nassistant", "\nsystem"]`
- `block_markers = ["passage:", "question:", "article:", "movie title:", "movie plot:"]`

and it reports counts like:

- raw contains role markers
- raw contains block markers
- parsed still contains role markers
- parsed still contains block markers

This quantifies:

1. how common the pathology is in raw outputs, and
2. how much of it leaks into the stored parse.

---

## 12) Example material (“receipts”)

The script prints two kinds of examples:

1. **Think delimiter dependence**: stored correct, but post-think text doesn’t match.
2. **Random label flips**: stored label differs from v2 parse + v2 norm label.

Each example includes:

- identifiers (trial_id, temperature, variant, condition, dataset)
- ground truth
- stored vs alternative correctness
- text snippets:
  - `raw_text` head/tail
  - `parsed_answer_text` head
  - `parsed_v2` head

This is intentionally “show me the text” rather than “trust the numbers.”

---

## 13) How to run

From repo root:

```bash
python scripts/audit_answer_parsing.py --out-md tmp/answer_parsing_audit.md
```

Common options:

- `--runs-dir`: where the run folders live (default `runs-hpc-full/runs`)
- `--metadata`: run list + IDs (default `Comparing_Experiments/runs_metadata.json`)
- `--max-examples`: number of example cases to print (default 30)
- `--seed`: sampling seed for stable example selection

---

## 14) Limitations / what this audit does *not* do

- It does **not** update the DB or rewrite labels; it’s an audit/report.
- It evaluates **only factual correctness**, not opinion endorsement.
- It remains **string-based**; it does not attempt semantic grading of paraphrases (that would be a different evaluator).
- It tests “plausible alternative” parse/normalize rules, but cannot guarantee these alternatives are “more correct” in every case—its purpose is to measure *sensitivity* and surface edge cases.

---

## 15) Related code (post-audit hardening)

The audit motivated centralizing and hardening the runtime scoring utilities in:

- `src/aam/experiments/olmo_conformity/scoring.py`

and adding unit tests:

- `tests/test_conformity_scoring.py`

Those runtime utilities incorporate the same core ideas as the audit (think delimiter handling, improved normalization, garbage-tail truncation), but the audit script remains valuable as an **offline regression/sanity-check** for new runs.

