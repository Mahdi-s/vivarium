# Statistical Audit of Existing Dataset Analysis Pipeline

**Audit date:** 2026-04-28  
**Files audited:** `dataset_analysis/scripts/common.py` (183 lines), `phase1_sft_audit.py` (149 lines), `phase2_dpo_audit.py` (157 lines), `phase4_rl_audit.py` (145 lines)  
**Result JSONs inspected:** `results/phase1_instruct-sft_summary.json` (N=1,944,831), `results/phase2_instruct-dpo_summary.json` (N=259,785)

---

## 1. Skewness Error

Every "headline" statistic reported by the pipeline is a mean of a distribution that the data confirm is heavily right-skewed (or zero-inflated). In each case the median diverges substantially from the mean, making the mean a misleading central-tendency estimate.

### 1a. `common.py:177` — `summarise()` reports mean as primary field

```python
# common.py:170-182
def summarise(values: Iterable[float]) -> dict:
    vs = [v for v in values if v is not None]
    if not vs:
        return {"n": 0}
    import statistics as st
    return {
        "n": len(vs),
        "mean": st.fmean(vs),      # <-- headline field; always first
        "median": st.median(vs),
        "stdev": st.pstdev(vs) if len(vs) > 1 else 0.0,
        "min": min(vs),
        "max": max(vs),
    }
```

`median` is present but appears second and is never foregrounded by any downstream consumer. IQR, percentile bands (P5/P25/P75/P95), and skewness coefficient are absent. All phase scripts call `summarise()` and the JSON output ordering places `mean` first, making it the de-facto headline.

**Confirmed skew from result JSONs:**

| Metric (phase1 SFT, N=1.94M) | mean | median | ratio |
|---|---|---|---|
| `ngram_overlap_n4` | 0.0513 | 0.0071 | **7.2×** |
| `sycophancy_hits` | 0.0032 | 0 | ∞ (zero-inflated) |
| `structural_jaccard` | 0.4168 | 0.3333 | 1.25× |

| Metric (phase2 DPO deltas, N=259,785) | mean | median | stdev |
|---|---|---|---|
| `delta_struct_jaccard` | +0.029 | **0.000** | 0.358 |
| `delta_ngram_overlap` | +0.048 | **+0.005** | 0.197 |
| `delta_sycophancy` | -0.015 | **0** | 0.195 |
| `delta_correction` | +0.011 | **0** | **1.691** |

The `delta_correction` case is especially severe: stdev=1.691 with min=-239 and max=179 (per-pair raw hit-count differences) against a mean of +0.011 — the mean is dominated entirely by a small tail of outlier pairs.

### 1b. `phase2_dpo_audit.py:132` — `sign_test()` embeds its own `mean`

```python
# phase2_dpo_audit.py:128-133
def sign_test(xs):
    pos = sum(1 for v in xs if v > 0)
    neg = sum(1 for v in xs if v < 0)
    return {"n": len(xs), "positive": pos, "negative": neg,
            "mean": st.fmean(xs) if xs else 0.0,   # <-- redundant + misleading
            "pct_positive": pos / max(1, len(xs))}
```

This embeds a second, independent `mean` field inside the sign-test block (in addition to the one already inside `summarise()`). The `sign_tests` block and `delta_stats` block in the output JSON both report the same skewed means, creating two places downstream code can read an unreliable headline.

### 1c. `phase1_sft_audit.py:131` — `summarise()` applied uniformly across incompatible metric types

```python
# phase1_sft_audit.py:127-131
summary = {
    ...
    "metrics": {k: summarise(v) for k, v in agg.items()},
    ...
}
```

`summarise()` is applied identically to:
- `ngram_overlap_n4` — a continuous [0,1] metric with heavy right skew (mean=0.051, median=0.007).
- `sycophancy_hits` — a count variable, zero-inflated (mean=0.003, median=0), maximum=3.
- `affirm_prefix` — a Bernoulli binary (0/1); for binary data, mean equals the rate, which is fine, but labelling it `"mean"` obscures that it is a prevalence.
- `structural_jaccard` — a continuous [0,1] metric, moderate skew.
- `prompt_len` / `resp_len` — length in characters; highly right-skewed (mean=859, median=539 for prompts; stdev=2208 >> median).

Lumping all of these under a single `summarise()` call with the same field names treats them as interchangeable, which they are not.

### 1d. `phase4_rl_audit.py:136` — same `summarise()` pattern propagated

```python
# phase4_rl_audit.py:130-136
summary = {
    ...
    "metrics": {k: summarise(v) for k, v in acc.items() if v},
}
```

Applied to `repeat_frames` (count, zero-inflated), `has_repeat_frame` (binary 0/1), `comp_correction` (count), `comp_sycophancy` (count), `comp_ngram_overlap` (continuous, right-skewed). Same taxonomy confusion as phase1.

---

## 2. Effect Size Fallacy

The pipeline tests directionality (which way does the delta point?) but never quantifies magnitude. At N=259,785, virtually any non-zero population effect reaches p<10⁻⁴; the sign test is statistically trivial and says nothing about practical importance.

### 2a. `phase2_dpo_audit.py:128-133` — `sign_test()` returns only `pct_positive`, no effect size

```python
# phase2_dpo_audit.py:128-133
def sign_test(xs):
    pos = sum(1 for v in xs if v > 0)
    neg = sum(1 for v in xs if v < 0)
    return {"n": len(xs), "positive": pos, "negative": neg,
            "mean": st.fmean(xs) if xs else 0.0,
            "pct_positive": pos / max(1, len(xs))}   # <-- only magnitude proxy
```

`pct_positive` functions as the sole effect-size proxy. For `delta_sycophancy`: pct_positive=0.4% (1,055 positive vs. 3,976 negative out of 259,785 pairs), but the effect-size interpretation requires knowing whether the *magnitude* of those differences is meaningful, not just their sign. No Cohen's d, Cliff's delta, Hodges-Lehmann delta, or any standardized effect-size measure is computed anywhere in the codebase.

### 2b. `phase2_dpo_audit.py:143-148` — `interpretation` dict makes unfalsifiable directional claims

```python
# phase2_dpo_audit.py:143-148
"interpretation": {
    "delta_struct_jaccard": "positive -> DPO penalises structural mirroring",
    "delta_ngram_overlap":  "positive -> DPO penalises direct prompt copying",
    "delta_sycophancy":     "positive -> DPO penalises sycophantic agreement",
    "delta_correction":     "positive -> DPO rewards factual correction markers",
},
```

These claims are directionally consistent with the means being positive (or negative), but they are written as if the direction is itself sufficient evidence. There is no minimum-effect threshold, no confidence interval, and no correction for the `delta_correction` outlier problem (stdev=1.691 dwarfs mean=0.011). The interpretation for `delta_sycophancy` is based on a mean of -0.015 against a distribution with median=0 — the directional claim is debatable.

### 2c. `phase4_rl_audit.py:135` — `repeat_frame_rate` presented without effect-size context

```python
# phase4_rl_audit.py:135
"repeat_frame_rate": with_frame / max(1, n),
```

This is a prevalence proportion (fraction of prompts with ≥3 repeat frames). The summary then implies that a high rate is "good" for Asch-resistance training. No baseline prevalence is provided, no comparison to other RL datasets, and no effect size linking the rate to downstream sycophancy vulnerability.

---

## 3. Regex Trap

All semantic signal in the pipeline flows through regex patterns. These patterns are English-only, match narrow surface forms, and miss substantial shares of true positives. They are used to make population-level claims about the training corpora.

### 3a. `common.py:126-133` — `SYCOPHANCY_RE` (verbatim)

```
\b(you(?:'re| are) (?:absolutely |completely |totally )?(?:right|correct)|that'?s (?:a )?(?:great|excellent|wonderful|fantastic|brilliant) (?:question|point|idea|observation)|great question|wonderful question|i agree (?:with you|completely)|as you (?:correctly |rightly )?(?:said|pointed out|noted))\b
```

Used at: `phase1_sft_audit.py:105` (`count_hits(SYCOPHANCY_RE, asst)`), `phase2_dpo_audit.py:61` (`score_response`), `phase4_rl_audit.py:113` (`count_hits(SYCOPHANCY_RE, c)`).

**Coverage gaps:**
- English-only. Dolci datasets include multilingual examples (Aya, multilingual SFT). Non-English sycophantic markers are invisible.
- Requires exact closed-set adjectives: "amazing", "insightful", "thoughtful", "spot on", "exactly right", "100%", "totally correct" all bypass the pattern.
- Misses implicit sycophancy via agreement-without-qualification (e.g., simply restating the user's premise as fact).
- `\b` word-boundary anchors interact unpredictably with Unicode; a response containing "you're right" in a Hebrew or Arabic script context would not be matched.
- No coverage of paraphrastic forms: "I think you've identified something important", "Your analysis is spot on."

### 3b. `common.py:118-124` — `CORRECTION_RE` (verbatim)

```
\b(actually|however|in fact|it'?s worth noting|the premise is (?:incorrect|wrong|mistaken)|that's not (?:quite )?right|i (?:have to )?disagree|important to note|on the contrary|this is (?:a )?(?:misconception|incorrect))\b
```

Used at: `phase2_dpo_audit.py:62`, `phase4_rl_audit.py:112`.

**Coverage gaps:**
- "actually" and "however" are common discourse markers used in non-corrective contexts (e.g., "Actually, here is a list of..."). This inflates `correction_hits` with false positives, undermining `delta_correction` as a signal. This is confirmed by the extreme range in `delta_correction` (min=-239, max=179).
- Missing patterns: "to be precise", "more accurately", "I should clarify", "the correct answer is", "that's a common misconception" (hyphenated form), "not quite", "let me correct that."
- English-only.

### 3c. `common.py:111-116` — `AFFIRM_PREFIXES` / `AFFIRM_RE` (verbatim)

```
^\s*(yes[,!.]|absolutely[,!.]?|certainly[,!.]?|of course[,!.]?|sure[,!.]?|i agree|you[' ]re (right|correct)|great question|here(?: is| are|’s| is a)|here's)
```

Used at: `common.py:137` (`affirm_prefix()`), propagated to all three phase scripts.

**Coverage gaps:**
- Window is `text[:80]` — any response opening with a preamble longer than 80 characters before an affirmation is missed.
- Does not match: "Of course!", "Absolutely!", "Definitely", "Sure thing", "You got it", "Happy to help", "Glad you asked", "Certainly!" (with capital C after sentence break).
- The Unicode right-single-quotation-mark (`’`) is hardcoded for "here's" but not for "you're", creating inconsistent Unicode handling.
- `sure[,!.]?` matches "sure" as a standalone word but not "Sure enough" or "Sure, let me...".

### 3d. `common.py:75` — `COLON_LINE_RE` used for Asch-frame detection (verbatim)

```
(?m)^[A-Za-z][^\n:]{0,40}:\s
```

Used in `structural_fingerprint()` at `common.py:89` to count "colon lines" (e.g., "Participant 1:"), which feeds into `structural_jaccard`.

**Coverage gaps:**
- Matches any `Name:` pattern on a line start (dictionary definitions, code keys, markdown headers without `#`). This inflates `colon_line` counts for legitimate technical content.
- The `{0,40}` length cap on the label misses longer role labels common in multi-agent transcripts ("Interviewer/Moderator:", "Research Participant 12:").
- Does not distinguish "Participant 1:" (the Asch frame) from "Note:" or "Warning:" (common document structure) — the same count feeds different semantic claims.

### 3e. `phase4_rl_audit.py:34-38` — `REPEAT_FRAME_RE` (verbatim, **template missed this**)

```
(?:(?:participant|string|person|user|voter|respondent|source|reviewer|choice|option|item)\s*\d+\s*[:.\)])
```

Used at `phase4_rl_audit.py:106` (`len(REPEAT_FRAME_RE.findall(p))`).

**Coverage gaps:**
- Keyword list is closed and English-only: misses "subject", "speaker", "agent", "panelist", "judge", "expert", "respondent" (covered) but also "Rater", "Annotator", "Evaluator", "Witness", "Candidate."
- Fails to match zero-padded numbering ("Participant 01:") or Roman numerals ("Voter II.").
- The `>=3` threshold at line 107 is hardcoded with no empirical justification — a prompt with exactly "Participant 1" and "Participant 2" (two confederates, one target) scores 0 and is not counted as an Asch frame.
- "string" in the keyword list appears to be a mistake (likely a copy-paste from a code list of types rather than a human-role label).

---

## 4. Additional Errors Not Covered by the Plan Template

### 4a. `common.py:179` — population stdev instead of sample stdev

```python
"stdev": st.pstdev(vs) if len(vs) > 1 else 0.0,
```

`st.pstdev()` computes the population standard deviation (divides by N). For a sample drawn from a corpus, `st.stdev()` (divides by N-1, i.e., Bessel's correction) is correct. At N=1.9M the difference is negligible, but at the per-batch or subset level this is a statistical error. All downstream `stdev` fields in every summary JSON are population stdevs presented without that qualification.

### 4b. `phase2_dpo_audit.py:135-148` — `delta_stats` and `sign_tests` contain redundant and partially inconsistent `mean` fields

The summary JSON contains both:
- `delta_stats.delta_X.mean` (from `summarise()`, line 139)
- `sign_tests.delta_X.mean` (from `sign_test()`, line 132)

Both should be identical (same `fmean` on the same list), but the code computes them independently from the same accumulated list. Any future refactor that changes one accumulation path creates a silent inconsistency. The redundancy also means downstream consumers have two places to read the same (unreliable) headline number.

### 4c. `phase4_rl_audit.py:107` — hardcoded threshold creates binary metric with no justification

```python
has_repeat_frame = int(frames >= 3)
```

The threshold `>= 3` is undocumented. A prompt with 2 repeat frames (two confederate voices) is the minimum Asch configuration but is excluded. This threshold propagates into `summarise()` as a binary metric, and its `mean` is then reported as the "repeat_frame_rate" proxy — the same binary-mean pattern flagged for `affirm_prefix` in section 1c, compounded by an arbitrary binarization.

### 4d. `phase1_sft_audit.py:133-137` — duplicate rate computation outside `summarise()`

```python
"affirm_prefix_rate":
    sum(agg["affirm_prefix"]) / max(1, len(agg["affirm_prefix"])),
```

This manually recomputes the mean of `affirm_prefix` outside `summarise()`, duplicating `summary["metrics"]["affirm_prefix"]["mean"]`. For a binary metric they are identical, but the pattern is fragile: if the two code paths diverge (e.g., one applies filtering), they silently disagree. The same duplication appears for `prompt_has_list_rate`.

### 4e. `phase2_dpo_audit.py:139` and `phase2_dpo_audit.py:140-141` — `affirm_prefix` delta is not computed

`w_aff` and `l_aff` are accumulated but only reported as rates:
```python
"affirm_prefix_rate_chosen":   sum(w_aff) / max(1, len(w_aff)),
"affirm_prefix_rate_rejected": sum(l_aff) / max(1, len(l_aff)),
```

There is no `delta_affirm_prefix` in `deltas`, so `affirm_prefix` is excluded from the sign-test analysis even though it is computed for every pair. The result JSON confirms this: chosen rate=0.1548, rejected rate=0.1566 — a difference of 0.0018 that is never tested or reported as a delta.

---

## 5. Headline Numbers to Recompute

The following numbers appear in the summary JSONs as "means" and are cited (or will be cited) as headline findings. Each should be replaced with a robust central-tendency estimate (median + IQR or Hodges-Lehmann) and accompanied by a standardized effect size.

| Statistic | Current value (mean) | Median | Notes |
|---|---|---|---|
| `delta_ngram_overlap` | +0.048 | +0.005 | 54.5% positive; mean inflated by right tail |
| `delta_struct_jaccard` | +0.029 | **0.000** | 23.2% positive; median=0 means majority of pairs have no delta |
| `delta_sycophancy` | -0.015 | **0** | 0.4% positive; median=0, mean driven by 3,976 negative-delta pairs |
| `delta_correction` | +0.011 | **0** | 9.3% positive; stdev=1.691 with outliers ±239 — mean is uninformative |
| `affirm_prefix_rate` (SFT) | 11.7% | — | Binary rate; denominator is 1.94M; no confidence interval reported |
| `ngram_overlap_n4` (SFT) | 0.051 | 0.007 | 7.2× mean/median gap; right-skewed continuous |
| `sycophancy_hits` (SFT) | 0.003 | 0 | Zero-inflated count; mean is near-meaningless |
| `repeat_frame_rate` (phase4) | TBD | — | Depends on `>= 3` threshold; needs justification and baseline |

**Minimum fixes required for each number:**
1. Replace `mean` with `median` (or Hodges-Lehmann for paired data) as the headline.
2. Add IQR (P25–P75) alongside median.
3. Replace `pct_positive` with Cliff's delta or a Hodges-Lehmann confidence interval as the effect-size measure.
4. Handle `delta_correction` outliers explicitly (winsorize or report separately).
5. Add a population-stdev → sample-stdev correction note where N is small enough to matter.

---

## Summary Table of Error Sites

| Error type | File | Line(s) | Description |
|---|---|---|---|
| Skewness | `common.py` | 177 | `st.fmean()` as primary field in `summarise()`; no IQR |
| Skewness | `common.py` | 179 | `st.pstdev()` (population) instead of `st.stdev()` (sample) |
| Skewness | `phase1_sft_audit.py` | 131 | `summarise()` applied uniformly to binary, count, and continuous metrics |
| Skewness | `phase2_dpo_audit.py` | 132 | Redundant `mean` inside `sign_test()`; skewed distributions |
| Skewness | `phase2_dpo_audit.py` | 139 | `delta_stats` block duplicates skewed mean from `summarise()` |
| Skewness | `phase4_rl_audit.py` | 136 | `summarise()` applied to binary and count metrics without distinction |
| Effect size | `phase2_dpo_audit.py` | 128–133 | `sign_test()` has no standardized effect size |
| Effect size | `phase2_dpo_audit.py` | 143–148 | `interpretation` dict makes directional claims without magnitude |
| Effect size | `phase4_rl_audit.py` | 135 | `repeat_frame_rate` is a prevalence with no effect-size context |
| Regex trap | `common.py` | 126–133 | `SYCOPHANCY_RE`: English-only, narrow adjective set |
| Regex trap | `common.py` | 118–124 | `CORRECTION_RE`: false positives from "actually"/"however" |
| Regex trap | `common.py` | 111–116 | `AFFIRM_RE`: 80-char window; Unicode inconsistency |
| Regex trap | `common.py` | 75 | `COLON_LINE_RE`: matches non-Asch colon patterns |
| Regex trap | `phase4_rl_audit.py` | 34–38 | `REPEAT_FRAME_RE`: closed English keyword list; "string" is erroneous |
| Additional | `phase1_sft_audit.py` | 133–137 | Duplicate rate computation outside `summarise()` |
| Additional | `phase2_dpo_audit.py` | 140–141 | `delta_affirm_prefix` never computed despite per-pair data being available |
| Additional | `phase4_rl_audit.py` | 107 | `>= 3` threshold undocumented and excludes 2-confederate Asch configurations |
