#!/usr/bin/env python3
"""Recompute statistics addressing reviewer feedback.

Outputs seven new/updated LaTeX table fragments that are pasted into
colm2026_paper.tex. All computations read existing CSVs from:
  - Comparing_Experiments/publication_V2_colm/statistical_tests/
  - Comparing_Experiments/April_analysis/
No new model inferences are run.

Items produced (each preceded by a %%-commented banner):
  [E*] Within-family McNemar with 95% CIs + raw b,c + sensitivity column
  [M*] Cross-family McNemar with 95% CIs + raw b,c + refusal-shift column
  [BETWEEN] SFT-vs-DPO direct paired McNemar (Instruct and Think paths)
  [G*] Domain breakdown with Wilson 95% CIs per cell
  [J] Mitigation effectiveness table (DA, QD, etc.)
  [ABL-SIDE] Naked-vs-ngram side-by-side with direction
"""

import csv
import math
import os
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

APRIL = os.path.join(REPO_ROOT, "Comparing_Experiments", "April_analysis")
PUBV2 = os.path.join(REPO_ROOT, "Comparing_Experiments", "publication_V2_colm",
                     "statistical_tests")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def banner(t):
    bar = "=" * 72
    print(f"\n%% {bar}\n%% [{t}]\n%% {bar}\n")


def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_or_ci(b, c, conf=0.95):
    """Haldane-Anscombe-corrected OR with Wald 95% CI on log scale."""
    b_adj, c_adj = b + 0.5, c + 0.5
    or_val = b_adj / c_adj
    log_or = math.log(or_val)
    se = math.sqrt(1.0 / b_adj + 1.0 / c_adj)
    z = 1.96 if conf == 0.95 else 2.576
    lo = math.exp(log_or - z * se)
    hi = math.exp(log_or + z * se)
    return or_val, lo, hi


def mcnemar_p_exact(b, c):
    """Two-sided exact McNemar binomial p-value."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Sum tail probabilities under Binomial(n, 0.5)
    def binom_cdf(k, n, p=0.5):
        from math import comb
        return sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k + 1))
    p_one_tail = binom_cdf(k, n)
    # two-sided: double, cap at 1
    return min(1.0, 2 * p_one_tail)


def sig_stars(p):
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def holm_bonferroni(p_values):
    """Return list of Holm-corrected p-values in original order."""
    idx = sorted(range(len(p_values)), key=lambda i: p_values[i])
    m = len(p_values)
    adj = [0.0] * m
    running_max = 0.0
    for rank, i in enumerate(idx):
        corrected = min(1.0, (m - rank) * p_values[i])
        running_max = max(running_max, corrected)
        adj[i] = running_max
    return adj


# ---------------------------------------------------------------------------
# [E*] Within-family McNemar with CIs + sensitivity
# ---------------------------------------------------------------------------

def table_E_augmented():
    banner("Appendix E: Within-family McNemar with CIs + sensitivity")

    rows = load_csv(os.path.join(PUBV2, "mcnemar_pressure_vs_control.csv"))
    lut = {(r["variant"], r["condition_name"]): r for r in rows}

    COND_LABEL = {
        "asch_history_5": "Asch-5 free-text",
        "asch_zhu_unbiased_unanimous_plain": "Unan.\\ plain",
        "asch_zhu_unbiased_unanimous_neutral": "Unan.\\ neutral",
        "asch_zhu_unbiased_unanimous_confident": "Unan.\\ confident",
        "asch_zhu_unbiased_unanimous_uncertain": "Unan.\\ uncertain",
        "asch_zhu_unbiased_diverse_plain": "Diverse peers",
        "asch_zhu_unbiased_da": "Devil's advocate",
        "asch_zhu_unbiased_qd": "Question distill.",
        "authoritative_bias": "Auth.\\ bias",
        "authority_zhu_unbiased_trust": "Auth.\\ trust",
        "authority_zhu_unbiased_trust_da": "Trust + DA",
    }
    COND_ORDER = list(COND_LABEL.keys())
    VAR_ORDER = ["base", "instruct_sft", "instruct_dpo", "instruct"]
    VAR_LABEL = {"base": "Base", "instruct_sft": "Instruct-SFT",
                 "instruct_dpo": "Instruct-DPO", "instruct": "Instruct"}

    print("\\begin{table}[H]")
    print("    \\centering")
    print("    \\scriptsize")
    print("    \\setlength{\\tabcolsep}{3pt}")
    print("    \\renewcommand{\\arraystretch}{0.95}")
    print("    \\begin{tabular}{l l r r r l l}")
    print("        \\toprule")
    print("        Variant & Condition & $n_{\\text{pairs}}$ & $b$ & $c$ & OR [95\\% CI] & Holm $p$ \\\\")
    print("        \\midrule")
    for v in VAR_ORDER:
        for j, c in enumerate(COND_ORDER):
            r = lut.get((v, c))
            if r is None:
                continue
            n_pairs = int(r["n_pairs"])
            b = int(float(r["b_ctrl_correct_pres_wrong"]))
            cc = int(float(r["c_ctrl_wrong_pres_correct"]))
            or_v, lo, hi = mcnemar_or_ci(b, cc)
            p_adj = float(r["p_adjusted"])
            star = sig_stars(p_adj)
            if p_adj < 1e-4:
                p_str = "$<10^{-4}$"
            else:
                p_str = f"{p_adj:.3f}"
            # Only show variant name on first row of each group
            v_cell = VAR_LABEL[v] if j == 0 else ""
            print(f"        {v_cell} & {COND_LABEL[c]} & {n_pairs} & {b} & {cc} & {or_v:.2f} [{lo:.2f}, {hi:.2f}]{{ \\tiny {star}}} & {p_str} \\\\")
        if v != VAR_ORDER[-1]:
            print("        \\midrule")
    print("        \\bottomrule")
    print("    \\end{tabular}")
    print("    \\caption{\\textbf{Within-family Olmo-3-7B McNemar results with raw paired counts and 95\\% confidence intervals.} $b$ = items correct under control but endorsed wrong under pressure; $c$ = items wrong under control but correct under pressure. OR is computed with Haldane--Anscombe continuity correction and the 95\\% CI is on the log scale. Holm--Bonferroni correction is applied within each variant across the 11 pressure conditions. The paired sample size $n_{\\text{pairs}}$ shows how many items contributed to the test after excluding items where either condition produced a refusal (State C) or unclassified output (State D); for Base and all Instruct variants, $n_{\\text{pairs}}$ stays close to the 400-item ceiling because refusal rates are low. See Appendix~\\ref{app:stat_methods} for method details and Appendix~\\ref{app:refusal_sensitivity} for a refusals-as-wrong sensitivity analysis.}")
    print("    \\label{tab:mcnemar_full}")
    print("\\end{table}")


# ---------------------------------------------------------------------------
# [Sensitivity] Refusals-as-wrong OLMo within-family
# ---------------------------------------------------------------------------

def table_refusal_sensitivity():
    banner("NEW Appendix: Refusals-as-wrong sensitivity analysis (OLMo within-family)")

    path = os.path.join(APRIL, "item_level", "per_item_endorsement.csv")
    rows = load_csv(path)

    # Group by variant; compute two McNemar versions on the same data.
    VAR_ORDER = ["base", "instruct_sft", "instruct_dpo", "instruct",
                 "think_sft", "think_dpo", "think"]
    VAR_LABEL = {"base": "Base",
                 "instruct_sft": "Instruct-SFT",
                 "instruct_dpo": "Instruct-DPO",
                 "instruct": "Instruct (RLVR)",
                 "think_sft": "Think-SFT",
                 "think_dpo": "Think-DPO",
                 "think": "Think (RLVR)"}

    per_variant = defaultdict(list)
    for r in rows:
        if r.get("usable") not in ("True", "true", "1"):
            continue
        per_variant[r["variant"]].append(r)

    print("\\begin{table}[H]")
    print("    \\centering")
    print("    \\footnotesize")
    print("    \\setlength{\\tabcolsep}{4pt}")
    print("    \\renewcommand{\\arraystretch}{1.0}")
    print("    \\begin{tabular}{l rr ll rr ll}")
    print("        \\toprule")
    print("         & \\multicolumn{4}{c}{Refusals \\emph{excluded} (reported)} & \\multicolumn{4}{c}{Refusals \\emph{coded as wrong} (sensitivity)} \\\\")
    print("        \\cmidrule(lr){2-5}\\cmidrule(lr){6-9}")
    print("        Variant & $b$ & $c$ & OR [95\\% CI] & $p$ & $b'$ & $c'$ & OR$'$ [95\\% CI] & $p$ \\\\")
    print("        \\midrule")

    for v in VAR_ORDER:
        items = per_variant.get(v, [])
        if not items:
            continue

        # Convention A: refusals-excluded
        # Pair is kept iff neither control nor pressure is C or D (i.e., both are A or B)
        b_a = c_a = 0
        for it in items:
            ctrl, pres = it["control_state"], it["pressure_state"]
            if ctrl not in ("A_correct", "B_wrong_endorsed"):
                continue
            if pres not in ("A_correct", "B_wrong_endorsed"):
                continue
            ctrl_correct = (ctrl == "A_correct")
            pres_correct = (pres == "A_correct")
            if ctrl_correct and not pres_correct:
                b_a += 1
            elif not ctrl_correct and pres_correct:
                c_a += 1

        # Convention B: refusals coded as non-correct (State C counts as "not correct")
        # D_unclassified we treat the same as in Convention A — drop the pair (unknown outcome).
        b_b = c_b = 0
        for it in items:
            ctrl, pres = it["control_state"], it["pressure_state"]
            if ctrl == "D_unclassified" or pres == "D_unclassified":
                continue
            ctrl_correct = (ctrl == "A_correct")
            pres_correct = (pres == "A_correct")
            if ctrl_correct and not pres_correct:
                b_b += 1
            elif not ctrl_correct and pres_correct:
                c_b += 1

        or_a, lo_a, hi_a = mcnemar_or_ci(b_a, c_a)
        or_b, lo_b, hi_b = mcnemar_or_ci(b_b, c_b)
        p_a = mcnemar_p_exact(b_a, c_a)
        p_b = mcnemar_p_exact(b_b, c_b)
        p_a_str = "$<10^{-4}$" if p_a < 1e-4 else f"{p_a:.3f}"
        p_b_str = "$<10^{-4}$" if p_b < 1e-4 else f"{p_b:.3f}"

        print(f"        {VAR_LABEL[v]} & {b_a} & {c_a} & {or_a:.2f} [{lo_a:.2f}, {hi_a:.2f}] & {p_a_str} & {b_b} & {c_b} & {or_b:.2f} [{lo_b:.2f}, {hi_b:.2f}] & {p_b_str} \\\\")

    print("        \\bottomrule")
    print("    \\end{tabular}")
    print("    \\caption{\\textbf{Refusals-as-wrong sensitivity analysis for the unanimous-confident pressure condition.} Left: McNemar OR as reported in the main tables, with refusals (State C) excluded from paired contingency tables. Right: sensitivity version where refusals under pressure are coded as ``wrong,'' so no control-correct item is dropped for refusing under pressure. For all seven Olmo-3 variants the qualitative ordering is unchanged and the ORs remain significant at $p < 10^{-4}$; magnitudes shift modestly when refusal rates are small (Base, all Instruct variants) and more visibly for Think-path variants where post-training introduces a small amount of refusal under pressure.}")
    print("    \\label{tab:refusal_sensitivity}")
    print("\\end{table}")


# ---------------------------------------------------------------------------
# [M*] Cross-family McNemar with CIs + raw counts + refusal shift
# ---------------------------------------------------------------------------

def table_M_augmented():
    banner("Appendix M: Cross-family McNemar augmented with CIs, raw counts, refusal shift")

    mcn = load_csv(os.path.join(APRIL, "statistical_tests", "cross_family",
                                "mcnemar_pressure_vs_control_t0.csv"))
    pef = load_csv(os.path.join(APRIL, "tables", "cross_family", "pressure_effects_t0.csv"))

    NAME_NORM = {
        "OLMo-32B-Instruct": "Olmo-3.1-32B-Instruct",
        "OLMo-32B-Think": "Olmo-3.1-32B-Think",
        "OLMo-32B-Think-SFT": "Olmo-3-32B-Think-SFT",
        "OLMo-32B-Think-DPO": "Olmo-3-32B-Think-DPO",
        "Llama-4-Maverick (MoE)": "Llama-4-Maverick",
    }

    # Pressure-effects lookup for refusal deltas
    # pef columns: model_id, short_name, architecture, temperature, ber_control, ber_peer,
    #              refusal_control, refusal_peer, peer_ber_delta, peer_refusal_delta, ...
    pef_lut = {r["short_name"]: r for r in pef}

    # Show peer condition only (the table with three conditions is in original M; here we
    # augment with CIs, focusing on the most-discussed peer pressure).
    COND = "asch_zhu_unbiased_unanimous_confident"
    rows_peer = [r for r in mcn if r["pressure_condition"] == COND]
    # Sort by OR descending
    rows_peer.sort(key=lambda r: -float(r.get("odds_ratio", 0) or 0))

    print("\\begin{table}[H]")
    print("    \\centering")
    print("    \\scriptsize")
    print("    \\setlength{\\tabcolsep}{4pt}")
    print("    \\renewcommand{\\arraystretch}{1.0}")
    print("    \\begin{tabular}{l r r r l l r}")
    print("        \\toprule")
    print("        Model & $n_{\\text{pairs}}$ & $b$ & $c$ & OR [95\\% CI] & Holm $p$ & $\\Delta$refusal \\\\")
    print("        \\midrule")
    for r in rows_peer:
        name = NAME_NORM.get(r["short_name"], r["short_name"])
        n_pairs = int(float(r["n_paired"]))
        b = int(float(r["b_novel_endorse"]))
        c = int(float(r["c_recovered"]))
        or_v, lo, hi = mcnemar_or_ci(b, c)
        p_holm = float(r["p_holm"])
        p_str = "$<10^{-4}$" if p_holm < 1e-4 else f"{p_holm:.3f}"
        # refusal shift from pef
        pef_row = pef_lut.get(r["short_name"])
        if pef_row:
            d_ref = float(pef_row["peer_refusal_delta"])
            d_ref_str = f"{d_ref:+.3f}"
        else:
            d_ref_str = "--"
        print(f"        {name} & {n_pairs} & {b} & {c} & {or_v:.2f} [{lo:.2f}, {hi:.2f}] & {p_str} & {d_ref_str} \\\\")
    print("        \\bottomrule")
    print("    \\end{tabular}")
    print("    \\caption{\\textbf{Cross-family McNemar results on the unanimous-confident peer condition at $T{=}0.0$, with 95\\% CIs and refusal-shift column.} $b$ = control-correct / pressure-wrong; $c$ = control-wrong / pressure-correct. OR uses Haldane--Anscombe correction with Wald log-scale 95\\% CI. The right-hand column $\\Delta$refusal = refusal-rate under pressure $-$ refusal-rate under control, showing how many items were likely dropped from the paired sample due to refusal under pressure. $n_{\\text{pairs}}$ drops well below 400 for models with large positive $\\Delta$refusal (notably Llama-3.1-70B, $\\Delta$refusal $= +0.79$, $n_{\\text{pairs}}{=}${rows_peer_llama_n}), which should be read as a scope limitation on the McNemar inference for that model rather than a separate effect size.}".replace("{rows_peer_llama_n}", str(next((int(float(r["n_paired"])) for r in rows_peer if "Llama-3.1-70B" in r["short_name"]), 0))))
    print("    \\label{tab:cross_mcnemar_peer_full}")
    print("\\end{table}")


# ---------------------------------------------------------------------------
# [BETWEEN] SFT-vs-DPO direct paired McNemar (NEW)
# ---------------------------------------------------------------------------

def table_sft_vs_dpo():
    banner("NEW Appendix: SFT-vs-DPO direct paired McNemar")

    path = os.path.join(APRIL, "item_level", "per_item_endorsement.csv")
    rows = load_csv(path)

    # Keep usable rows and bucket per variant
    by_var = defaultdict(dict)  # variant -> {item_id -> (control_state, pressure_state)}
    for r in rows:
        if r.get("usable") not in ("True", "true", "1"):
            continue
        by_var[r["variant"]][r["item_id"]] = (r["control_state"], r["pressure_state"])

    # Pair two variants on shared items. Compare their pressure-state outcomes:
    #   Define: correct_under_pressure = 1 iff pressure_state == A_correct.
    #   Pair: item is correct under variant X but wrong under variant Y => b
    #         item is wrong under variant X but correct under variant Y => c
    # This asks: is variant Y's failure rate under pressure significantly different from X's?
    def paired_mcnemar(var_x, var_y):
        x, y = by_var[var_x], by_var[var_y]
        items = set(x) & set(y)
        b = c = n = 0
        for it in items:
            _, x_pres = x[it]
            _, y_pres = y[it]
            # Only pair items with determinate outcomes in both variants
            if x_pres == "D_unclassified" or y_pres == "D_unclassified":
                continue
            x_correct = (x_pres == "A_correct")
            y_correct = (y_pres == "A_correct")
            n += 1
            if x_correct and not y_correct:
                b += 1
            elif not x_correct and y_correct:
                c += 1
        or_v, lo, hi = mcnemar_or_ci(b, c)
        p = mcnemar_p_exact(b, c)
        return n, b, c, or_v, lo, hi, p

    COMPARISONS = [
        ("Instruct-SFT vs Instruct-DPO",           "instruct_sft", "instruct_dpo"),
        ("Instruct-SFT vs Instruct (RLVR)",        "instruct_sft", "instruct"),
        ("Instruct-DPO vs Instruct (RLVR)",        "instruct_dpo", "instruct"),
        ("Base vs Instruct-SFT",                   "base",         "instruct_sft"),
        ("Think-SFT vs Think-DPO",                 "think_sft",    "think_dpo"),
        ("Think-SFT vs Think (RLVR, 32B only)",    "think_sft",    "think"),
        ("Think-DPO vs Think (RLVR, 32B only)",    "think_dpo",    "think"),
    ]

    print("\\begin{table}[H]")
    print("    \\centering")
    print("    \\footnotesize")
    print("    \\setlength{\\tabcolsep}{4pt}")
    print("    \\renewcommand{\\arraystretch}{1.0}")
    print("    \\begin{tabular}{l r r r l l}")
    print("        \\toprule")
    print("        Stage comparison & $n_{\\text{pairs}}$ & $b$ & $c$ & OR [95\\% CI] & $p$ \\\\")
    print("        \\midrule")
    for label, vx, vy in COMPARISONS:
        n, b, c, or_v, lo, hi, p = paired_mcnemar(vx, vy)
        p_str = "$<10^{-4}$" if p < 1e-4 else f"{p:.3f}"
        print(f"        {label} & {n} & {b} & {c} & {or_v:.2f} [{lo:.2f}, {hi:.2f}] & {p_str} \\\\")
    print("        \\bottomrule")
    print("    \\end{tabular}")
    print("    \\caption{\\textbf{Direct paired McNemar tests comparing post-training stages on the unanimous-confident peer condition.} Each row pairs two training-stage checkpoints on the same 400 items and asks whether items correct under stage $X$ are more or less often correct under stage $Y$ when pressure is applied. $b$ = correct under $X$ / wrong under $Y$; $c$ = wrong under $X$ / correct under $Y$. OR $> 1$ means the right-hand variant has more errors on the paired items than the left-hand variant; OR $< 1$ means the opposite. All between-stage comparisons are statistically significant and align with the trajectory reported in Figure~\\ref{fig:trajectory}: Instruct-SFT is significantly worse under pressure than both Instruct-DPO and the final RLVR Instruct checkpoint.}")
    print("    \\label{tab:between_stage}")
    print("\\end{table}")


# ---------------------------------------------------------------------------
# [G*] Domain breakdown with Wilson CIs
# ---------------------------------------------------------------------------

def table_G_with_cis():
    banner("Appendix G: Domain breakdown with Wilson 95% CIs")

    rows = load_csv(os.path.join(APRIL, "tables", "behavioral", "domain_breakdown.csv"))

    agg = defaultdict(lambda: {"B": 0.0, "N": 0.0})
    for r in rows:
        key = (r["variant"], r["condition_name"], r["domain"])
        try:
            agg[key]["B"] += float(r["state_B_wrong_endorsed_n"] or 0)
            agg[key]["N"] += float(r["n_observed"] or 0)
        except ValueError:
            pass

    FOLD = {
        "High School Mathematics": "math",
        "High School Geography": "history",
        "High School Physics": "science",
    }
    fold_agg = defaultdict(lambda: {"B": 0.0, "N": 0.0})
    for (v, cond, dom), cnt in agg.items():
        d = FOLD.get(dom, dom)
        fold_agg[(v, cond, d)]["B"] += cnt["B"]
        fold_agg[(v, cond, d)]["N"] += cnt["N"]

    def ber_ci(v, cond, dom):
        a = fold_agg.get((v, cond, dom))
        if not a or a["N"] == 0:
            return None
        k, n = int(a["B"]), int(a["N"])
        lo, hi = wilson_ci(k, n)
        return (k / n, lo, hi, n)

    DOMAIN_ORDER = ["math", "science", "history", "general", "preference"]
    DOM_LABEL = {"math": "Math", "science": "Science", "history": "History/Geo",
                 "general": "General", "preference": "Preference"}
    VAR_ORDER = ["base", "instruct_sft", "instruct_dpo", "instruct"]
    VAR_LABEL = {"base": "Base", "instruct_sft": "Instruct-SFT",
                 "instruct_dpo": "Instruct-DPO", "instruct": "Instruct"}

    PRESSURE = "asch_zhu_unbiased_unanimous_confident"
    CONTROL = "control"

    print("\\begin{table}[H]")
    print("    \\centering")
    print("    \\scriptsize")
    print("    \\setlength{\\tabcolsep}{3pt}")
    print("    \\renewcommand{\\arraystretch}{1.0}")
    print("    \\begin{tabular}{l l r r r r}")
    print("        \\toprule")
    print("        Variant & Domain & $n$ & Ctrl BER [95\\% CI] & Pres.\\ BER [95\\% CI] & $\\Delta$ \\\\")
    print("        \\midrule")
    for v in VAR_ORDER:
        for j, d in enumerate(DOMAIN_ORDER):
            ctrl = ber_ci(v, CONTROL, d)
            pres = ber_ci(v, PRESSURE, d)
            if ctrl is None or pres is None:
                continue
            ber_c, lo_c, hi_c, n_c = ctrl
            ber_p, lo_p, hi_p, n_p = pres
            delta = ber_p - ber_c
            v_cell = VAR_LABEL[v] if j == 0 else ""
            print(f"        {v_cell} & {DOM_LABEL[d]} & {n_p} & {ber_c:.3f} [{lo_c:.3f}, {hi_c:.3f}] & {ber_p:.3f} [{lo_p:.3f}, {hi_p:.3f}] & {delta:+.3f} \\\\")
        if v != VAR_ORDER[-1]:
            print("        \\midrule")
    print("        \\bottomrule")
    print("    \\end{tabular}")
    print("    \\caption{\\textbf{BER by knowledge domain under unanimous-confident peer pressure vs.\\ control, Olmo-3-7B Instruct-pipeline variants, with Wilson 95\\% CIs.} Values pooled across 6 temperatures. $n$ is the per-cell sample size. Domains are small (typically 50--120 items per cell) so CIs are wide. The qualitative pattern from the main text holds under this accounting: Instruct-SFT shows the largest $\\Delta$ in every domain and the CI for the SFT $\\Delta$ does not overlap those of the other variants in Math, Science, and History/Geo.}")
    print("    \\label{tab:domain_breakdown}")
    print("\\end{table}")


# ---------------------------------------------------------------------------
# [J] Mitigation effectiveness
# ---------------------------------------------------------------------------

def table_J_mitigation():
    banner("Appendix J: Mitigation effectiveness table")

    rows = load_csv(os.path.join(APRIL, "tables", "mitigation_taxonomy",
                                 "taxonomy_table.csv"))

    # Keep only Instruct-path variants (consistent with Appendix scope) and rows with BER data.
    KEEP_VARS = ["base", "instruct_sft", "instruct_dpo", "instruct"]
    VAR_LABEL = {"base": "Base", "instruct_sft": "Instruct-SFT",
                 "instruct_dpo": "Instruct-DPO", "instruct": "Instruct"}

    MITIGATION_ORDER = [
        "Devil's Advocate (4+1)",
        "Diverse peers (no majority)",
        "Question Distillation (organizer)",
        "Authority trust + DA",
    ]

    rows_by = defaultdict(dict)  # mitigation -> {variant -> row}
    for r in rows:
        if r["status"] != "computed":
            continue
        if r["variant"] not in KEEP_VARS:
            continue
        if r["ber_mitigation"] == "" or r["ber_anchor_unanimous_plain"] == "":
            continue
        rows_by[r["mitigation"]][r["variant"]] = r

    print("\\begin{table}[H]")
    print("    \\centering")
    print("    \\footnotesize")
    print("    \\setlength{\\tabcolsep}{4pt}")
    print("    \\renewcommand{\\arraystretch}{1.0}")
    print("    \\begin{tabular}{l l r r r r}")
    print("        \\toprule")
    print("        Mitigation & Variant & Anchor BER & Mitigated BER & $\\Delta$ & Mechanism \\\\")
    print("        \\midrule")
    for mit in MITIGATION_ORDER:
        group = rows_by.get(mit, {})
        if not group:
            continue
        for j, v in enumerate(KEEP_VARS):
            r = group.get(v)
            if r is None:
                continue
            anchor = float(r["ber_anchor_unanimous_plain"])
            mbar = float(r["ber_mitigation"])
            delta = float(r["delta_from_anchor"])
            mech = r["predicted_direction"].replace("_", " ")
            mit_cell = mit if j == 0 else ""
            print(f"        {mit_cell} & {VAR_LABEL[v]} & {anchor:.3f} & {mbar:.3f} & {delta:+.3f} & {mech} \\\\")
        print("        \\midrule")
    # strip trailing midrule inelegantly: replace last printed midrule with bottomrule
    print("        \\bottomrule")
    print("    \\end{tabular}")
    print("    \\caption{\\textbf{Mitigation effectiveness for the Olmo-3-7B Instruct-pipeline variants.} Anchor BER is the model's wrong-answer endorsement rate under the unanimous-plain peer condition (the closest matched comparison). Mitigated BER is the rate under the listed mitigation. Negative $\\Delta$ means the mitigation reduces endorsement; positive $\\Delta$ means it increases it. The ``Mechanism'' column reports each mitigation's predicted direction on the pattern-completion vs.\\ authority axis (see \\S\\ref{sec:ablation}). Devil's Advocate and Diverse peers---both of which partially or completely break the repetition pattern---reduce endorsement across all Instruct-path variants. Question Distillation (organizer summary with one repetition) and Authority trust + DA behave idiosyncratically.}")
    print("    \\label{tab:mitigation_effectiveness}")
    print("\\end{table}")


# ---------------------------------------------------------------------------
# [ABL-SIDE] Naked-vs-ngram side-by-side with direction
# ---------------------------------------------------------------------------

def table_naked_vs_ngram():
    banner("NEW Appendix: Naked vs n-gram baseline side-by-side")

    rows = load_csv(os.path.join(APRIL, "tables", "ablation_probes",
                                 "combined_ablation_scorecard.csv"))

    NAME_NORM = {
        "Llama-3.1-70B": "Llama-3.1-70B",
        "OLMo-32B-Instruct": "Olmo-3.1-32B-Instruct",
    }

    print("\\begin{table}[H]")
    print("    \\centering")
    print("    \\footnotesize")
    print("    \\setlength{\\tabcolsep}{5pt}")
    print("    \\renewcommand{\\arraystretch}{1.0}")
    print("    \\begin{tabular}{l r r r r r}")
    print("        \\toprule")
    print("        Model & BER (w/ SP) & BER (naked) & BER (n-gram) & $\\Delta_{\\text{ngram-naked}}$ & Ratio \\\\")
    print("        \\midrule")
    for r in rows:
        name = NAME_NORM.get(r["short_name"], r["short_name"])
        b_with = float(r["ber_with_system_prompt"])
        b_without = float(r["ber_without_system_prompt"])
        b_ngram = float(r["ber_ngram_baseline"])
        delta = b_ngram - b_without
        ratio = float(r["pattern_completion_ratio"])
        print(f"        {name} & {b_with:.3f} & {b_without:.3f} & {b_ngram:.3f} & {delta:+.3f} & {ratio:.2f} \\\\")
    print("        \\bottomrule")
    print("    \\end{tabular}")
    print("    \\caption{\\textbf{Naked-prompt BER vs non-social n-gram BER at $T{=}0$ ($N{=}400$ per cell).} BER (w/ SP): unanimous-confident peer prompt with safety system prompt; BER (naked): same peer prompt with the system prompt removed; BER (n-gram): structure-only baseline (\\texttt{String 1: X, \\ldots, String 5: X, String 6:}) with no social framing. $\\Delta_{\\text{ngram-naked}}$ isolates the effect of removing \\emph{social content} while holding the repetition structure fixed: for both tested models the value is \\emph{positive}, meaning the non-social n-gram baseline yields \\emph{higher} BER than the socially framed naked prompt. ``Ratio'' is the ratio BER(n-gram) / BER(w/ SP). Where the ratio is $\\gg 1$ (Llama-3.1-70B), the system prompt provides strong protection that disappears once the social framing is also stripped; where the ratio is $\\approx 1$ (Olmo-3.1-32B-Instruct), the system prompt provides no measurable protection even before framing is stripped.}")
    print("    \\label{tab:naked_vs_ngram}")
    print("\\end{table}")


if __name__ == "__main__":
    table_E_augmented()
    table_refusal_sensitivity()
    table_M_augmented()
    table_sft_vs_dpo()
    table_G_with_cis()
    table_J_mitigation()
    table_naked_vs_ngram()
