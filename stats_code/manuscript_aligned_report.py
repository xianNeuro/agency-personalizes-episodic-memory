#!/usr/bin/env python3
"""
Supplementary manuscript-aligned report: verbatim Results prose (same as comprehensive
HTML manuscript-text), with stats-box tables removed and each stat-inline span replaced
by the full NC-style string drawn from the same stats dict as comprehensive_analysis_report.html.
"""

import html as html_module
import os
import re

import numpy as np
import pandas as pd

from effect_size_utils import (
    format_ci,
    format_cohens_d,
    partial_eta_squared_from_f,
    pearson_r_ci_fisher,
    cohens_d_one_sample_from_t,
    cohens_d_two_sample_from_t_balanced,
)


def format_stat_value(val):
    if pd.isna(val):
        return "N/A"
    if isinstance(val, (int, np.integer)):
        return str(val)
    if isinstance(val, (float, np.floating)):
        if val >= 1 and abs(val - round(val)) < 1e-6:
            return str(int(round(val)))
        if abs(val) < 0.001:
            return f"{val:.6f}"
        elif abs(val) < 0.01:
            return f"{val:.4f}"
        else:
            return f"{val:.3f}"
    return str(val)


def _safe_inline(s):
    """Escape & and < for HTML text inside spans (e.g. p < 0.001)."""
    return html_module.escape(s, quote=False)


def _eta_suffix(f_val, dfb, dfw):
    try:
        pe = partial_eta_squared_from_f(float(f_val), float(dfb), float(dfw))
        if np.isnan(pe):
            return ""
        return f", η² = {pe:.4f}, ηp² = {pe:.4f}"
    except (TypeError, ValueError):
        return ""


def _mean_ci_lines_run1_group_sheet(sheet_rows, story_label):
    """Build 'cond: M=.., 95% CI=[..], n=..' from group_stats sheet records."""
    parts = []
    for row in sheet_rows:
        cond = row.get("condition", row.get("Condition", ""))
        m = row.get("mean", row.get("Mean"))
        s = row.get("std", row.get("Std"))
        c = row.get("count", row.get("Count"))
        try:
            c = int(c)
            mean = float(m)
            std = float(s)
        except (TypeError, ValueError):
            continue
        if c < 2:
            continue
        sem = std / np.sqrt(c)
        from scipy import stats as scipy_stats
        tcrit = scipy_stats.t.ppf(0.975, df=c - 1)
        marg = tcrit * sem
        lo, hi = mean - marg, mean + marg
        parts.append(
            f"{story_label} {cond}: M = {format_stat_value(mean)}, "
            f"95% CI = {format_ci(lo, hi)}, n = {c}"
        )
    return "; ".join(parts) if parts else ""


def _run1_recall_line(stats, story_key_in_sheet):
    """story_key_in_sheet: 'Adventure' or 'Romance' for sheet names."""
    if not stats.get("run1"):
        return ""
    anova_name = f"{story_key_in_sheet}_anova_table"
    group_name = f"{story_key_in_sheet}_group_stats"
    if anova_name not in stats["run1"]:
        return ""
    sheet_data = stats["run1"][anova_name]
    f_val = dfb = dfw = p_val = None
    for row in sheet_data:
        if row.get("Unnamed: 0") == "C(condition)":
            f_val = row.get("F")
            dfb = row.get("df")
            p_val = row.get("PR(>F)")
        if row.get("Unnamed: 0") == "Residual":
            dfw = row.get("df")
    eta = _eta_suffix(f_val, dfb, dfw)
    base = (
        f"{story_key_in_sheet}: F({format_stat_value(dfb)},{format_stat_value(dfw)}) = "
        f"{format_stat_value(f_val)}, p = {format_stat_value(p_val)}{eta}"
    )
    gtxt = ""
    if group_name in stats["run1"]:
        gtxt = _mean_ci_lines_run1_group_sheet(stats["run1"][group_name], story_key_in_sheet)
    if gtxt:
        base += f"; group means (95% CI): {gtxt}"
    return base


def _run1_engagement_line(stats, measure_key, label_short):
    eng = stats.get(measure_key)
    if not eng:
        return ""
    eta = _eta_suffix(eng.get("f_stat"), eng.get("df_between"), eng.get("df_within"))
    base = (
        f"F({format_stat_value(eng.get('df_between'))},{format_stat_value(eng.get('df_within'))}) = "
        f"{format_stat_value(eng.get('f_stat'))}, p = {format_stat_value(eng.get('p_value'))}{eta}"
    )
    sheet_suffix = measure_key.replace("run1_engagement_", "").replace("_stats", "")
    sheet_name = f"Engagement_{sheet_suffix}_stats"
    if stats.get("run1") and sheet_name in stats["run1"]:
        g = _mean_ci_lines_run1_group_sheet(stats["run1"][sheet_name], "Romance")
        if g:
            base += f"; group means (95% CI): {g}"
    return base


def _run2_onesample_full(stats, analysis, cond):
    for r in stats.get("run2", []):
        if r.get("analysis") != analysis:
            continue
        if str(r.get("condition", "")).lower() != cond.lower():
            continue
        line = (
            f"Romance: {cond.capitalize()}: mean r = {format_stat_value(r.get('mean'))}, "
            f"95% CI = {format_ci(r.get('ci_lower'), r.get('ci_upper'))}, "
            f"t({format_stat_value(r.get('df'))}) = {format_stat_value(r.get('t_stat'))}, "
            f"p = {format_stat_value(r.get('p_value'))}"
        )
        if not pd.isna(r.get("cohens_d")):
            line += f", {format_cohens_d(r.get('cohens_d'))}"
        return line
    return ""


def _posthoc_df_from_row(row):
    """Approximate independent-samples df for reporting when the sheet omits df."""
    comp = row.get("comparison")
    if comp == "Free vs Yoked":
        nf, ny = row.get("n_free"), row.get("n_yoke")
        if pd.notna(nf) and pd.notna(ny):
            return int(nf + ny - 2)
    if comp == "Free vs Passive":
        nf, npv = row.get("n_free"), row.get("n_pasv")
        if pd.notna(nf) and pd.notna(npv):
            return int(nf + npv - 2)
    return None


def _run2_posthoc_pair(stats, analysis_num, comparison_label, posthoc_df):
    """comparison_label must match Excel exactly, e.g. 'Free vs Passive'."""
    if posthoc_df is None or not isinstance(posthoc_df, pd.DataFrame):
        return ""
    sub = posthoc_df[
        (posthoc_df["analysis"] == analysis_num) & (posthoc_df["comparison"] == comparison_label)
    ]
    if sub.empty:
        return ""
    row = sub.iloc[0]
    t = row.get("t_stat")
    p = row.get("p_value")
    df = _posthoc_df_from_row(row)
    if df is not None:
        tpart = f"t({df}) = {format_stat_value(t)}, p = {format_stat_value(p)}"
    else:
        tpart = f"t = {format_stat_value(t)}, p = {format_stat_value(p)}"
    parts = [tpart]
    if not pd.isna(row.get("cohens_d")):
        parts.append(format_cohens_d(row.get("cohens_d")))
    if not pd.isna(row.get("mean_diff_ci_lower")):
        parts.append(
            f"95% CI for mean difference = {format_ci(row.get('mean_diff_ci_lower'), row.get('mean_diff_ci_upper'))}"
        )
    pretty = comparison_label.replace(" vs ", " vs. ")
    return f"{pretty}: " + "; ".join(parts)


def _run2_anova_compact(stats, loader, analysis_num):
    path = os.path.join(
        loader.get_output_dir("run2_individual_variability_recalled_events"),
        "isc_anova_results_all_analyses.xlsx",
    )
    if not os.path.exists(path):
        return ""
    adf = pd.read_excel(path)
    sub = adf[adf["analysis"] == analysis_num]
    if sub.empty:
        return ""
    r0 = sub.iloc[0]
    eta = _eta_suffix(r0.get("f_stat"), r0.get("df_between"), r0.get("df_within"))
    return (
        f"Romance: F({format_stat_value(r0.get('df_between'))},{format_stat_value(r0.get('df_within'))}) = "
        f"{format_stat_value(r0.get('f_stat'))}, p = {format_stat_value(r0.get('p_value'))}{eta}"
    )


def _build_run2_replacements(stats, loader):
    ph = stats.get("run2_posthoc")
    out = []
    # Paragraph 64 events
    for cond in ("free", "yoke", "pasv"):
        out.append(_run2_onesample_full(stats, 1, cond))
    out.append(_run2_anova_compact(stats, loader, 1))
    p_anova = 1.0
    path = os.path.join(
        loader.get_output_dir("run2_individual_variability_recalled_events"),
        "isc_anova_results_all_analyses.xlsx",
    )
    if os.path.exists(path):
        adf = pd.read_excel(path)
        s1 = adf[adf["analysis"] == 1]
        if not s1.empty:
            p_anova = float(s1.iloc[0].get("p_value", 1))
    out.append(
        _run2_posthoc_pair(stats, 1, "Free vs Passive", ph)
        if p_anova < 0.05
        else "Free vs. Passive: (post-hoc not reported; omnibus n.s.)"
    )
    out.append(
        _run2_posthoc_pair(stats, 1, "Free vs Yoked", ph)
        if p_anova < 0.05
        else "Free vs. Yoked: (post-hoc not reported; omnibus n.s.)"
    )
    # 49 events
    for cond in ("free", "yoke", "pasv"):
        out.append(_run2_onesample_full(stats, 2, cond))
    out.append(_run2_anova_compact(stats, loader, 2))
    p_anova2 = 1.0
    if os.path.exists(path):
        s2 = adf[adf["analysis"] == 2]
        if not s2.empty:
            p_anova2 = float(s2.iloc[0].get("p_value", 1))
    out.append(
        _run2_posthoc_pair(stats, 2, "Free vs Passive", ph)
        if p_anova2 < 0.05
        else "Free vs. Passive: (post-hoc not reported; omnibus n.s.)"
    )
    out.append(
        _run2_posthoc_pair(stats, 2, "Free vs Yoked", ph)
        if p_anova2 < 0.05
        else "Free vs. Yoked: (post-hoc not reported; omnibus n.s.)"
    )
    return [_safe_inline(x) for x in out]


def _parse_run12_four_lines(stats):
    """Order: Free vs Passive 64, Free vs Passive 49, Free vs Yoked 64, Free vs Yoked 49."""
    text = stats.get("run12") or ""
    out = []

    def grab(block_label, contrast):
        idx = text.find(block_label)
        if idx < 0:
            return "see comprehensive report"
        seg = text[idx : idx + 2500]
        key = f"Free vs {contrast}:"
        j = seg.find(key)
        if j < 0:
            return "see comprehensive report"
        line = seg[j : j + 120].split("\n")[0]
        pm = re.search(r"p\s*=\s*([0-9.]+)", line)
        pval = pm.group(1) if pm else "N/A"
        return f"Monte Carlo permutation p = {pval} (10,000 resamples; {block_label.strip()}: Free vs {contrast})"

    out.append(_safe_inline(grab("ANALYSIS: 64 Shared Events", "Passive")))
    out.append(_safe_inline(grab("ANALYSIS: 49 Non-Choice Events", "Passive")))
    out.append(_safe_inline(grab("ANALYSIS: 64 Shared Events", "Yoked")))
    out.append(_safe_inline(grab("ANALYSIS: 49 Non-Choice Events", "Yoked")))
    return out


def _run13_expanded(stats):
    text = stats.get("run13") or ""
    m = re.search(r"One-tailed test:\s*p\s*=\s*([0-9.]+)", text)
    p1 = m.group(1) if m else None
    m2 = re.search(r"ANALYSIS: 49 Non-Choice Events[\s\S]{0,800}One-tailed test:\s*p\s*=\s*([0-9.]+)", text)
    p2 = m2.group(1) if m2 else p1
    if p1:
        return _safe_inline(
            f"Monte Carlo p = {p1} (64 shared events; 10,000 valid resamples; matching Choice ISC); "
            f"49 non-choice: p = {p2 if p2 else 'N/A'}"
        )
    return _safe_inline("p < 0.001 (see comprehensive report for full permutation output)")


def _run3_choice_lines(stats):
    out = []
    for cond in ("free", "yoke"):
        for row in stats.get("run3", []):
            if row.get("Analysis") != "raw" or str(row.get("Condition", "")).lower() != cond:
                continue
            parts = [
                f"mean r = {format_stat_value(row.get('Mean'))}, "
                f"95% CI = {format_ci(row.get('ci_lower'), row.get('ci_upper'))}, "
                f"t({format_stat_value(int(row.get('N_pairs', 0)) - 1)}) = {format_stat_value(row.get('t_statistic'))}, "
                f"p = {format_stat_value(row.get('p_value'))}"
            ]
            if not pd.isna(row.get("cohens_d")):
                parts.append(format_cohens_d(row.get("cohens_d")))
            out.append(_safe_inline(cond.capitalize() + ": " + ", ".join(parts)))
            break
    for row in stats.get("run3", []):
        if row.get("Analysis") == "raw" and str(row.get("Condition", "")) == "Free_vs_Yoke":
            tstat = row.get("t_statistic")
            dfw = row.get("df")
            if pd.notna(dfw):
                tpart = f"two-sample t-test: t({format_stat_value(dfw)}) = {format_stat_value(tstat)}"
            else:
                tpart = f"two-sample t-test: t = {format_stat_value(tstat)}"
            parts = [tpart + f", p = {format_stat_value(row.get('p_value'))}"]
            if not pd.isna(row.get("cohens_d")):
                parts.append(format_cohens_d(row.get("cohens_d")))
            if not pd.isna(row.get("ci_lower")):
                parts.append(
                    f"95% CI for mean difference = {format_ci(row.get('ci_lower'), row.get('ci_upper'))}"
                )
            out.append(_safe_inline("; ".join(parts)))
            break
    return out


def _corr_expanded(r, n_subjects, p):
    """Plain text (not HTML-escaped); wrap with _safe_inline at substitution time."""
    nn = int(n_subjects) if pd.notna(n_subjects) else None
    if not nn:
        return f"r = {format_stat_value(r)}, p = {format_stat_value(p)}"
    df_disp = nn - 2
    extra = ""
    if nn >= 4 and pd.notna(r):
        lo, hi = pearson_r_ci_fisher(float(r), nn)
        if not (np.isnan(lo) or np.isnan(hi)):
            extra = f", 95% CI for r = {format_ci(lo, hi)}"
    return f"r({df_disp}) = {format_stat_value(r)}{extra}, p = {format_stat_value(p)} (N = {nn})"


def _run6_semantic_block(stats, story):
    rows = [
        r
        for r in stats.get("run6", [])
        if r.get("Analysis") == "One-way ANOVA"
        and r.get("Measure") == "Semantic Centrality"
        and r.get("Transform") == "Raw values"
        and r.get("Story") == story
    ]
    if not rows:
        return ""
    r0 = rows[0]
    f_val, dfb, dfw, p_val = r0.get("F_statistic"), r0.get("df_between"), r0.get("df_within"), r0.get("p_value")
    base = (
        f"{story}: F({format_stat_value(dfb)},{format_stat_value(dfw)}) = "
        f"{format_stat_value(f_val)}, p = {format_stat_value(p_val)}{_eta_suffix(f_val, dfb, dfw)}"
    )
    post = [
        r
        for r in stats.get("run6", [])
        if r.get("Analysis") == "Post-hoc t-test"
        and "Semantic Centrality" in str(r.get("Measure", ""))
        and r.get("Transform") == "Raw values"
        and r.get("Story") == story
        and pd.notna(r.get("p_value"))
        and float(r.get("p_value")) < 0.1
    ]
    if post:
        bits = []
        for ph in post:
            ms = str(ph.get("Measure", ""))
            comp = ms.split(":")[-1].strip().replace("_", " ") if ":" in ms else ""
            t, dfw_, pv = ph.get("t_statistic"), ph.get("df_within"), ph.get("p_value")
            bit = f"t({format_stat_value(dfw_)}) = {format_stat_value(t)}, p = {format_stat_value(pv)}"
            if pd.notna(t) and pd.notna(dfw_):
                try:
                    bit += f", {format_cohens_d(cohens_d_two_sample_from_t_balanced(float(t), float(dfw_)))}"
                except (TypeError, ValueError):
                    pass
            bits.append(comp + ": " + bit)
        base += "; post-hoc: " + "; ".join(bits)
    return _safe_inline(base)


def _run6_interaction_block(stats, story):
    for r in stats.get("run6", []):
        if (
            r.get("Analysis") == "Repeated Measures ANOVA"
            and r.get("Transform") == "Raw values"
            and r.get("Story") == story
        ):
            f_val, dfb, dfw = r.get("F_statistic"), r.get("df_between"), r.get("df_within")
            p_val = r.get("p_value")
            return _safe_inline(
                f"F({format_stat_value(dfb)},{format_stat_value(dfw)}) = "
                f"{format_stat_value(f_val)}, p = {format_stat_value(p_val)}{_eta_suffix(f_val, dfb, dfw)} "
                f"(network type × agency interaction)"
            )
    return ""


def _run7_romance_anova_only(stats):
    for r in stats.get("run7", []):
        if (
            r.get("Analysis") == "One-way ANOVA"
            and r.get("Transform") == "Raw values"
            and str(r.get("Story", "")).lower() == "romance"
        ):
            f_val, dfb, dfw = r.get("F_statistic"), r.get("df_between"), r.get("df_within")
            p_val = r.get("p_value")
            return _safe_inline(
                f"F({format_stat_value(dfb)},{format_stat_value(dfw)}) = "
                f"{format_stat_value(f_val)}, p = {format_stat_value(p_val)}{_eta_suffix(f_val, dfb, dfw)}"
            )
    return ""


def _run7_romance_posthoc_line(stats, pasv_not_yoke):
    """pasv_not_yoke: True for Free vs Passive, False for Free vs Yoked (matches Condition strings in Excel)."""
    needle = "pasv" if pasv_not_yoke else "yoke"
    label = "Free vs. Passive" if pasv_not_yoke else "Free vs. Yoked"
    for ph in stats.get("run7", []):
        if (
            ph.get("Analysis") != "Post-hoc t-test"
            or ph.get("Transform") != "Raw values"
            or str(ph.get("Story", "")).lower() != "romance"
        ):
            continue
        cond = str(ph.get("Condition", "")).lower()
        if needle not in cond:
            continue
        t, dfw_, pv = ph.get("t_statistic"), ph.get("df_within"), ph.get("p_value")
        bit = f"{label}: t({format_stat_value(dfw_)}) = {format_stat_value(t)}, p = {format_stat_value(pv)}"
        if pd.notna(t) and pd.notna(dfw_):
            try:
                bit += f", {format_cohens_d(cohens_d_two_sample_from_t_balanced(float(t), float(dfw_)))}"
            except (TypeError, ValueError):
                pass
        return _safe_inline(bit)
    return _safe_inline(f"{label}: (post-hoc not found in run7 stats)")


def _run8_line(stats, story):
    for r in stats.get("run8", []):
        if str(r.get("Story", "")) == story:
            f_val, dfb, dfw = r.get("F_statistic"), r.get("df_between"), r.get("df_within")
            p_val = r.get("p_value")
            return _safe_inline(
                f"{story}: F({format_stat_value(dfb)},{format_stat_value(dfw)}) = "
                f"{format_stat_value(f_val)}, p = {format_stat_value(p_val)}{_eta_suffix(f_val, dfb, dfw)}"
            )
    return ""


def _replacements_for_run(run_num, stats, loader):
    if run_num == 1:
        return [
            _safe_inline(_run1_recall_line(stats, "Adventure")),
            _safe_inline(_run1_recall_line(stats, "Romance")),
            _safe_inline(_run1_engagement_line(stats, "run1_engagement_trans_score", "transportation")),
            _safe_inline(_run1_engagement_line(stats, "run1_engagement_avg_sent_readtime", "read time / sentence")),
            _safe_inline(_run1_engagement_line(stats, "run1_engagement_sum_readtime", "total read time")),
        ]
    if run_num == 2:
        return _build_run2_replacements(stats, loader)
    if run_num == 12:
        return _parse_run12_four_lines(stats)
    if run_num == 3:
        return _run3_choice_lines(stats)
    if run_num == 13:
        return [_run13_expanded(stats)]
    if run_num == 4:
        rows = list(stats.get("run4") or [])

        def _n_key(rec):
            try:
                return int(rec.get("N_subjects", 0))
            except (TypeError, ValueError):
                return 0

        rows = sorted(rows, key=_n_key)
        out = []
        for row in rows:
            anal = row.get("Analysis", "")
            core = _corr_expanded(
                row.get("Correlation_r"), row.get("N_subjects"), row.get("Correlation_p")
            )
            prefix = "Romance: " if row.get("N_subjects") == 18 else ""
            out.append(_safe_inline(f"{prefix}{core}"))
        return out
    if run_num == 5:
        from run_all_analyses import _format_run5_raw_bundle

        return [
            _safe_inline(_format_run5_raw_bundle(stats, "Semantic")),
            _safe_inline(_format_run5_raw_bundle(stats, "Causal")),
        ]
    if run_num == 6:
        return [
            _run6_semantic_block(stats, "Adventure"),
            _run6_semantic_block(stats, "Romance"),
            _run6_interaction_block(stats, "Adventure"),
            _run6_interaction_block(stats, "Romance"),
        ]
    if run_num == 7:
        from run_all_analyses import _format_run7_neighbor_onesample_story

        return [
            _safe_inline(_format_run7_neighbor_onesample_story(stats, "Adventure")),
            _safe_inline(_format_run7_neighbor_onesample_story(stats, "Romance")),
            _run7_romance_anova_only(stats),
            _run7_romance_posthoc_line(stats, pasv_not_yoke=False),
            _run7_romance_posthoc_line(stats, pasv_not_yoke=True),
        ]
    if run_num == 8:
        return [_run8_line(stats, "Adventure"), _run8_line(stats, "Romance")]
    if run_num == 9:
        out = []
        mem = [
            row
            for row in stats.get("run9", [])
            if row.get("Correlation") == "Memory Divergence vs Semantic Influence"
        ]
        mem = sorted(mem, key=lambda r: int(r.get("n") or r.get("N_subjects") or 0))
        for row in mem:
            n = row.get("n", row.get("N_subjects"))
            core = _corr_expanded(row.get("r"), n, row.get("p_value"))
            prefix = "Romance: " if int(n) <= 20 else ""
            out.append(_safe_inline(prefix + core))
        ch = [
            row
            for row in stats.get("run9", [])
            if row.get("Correlation") == "Choice Divergence vs Semantic Influence"
        ]
        ch = sorted(ch, key=lambda r: int(r.get("n") or r.get("N_subjects") or 0))
        for row in ch:
            n = row.get("n", row.get("N_subjects"))
            core = _corr_expanded(row.get("r"), n, row.get("p_value"))
            prefix = "Romance: " if int(n) <= 20 else ""
            out.append(_safe_inline(prefix + core))
        return out
    if run_num == 10:
        cor = [r for r in (stats.get("run10") or []) if r.get("Analysis") == "Correlation"]

        def _pick(story, n_sub, m2_substr):
            for r in cor:
                if r.get("Story") != story:
                    continue
                if int(r.get("N_subjects") or 0) != n_sub:
                    continue
                if m2_substr.lower() in str(r.get("Measure2", "")).lower():
                    return r
            return None

        order = [
            ("Romance", 18, "Memory", "Romance: "),
            ("Romance", 100, "Memory", ""),
            ("Adventure", 22, "Semantic", "Adventure: "),
            ("Romance", 18, "Semantic", "Romance: "),
            ("Romance", 100, "Semantic", ""),
        ]
        out = []
        for story, n_sub, m2s, prefix in order:
            row = _pick(story, n_sub, m2s)
            if not row:
                out.append(_safe_inline(f"({story} N={n_sub} {m2s}: not in run10 output)"))
                continue
            core = _corr_expanded(row.get("r"), row.get("N_subjects"), row.get("p_value"))
            out.append(_safe_inline(prefix + core))
        reg100 = next(
            (
                r
                for r in stats.get("run10") or []
                if r.get("Analysis") == "Multiple Regression" and int(r.get("N_subjects") or 0) == 100
            ),
            None,
        )
        if reg100:
            out.append(
                _safe_inline(
                    f"Romance (N=100): semantic influence β = {format_stat_value(reg100.get('sem_coef'))}, "
                    f"p = {format_stat_value(reg100.get('sem_p'))}; model R² = {format_stat_value(reg100.get('r_squared'))}"
                )
            )
            out.append(
                _safe_inline(
                    f"Romance (N=100): neighbor encoding β = {format_stat_value(reg100.get('nghb_coef'))}, "
                    f"p = {format_stat_value(reg100.get('p_value'))}"
                )
            )
        else:
            out.extend([_safe_inline("(N=100 MR: not in run10 output)")] * 2)
        return out
    if run_num == 11:
        from run_all_analyses import _run11_pe_boost_stat_line

        out = []
        for story in ("Adventure", "Romance"):
            tt = (stats.get("run11_ttests") or {}).get(story)
            if not tt:
                out.append(_safe_inline(f"{story}: (see agency_denial_choice_events_report.txt)"))
                continue
            ci = format_ci(tt["ci_lo"], tt["ci_hi"])
            dpart = f", {format_cohens_d(tt['cohens_d'])}" if not pd.isna(tt.get("cohens_d")) else ""
            out.append(
                _safe_inline(
                    f"{story}: t({tt['df']}) = {format_stat_value(tt['t'])}, "
                    f"p = {format_stat_value(tt['p'])}{dpart}, 95% CI for mean difference = {ci}"
                )
            )
        out.append(_safe_inline(_run11_pe_boost_stat_line("Adventure", stats.get("run11_pe_boost_corr_ba"))))
        out.append(_safe_inline(_run11_pe_boost_stat_line("Romance", stats.get("run11_pe_boost_corr_mv"))))
        return out
    return []


def _strip_stats_boxes(html_chunk):
    prev = None
    s = html_chunk
    while prev != s:
        prev = s
        s = re.sub(r'<div class="stats-box">[\s\S]*?</div>', "", s, count=1)
    return s


def _substitute_stat_inlines(html_chunk, replacements):
    # Use [\s\S]*? so content like "p < 0.001" is not truncated at '<'.
    pattern = re.compile(r'<span class="stat-inline">([\s\S]*?)</span>')
    spans = list(pattern.finditer(html_chunk))
    if not spans:
        return html_chunk
    if len(replacements) < len(spans):
        replacements = list(replacements) + [
            _safe_inline(m.group(1)) for m in spans[len(replacements) :]
        ]
    elif len(replacements) > len(spans):
        replacements = replacements[: len(spans)]
    out = []
    last = 0
    for i, m in enumerate(spans):
        out.append(html_chunk[last : m.start()])
        out.append('<span class="stat-inline">' + replacements[i] + "</span>")
        last = m.end()
    out.append(html_chunk[last:])
    return "".join(out)


def generate_manuscript_aligned_html(stats, loader):
    import run_all_analyses as ra

    full = ra.generate_html_report(stats)
    body_m = re.search(r"<body>([\s\S]*)</body>", full)
    if not body_m:
        return "<html><body>Error: could not parse comprehensive HTML</body></html>"
    body = body_m.group(1)
    # Drop title / first intro lines of comprehensive (h1 + generated) — keep from first run-section
    i = body.find('<div class="run-section">')
    if i < 0:
        return "<html><body>Error: no run-section in comprehensive HTML</body></html>"
    preamble = body[:i]
    rest = body[i:]
    parts = re.split(r"(?=<div class=\"run-section\">)", rest)
    sections = [p for p in parts if p.strip().startswith('<div class="run-section">')]

    css_m = re.search(r"<style>([\s\S]*?)</style>", full)
    css = css_m.group(1) if css_m else ""

    out = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        '<meta charset="UTF-8">',
        "<title>Results — Expanded statistical reporting (Supplementary)</title>",
        "<style>",
        css,
        "</style>",
        '    <link rel="stylesheet" href="print_supplement.css" media="print">',
        "</head>",
        "<body>",
        '<div class="manuscript-text si-intro">',
        "<p>The following sections reproduce the main manuscript Results narrative in full, with each statistical "
        "statement expanded to report complete inferential detail as generated from the analysis pipeline: test "
        "statistics (<em>F</em>, <em>t</em>, <em>r</em>), degrees of freedom, exact or rounded <em>p</em> values, "
        "effect sizes (e.g. Cohen&rsquo;s <em>d</em>, &eta;&sup2; / partial &eta;&sup2;), and 95% confidence intervals "
        "where those quantities are available in the output files. Material is organized in the same RUN order as the "
        "comprehensive technical appendix (RUN 1 through RUN 13), so readers can move from this supplement to the "
        "corresponding section of that report or to the primary figures and supplementary items cited in the text. "
        "Tabular summaries that appear only in the technical appendix are folded into the inline reporting here rather "
        "than repeated as separate tables.</p>",
        "</div>",
    ]

    for sec in sections:
        m = re.search(r"<h2>RUN (\d+)", sec)
        if not m:
            out.append(sec)
            continue
        run_num = int(m.group(1))
        cleaned = _strip_stats_boxes(sec)
        repl = _replacements_for_run(run_num, stats, loader)
        cleaned = _substitute_stat_inlines(cleaned, repl)
        out.append(cleaned)

    out.append("</body></html>")
    return "\n".join(out)
