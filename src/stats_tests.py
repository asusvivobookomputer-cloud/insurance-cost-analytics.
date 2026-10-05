"""Hypothesis testing toolkit with automatic test selection.

Decision logic (as in standard statistics courses)
    1. Normality of every group     -> Shapiro-Wilk
    2. Equality of variances        -> Bartlett (normal data) / Levene (non-normal data)
    3. Choose the test:
         2 groups : Student t | Welch t | Mann-Whitney U
         3+ groups: one-way ANOVA | Welch ANOVA | Kruskal-Wallis
    4. If the omnibus test is significant -> post-hoc (Tukey | Conover-Holm)
    5. ALWAYS report an effect size, because with n > 1000 almost everything is "significant".
"""
import numpy as np
import pandas as pd
import pingouin as pg
import scikit_posthocs as sp
from scipy import stats

from src.config import ALPHA


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise pingouin column names ('p-val' in <=0.5.x, 'p_val' in >=0.6) to underscores."""
    return df.rename(columns=lambda c: c.replace("-", "_"))


def compare_groups(df: pd.DataFrame, value: str, group: str, alpha: float = ALPHA) -> dict:
    """Compare ``value`` across the levels of ``group`` with the statistically appropriate test."""
    data = df[[value, group]].dropna().copy()
    data[group] = data[group].astype(str)
    levels = sorted(data[group].unique())
    samples = [data.loc[data[group] == lv, value].to_numpy() for lv in levels]
    k, n = len(levels), len(data)

    normal = all(stats.shapiro(s).pvalue > alpha for s in samples)
    var_test = stats.bartlett if normal else stats.levene
    equal_var = var_test(*samples).pvalue > alpha

    posthoc = None
    if k == 2:
        if normal:
            res = stats.ttest_ind(*samples, equal_var=equal_var)
            test = "Student t-test" if equal_var else "Welch t-test"
            stat, p = res.statistic, res.pvalue
            effect_name, effect = "Cohen d", pg.compute_effsize(samples[0], samples[1], eftype="cohen")
        else:
            r = _norm(pg.mwu(samples[0], samples[1]))
            test, stat, p = "Mann-Whitney U", r["U_val"].iloc[0], r["p_val"].iloc[0]
            effect_name, effect = "rank-biserial r", r["RBC"].iloc[0]
    else:
        if normal and equal_var:
            res = stats.f_oneway(*samples)
            grand = data[value].mean()
            ssb = sum(len(s) * (s.mean() - grand) ** 2 for s in samples)
            sst = ((data[value] - grand) ** 2).sum()
            test, stat, p = "One-way ANOVA", res.statistic, res.pvalue
            effect_name, effect = "eta squared", ssb / sst
            if p < alpha:
                posthoc = pg.pairwise_tukey(data=data, dv=value, between=group)
        elif normal:
            r = _norm(pg.welch_anova(data=data, dv=value, between=group))
            test, stat, p = "Welch ANOVA", r["F"].iloc[0], r["p_unc"].iloc[0]
            effect_name, effect = "eta squared", r["np2"].iloc[0]
            if p < alpha:
                posthoc = pg.pairwise_gameshowell(data=data, dv=value, between=group)
        else:
            r = _norm(pg.kruskal(data=data, dv=value, between=group))
            h, p = r["H"].iloc[0], r["p_unc"].iloc[0]
            test, stat = "Kruskal-Wallis H", h
            effect_name, effect = "epsilon squared", (h - k + 1) / (n - k)
            if p < alpha:
                posthoc = sp.posthoc_conover(data, val_col=value, group_col=group, p_adjust="holm")

    return {
        "comparison": f"{value} by {group}",
        "n_groups": k,
        "all_groups_normal": normal,
        "equal_variances": equal_var,
        "test": test,
        "statistic": round(float(stat), 3),
        "p_value": float(p),
        "effect_size_type": effect_name,
        "effect_size": round(float(effect), 3),
        "significant": bool(p < alpha),
        "posthoc": posthoc,
    }


def chi_square_test(df: pd.DataFrame, a: str, b: str, alpha: float = ALPHA) -> dict:
    """Chi-square test of independence (Fisher exact for sparse 2x2 tables) + Cramer's V."""
    table = pd.crosstab(df[a], df[b])
    chi2, p, _, expected = stats.chi2_contingency(table)
    test = "Chi-square independence"
    if table.shape == (2, 2) and (expected < 5).any():
        _, p = stats.fisher_exact(table)
        test = "Fisher exact"
    cramers_v = np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1)))
    return {
        "comparison": f"{a} vs {b}",
        "test": test,
        "statistic": round(float(chi2), 3),
        "p_value": float(p),
        "effect_size_type": "Cramer's V",
        "effect_size": round(float(cramers_v), 3),
        "significant": bool(p < alpha),
    }


def correlation_report(df: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    """Pearson and Spearman correlation of every feature with the target."""
    rows = []
    for f in features:
        pear = _norm(pg.corr(df[f], df[target], method="pearson")).iloc[0]
        spear = _norm(pg.corr(df[f], df[target], method="spearman")).iloc[0]
        rows.append(
            {"feature": f, "pearson_r": round(pear["r"], 3), "pearson_p": pear["p_val"],
             "spearman_rho": round(spear["r"], 3), "spearman_p": spear["p_val"]}
        )
    return pd.DataFrame(rows).set_index("feature")


def partial_correlation(df: pd.DataFrame, x: str, y: str, covar: str) -> pd.Series:
    """Spearman partial correlation of x and y controlling for ``covar``."""
    r = _norm(pg.partial_corr(data=df, x=x, y=y, covar=covar, method="spearman")).iloc[0]
    return pd.Series({"x": x, "y": y, "controlling_for": covar,
                      "partial_rho": round(r["r"], 3), "p_value": r["p_val"]})
