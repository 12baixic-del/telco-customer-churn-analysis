"""第二阶段：统计检验、置信区间、多重比较校正与多变量交叉分群分析模块。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .config import KEY_CATEGORICAL_VARS, NUMERIC_COLS, TARGET_COL


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """计算二项比例的 Wilson 95% 置信区间（对小样本和极端比例稳健）。"""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR 多重比较校正，返回校正后的 q-value 列表。"""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    q_vals = [0.0] * m
    prev_q = 1.0
    for rank_idx in range(m - 1, -1, -1):
        orig_idx, p = indexed[rank_idx]
        rank = rank_idx + 1
        q = min(prev_q, p * m / rank)
        q_vals[orig_idx] = float(min(1.0, max(0.0, q)))
        prev_q = q_vals[orig_idx]
    return q_vals


def cramers_v(contingency: np.ndarray) -> float:
    """计算列联表的效应量 Cramér's V。"""
    chi2 = stats.chi2_contingency(contingency, correction=False)[0]
    n = contingency.sum()
    if n <= 0:
        return 0.0
    r, k = contingency.shape
    min_dim = min(r - 1, k - 1)
    if min_dim <= 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min_dim)))


def compute_segment_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """计算单变量各分群客户数、流失数、流失率与 Wilson 95% 置信区间。"""
    work = df.copy()
    if col == "SeniorCitizen":
        work[col] = work[col].map({0: "No", 1: "Yes"}).fillna(work[col].astype(str))

    grouped = (
        work.groupby(col, observed=True, dropna=False)
        .agg(customers=(TARGET_COL, "size"), churned=(TARGET_COL, lambda s: int((s == "Yes").sum())))
        .reset_index()
    )
    grouped["churn_rate"] = grouped["churned"] / grouped["customers"]
    ci_bounds = [wilson_ci(int(k), int(n)) for k, n in zip(grouped["churned"], grouped["customers"])]
    grouped["ci_low"] = [b[0] for b in ci_bounds]
    grouped["ci_high"] = [b[1] for b in ci_bounds]
    grouped["share_of_total"] = grouped["customers"] / len(work)
    grouped["small_sample_flag"] = grouped["customers"] < 30
    return grouped.sort_values("churn_rate", ascending=False).reset_index(drop=True)


def run_categorical_tests(df: pd.DataFrame, cols: list[str] | None = None) -> list[dict[str, Any]]:
    """对关键分类变量执行卡方检验、效应量 Cramér's V 与 FDR 多重比较校正。"""
    cols = cols or KEY_CATEGORICAL_VARS
    results = []
    raw_p = []
    for col in cols:
        series = df[col].map({0: "No", 1: "Yes"}) if col == "SeniorCitizen" else df[col].astype(str)
        ct = pd.crosstab(series, df[TARGET_COL])
        chi2, p_val, dof, _ = stats.chi2_contingency(ct)
        v = cramers_v(ct.to_numpy())
        seg_df = compute_segment_table(df, col)
        results.append({
            "feature": col,
            "chi2": float(chi2),
            "dof": int(dof),
            "p_value": float(p_val),
            "cramers_v": float(v),
            "groups": seg_df.to_dict(orient="records"),
        })
        raw_p.append(float(p_val))

    q_vals = benjamini_hochberg(raw_p)
    for row, q in zip(results, q_vals):
        row["fdr_q_value"] = q
        row["significant_fdr_05"] = bool(q < 0.05)
    return sorted(results, key=lambda r: r["cramers_v"], reverse=True)


def run_numeric_tests(df: pd.DataFrame, cols: list[str] | None = None) -> list[dict[str, Any]]:
    """对数值变量按 Churn 分组执行 Mann-Whitney U 检验、Rank-Biserial 效应量与 FDR 校正。"""
    cols = cols or NUMERIC_COLS
    results = []
    raw_p = []
    churn_yes = df[TARGET_COL] == "Yes"

    for col in cols:
        s_yes = df.loc[churn_yes, col].dropna()
        s_no = df.loc[~churn_yes, col].dropna()
        u_stat, p_val = stats.mannwhitneyu(s_yes, s_no, alternative="two-sided")
        n1, n0 = len(s_yes), len(s_no)
        # Rank-Biserial 相关系数（>0 表示流失组倾向于更大值，<0 表示更小值）
        rbc = float(2.0 * u_stat / (n1 * n0) - 1.0) if (n1 > 0 and n0 > 0) else 0.0
        results.append({
            "feature": col,
            "n_churn": int(n1),
            "n_retain": int(n0),
            "churn_mean": float(s_yes.mean()),
            "churn_std": float(s_yes.std()),
            "churn_median": float(s_yes.median()),
            "churn_iqr": float(s_yes.quantile(0.75) - s_yes.quantile(0.25)),
            "retain_mean": float(s_no.mean()),
            "retain_std": float(s_no.std()),
            "retain_median": float(s_no.median()),
            "retain_iqr": float(s_no.quantile(0.75) - s_no.quantile(0.25)),
            "mannwhitney_u": float(u_stat),
            "p_value": float(p_val),
            "rank_biserial_r": rbc,
        })
        raw_p.append(float(p_val))

    q_vals = benjamini_hochberg(raw_p)
    for row, q in zip(results, q_vals):
        row["fdr_q_value"] = q
        row["significant_fdr_05"] = bool(q < 0.05)
    return results


def compute_cross_segment(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    """计算二维交叉分群流失率、样本量、占总流失比与 Wilson 95% CI。"""
    grouped = (
        df.groupby([col_a, col_b], observed=True, dropna=False)
        .agg(customers=(TARGET_COL, "size"), churned=(TARGET_COL, lambda s: int((s == "Yes").sum())))
        .reset_index()
    )
    total_churned = int((df[TARGET_COL] == "Yes").sum())
    grouped["churn_rate"] = grouped["churned"] / grouped["customers"]
    ci_bounds = [wilson_ci(int(k), int(n)) for k, n in zip(grouped["churned"], grouped["customers"])]
    grouped["ci_low"] = [b[0] for b in ci_bounds]
    grouped["ci_high"] = [b[1] for b in ci_bounds]
    grouped["share_of_all_churn"] = grouped["churned"] / max(total_churned, 1)
    grouped["small_sample_flag"] = grouped["customers"] < 30
    return grouped.sort_values("churn_rate", ascending=False).reset_index(drop=True)


def run_full_eda(df: pd.DataFrame) -> dict[str, Any]:
    """运行完整的第二阶段统计检验与多变量交叉分群分析。"""
    work = df.copy()
    work["tenure_group"] = pd.cut(
        work["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12m", "13-24m", "25-48m", "49-72m"],
    )
    work["charge_tier"] = pd.qcut(
        work["MonthlyCharges"],
        q=3,
        labels=["Low (<=$35.5)", "Mid ($35.5-$83.5)", "High (>$83.5)"],
    )

    overall_k = int((work[TARGET_COL] == "Yes").sum())
    overall_n = int(len(work))
    overall_ci = wilson_ci(overall_k, overall_n)

    return {
        "overall": {
            "customers": overall_n,
            "churned": overall_k,
            "retained": overall_n - overall_k,
            "churn_rate": float(overall_k / overall_n),
            "ci_low": overall_ci[0],
            "ci_high": overall_ci[1],
        },
        "categorical_tests": run_categorical_tests(work),
        "numeric_tests": run_numeric_tests(work),
        "tenure_groups": compute_segment_table(work, "tenure_group").to_dict(orient="records"),
        "cross_segments": {
            "contract_x_tenure": compute_cross_segment(work, "Contract", "tenure_group").to_dict(orient="records"),
            "internet_x_techsupport": compute_cross_segment(work, "InternetService", "TechSupport").to_dict(orient="records"),
            "contract_x_payment": compute_cross_segment(work, "Contract", "PaymentMethod").to_dict(orient="records"),
            "contract_x_charge_tier": compute_cross_segment(work, "Contract", "charge_tier").to_dict(orient="records"),
        },
    }
