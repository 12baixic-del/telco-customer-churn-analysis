"""第五阶段：商业价值情景分析、风险分层、Top-K 触达预算与 A/B 测试样本量设计模块。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def build_risk_tiers(X_test: pd.DataFrame, y_test: pd.Series, prob: np.ndarray) -> list[dict[str, Any]]:
    """按预测流失概率划分为 4 个风险分层，输出各层样本量、实际流失率与月费特征。"""
    work = X_test.reset_index(drop=True).copy()
    work["actual_churn"] = y_test.to_numpy()
    work["pred_prob"] = prob

    bins = [-0.01, 0.20, 0.40, 0.60, 1.01]
    labels = ["Low (<20%)", "Medium-Low (20%-40%)", "Medium-High (40%-60%)", "High (>=60%)"]
    work["risk_tier"] = pd.cut(work["pred_prob"], bins=bins, labels=labels)

    total_n = len(work)
    total_churn = int(work["actual_churn"].sum())
    rows = []
    for label in reversed(labels):
        sub = work[work["risk_tier"] == label]
        n = int(len(sub))
        churned = int(sub["actual_churn"].sum())
        rows.append({
            "risk_tier": label,
            "customers": n,
            "share_of_customers": float(n / total_n) if total_n > 0 else 0.0,
            "actual_churned": churned,
            "actual_churn_rate": float(churned / n) if n > 0 else 0.0,
            "share_of_all_churn": float(churned / total_churn) if total_churn > 0 else 0.0,
            "avg_monthly_charges": float(sub["MonthlyCharges"].mean()) if n > 0 else 0.0,
        })
    return rows


def simulate_budget_scenario(
    y_true: np.ndarray,
    monthly_charges: np.ndarray,
    prob: np.ndarray,
    top_pct: int,
    contact_cost: float,
    assumed_save_rate: float,
    assumed_retained_months: int = 6,
    assumed_gross_margin: float = 0.60,
) -> dict[str, Any]:
    """在明确标注的假设下，模拟 Top-K% 触达预算的成本与假设净收益（非真实财务 ROI）。"""
    n = len(y_true)
    k = max(1, int(round(n * (top_pct / 100.0))))
    order = np.argsort(-prob)[:k]

    tp_mask = y_true[order] == 1
    captured_churners = int(tp_mask.sum())
    total_cost = float(k * contact_cost)

    # 仅对真实潜在流失者（TP）按假设挽回率计算假设挽回价值；FP 仅产生触达成本
    tp_monthly_sum = float(monthly_charges[order][tp_mask].sum())
    assumed_saved_customers = float(captured_churners * assumed_save_rate)
    assumed_gross_saved_value = float(
        tp_monthly_sum * assumed_save_rate * assumed_retained_months * assumed_gross_margin
    )
    assumed_net_benefit = float(assumed_gross_saved_value - total_cost)

    return {
        "top_pct": int(top_pct),
        "contacted_customers": k,
        "captured_churners": captured_churners,
        "precision_at_k": float(captured_churners / k),
        "total_contact_cost": total_cost,
        "assumed_saved_customers": assumed_saved_customers,
        "assumed_gross_saved_value": assumed_gross_saved_value,
        "assumed_net_benefit": assumed_net_benefit,
    }


def compute_ab_test_sample_size(
    baseline_rate: float,
    relative_lift: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """计算双侧比例检验探测相对流失率下降所需每组最小样本量。"""
    p1 = float(baseline_rate)
    p2 = float(max(1e-6, p1 * (1.0 - relative_lift)))
    p_bar = 0.5 * (p1 + p2)
    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_beta = float(stats.norm.ppf(power))
    numer = (z_alpha * np.sqrt(2.0 * p_bar * (1.0 - p_bar)) + z_beta * np.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))) ** 2
    denom = max((p1 - p2) ** 2, 1e-12)
    return int(np.ceil(numer / denom))


def run_business_scenarios(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_model_name: str,
    best_prob: np.ndarray,
) -> dict[str, Any]:
    """生成完整的商业风险分层、预算情景矩阵与 A/B 测试样本量设计。"""
    y_arr = y_test.to_numpy()
    mc_arr = X_test["MonthlyCharges"].to_numpy(dtype=float)

    risk_tiers = build_risk_tiers(X_test, y_test, best_prob)

    # 情景矩阵：触达成本 ($10 / $20 / $30) × 假设挽回率 (15% / 25% / 35%) × Top-K (10% / 20% / 30% / 40%)
    scenario_grid = []
    for cost in (10.0, 20.0, 30.0):
        for save_rate in (0.15, 0.25, 0.35):
            for top_pct in (10, 20, 30, 40):
                sim = simulate_budget_scenario(
                    y_true=y_arr,
                    monthly_charges=mc_arr,
                    prob=best_prob,
                    top_pct=top_pct,
                    contact_cost=cost,
                    assumed_save_rate=save_rate,
                )
                sim["contact_cost_per_user"] = cost
                sim["assumed_save_rate"] = save_rate
                scenario_grid.append(sim)

    # 以 High Risk + Medium-High (>=40%) 作为试点试验池基准流失率测算 A/B 样本量
    pilot_mask = best_prob >= 0.40
    pilot_base_rate = float(y_arr[pilot_mask].mean()) if pilot_mask.any() else float(y_arr.mean())
    ab_designs = [
        {
            "target_pool": "模型预测高风险池 (p >= 0.40)",
            "baseline_churn_rate": pilot_base_rate,
            "relative_reduction_target": rel_red,
            "expected_treatment_rate": float(pilot_base_rate * (1.0 - rel_red)),
            "required_n_per_arm": compute_ab_test_sample_size(pilot_base_rate, rel_red),
            "required_total_n": 2 * compute_ab_test_sample_size(pilot_base_rate, rel_red),
        }
        for rel_red in (0.15, 0.20, 0.25)
    ]

    return {
        "reference_model": best_model_name,
        "disclaimer": (
            "本数据集不包含真实干预成本、客户毛利率与挽回转化率；"
            "以下情景模拟基于明确标注的假设参数（额外留存 6 个月、毛利率 60%），"
            "仅用于预算敏感性规划与优先级排序参考，不得视为真实商业 ROI。"
        ),
        "risk_tiers": risk_tiers,
        "scenario_grid": scenario_grid,
        "ab_test_sample_sizes": ab_designs,
    }
