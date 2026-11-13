"""第六阶段：统一风格高质量静态可视化生成模块（使用英文图表标签以确保跨平台零字体方框）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

from .config import FIGURES_DIR

PALETTE_PRIMARY = "#2B5C8F"
PALETTE_ACCENT = "#C0392B"
PALETTE_TEAL = "#1B7837"
PALETTE_GRAY = "#7F8C8D"


def _apply_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def plot_churn_overview(eda_summary: dict[str, Any], out_dir: Path = FIGURES_DIR) -> Path:
    """图 1：整体流失概览与核心单变量分群流失率（含样本量与 95% CI）。"""
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Overall
    ov = eda_summary["overall"]
    bars = axes[0].bar(["Retained (No)", "Churned (Yes)"], [ov["retained"], ov["churned"]], color=[PALETTE_PRIMARY, PALETTE_ACCENT], width=0.55)
    for bar, val in zip(bars, [ov["retained"], ov["churned"]]):
        pct = val / ov["customers"]
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 120, f"{val:,}\n({pct:.1%})", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].set_ylim(0, ov["customers"] * 0.92)
    axes[0].set_title(f"Overall Churn (n={ov['customers']:,})", fontweight="bold")
    axes[0].set_ylabel("Customers")

    # 2. By Contract
    cat_map = {r["feature"]: r["groups"] for r in eda_summary["categorical_tests"]}
    c_df = pd.DataFrame(cat_map["Contract"])
    y_pos = np.arange(len(c_df))
    xerr = np.vstack([c_df["churn_rate"] - c_df["ci_low"], c_df["ci_high"] - c_df["churn_rate"]])
    axes[1].barh(y_pos, c_df["churn_rate"], xerr=xerr, color=PALETTE_PRIMARY, capsize=4, height=0.55)
    axes[1].set_yticks(y_pos, labels=c_df["Contract"])
    axes[1].invert_yaxis()
    axes[1].axvline(ov["churn_rate"], color=PALETTE_ACCENT, linestyle="--", linewidth=1.2, label=f"Overall {ov['churn_rate']:.1%}")
    for i, r in c_df.iterrows():
        axes[1].text(r["ci_high"] + 0.015, i, f"{r['churn_rate']:.1%} (n={int(r['customers']):,})", va="center", fontsize=9)
    axes[1].set_xlim(0, 0.72)
    axes[1].set_title("Churn Rate by Contract (95% Wilson CI)", fontweight="bold")
    axes[1].set_xlabel("Churn Rate")
    axes[1].legend(loc="lower right", fontsize=8)

    # 3. By Tenure Group
    t_df = pd.DataFrame(eda_summary["tenure_groups"]).sort_values("tenure_group")
    order_map = {"0-12m": 0, "13-24m": 1, "25-48m": 2, "49-72m": 3}
    t_df["ord"] = t_df["tenure_group"].map(order_map)
    t_df = t_df.sort_values("ord")
    yerr = np.vstack([t_df["churn_rate"] - t_df["ci_low"], t_df["ci_high"] - t_df["churn_rate"]])
    axes[2].bar(t_df["tenure_group"], t_df["churn_rate"], yerr=yerr, color=PALETTE_PRIMARY, capsize=4, width=0.55)
    axes[2].axhline(ov["churn_rate"], color=PALETTE_ACCENT, linestyle="--", linewidth=1.2)
    for idx, (_, r) in enumerate(t_df.iterrows()):
        axes[2].text(idx, r["ci_high"] + 0.015, f"{r['churn_rate']:.1%}\n(n={int(r['customers']):,})", ha="center", va="bottom", fontsize=9)
    axes[2].set_ylim(0, 0.65)
    axes[2].set_title("Churn Rate by Tenure Bucket (95% CI)", fontweight="bold")
    axes[2].set_xlabel("Tenure (Months)")

    fig.suptitle("Telco Customer Churn Overview & Core Segments", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = out_dir / "01_churn_overview.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_cross_segments(eda_summary: dict[str, Any], out_dir: Path = FIGURES_DIR) -> Path:
    """图 2：多变量交叉分群热力图（标注流失率与样本量 n）。"""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))

    # Heatmap 1: Contract x Tenure
    ct_df = pd.DataFrame(eda_summary["cross_segments"]["contract_x_tenure"])
    piv_rate = ct_df.pivot(index="Contract", columns="tenure_group", values="churn_rate")[["0-12m", "13-24m", "25-48m", "49-72m"]]
    piv_n = ct_df.pivot(index="Contract", columns="tenure_group", values="customers")[["0-12m", "13-24m", "25-48m", "49-72m"]]
    annot1 = piv_rate.copy().astype(object)
    for r in piv_rate.index:
        for c in piv_rate.columns:
            annot1.loc[r, c] = f"{piv_rate.loc[r, c]:.1%}\n(n={int(piv_n.loc[r, c])})"
    sns.heatmap(piv_rate, annot=annot1, fmt="", cmap="YlOrRd", vmin=0, vmax=0.60, ax=axes[0], cbar_kws={"label": "Churn Rate"})
    axes[0].set_title("Contract x Tenure Group (Rate & Sample Size)", fontweight="bold")
    axes[0].set_xlabel("Tenure Group")

    # Heatmap 2: InternetService x TechSupport
    it_df = pd.DataFrame(eda_summary["cross_segments"]["internet_x_techsupport"])
    piv2_rate = it_df.pivot(index="InternetService", columns="TechSupport", values="churn_rate")
    piv2_n = it_df.pivot(index="InternetService", columns="TechSupport", values="customers")
    annot2 = piv2_rate.copy().astype(object)
    for r in piv2_rate.index:
        for c in piv2_rate.columns:
            val = piv2_rate.loc[r, c]
            cnt = piv2_n.loc[r, c]
            annot2.loc[r, c] = f"{val:.1%}\n(n={int(cnt)})" if pd.notna(val) else "N/A"
    sns.heatmap(piv2_rate, annot=annot2, fmt="", cmap="YlOrRd", vmin=0, vmax=0.60, ax=axes[1], cbar_kws={"label": "Churn Rate"})
    axes[1].set_title("InternetService x TechSupport (Rate & Sample Size)", fontweight="bold")
    axes[1].set_xlabel("TechSupport")

    fig.suptitle("Multi-Dimensional Segment Cross-Analysis (With Explicit Sample Sizes)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = out_dir / "02_cross_segments_heatmap.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_model_curves(
    y_test: pd.Series,
    test_probs: dict[str, np.ndarray],
    calibration_curves: dict[str, dict[str, list[float]]],
    out_dir: Path = FIGURES_DIR,
) -> Path:
    """图 3：三模型测试集 ROC、PR 与概率校准曲线。"""
    _apply_style()
    y_arr = y_test.to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    colors = {"Logistic Regression": PALETTE_PRIMARY, "Random Forest": PALETTE_TEAL, "XGBoost": PALETTE_ACCENT}
    for name, prob in test_probs.items():
        col = colors.get(name, PALETTE_GRAY)
        fpr, tpr, _ = roc_curve(y_arr, prob)
        p, r, _ = precision_recall_curve(y_arr, prob)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_arr, prob):.3f})", color=col, linewidth=2)
        axes[1].plot(r, p, label=f"{name} (AP={average_precision_score(y_arr, prob):.3f})", color=col, linewidth=2)
        cal = calibration_curves[name]
        axes[2].plot(cal["mean_predicted_prob"], cal["fraction_of_positives"], marker="o", label=name, color=col, linewidth=1.8)

    axes[0].plot([0, 1], [0, 1], "--", color=PALETTE_GRAY, linewidth=1)
    axes[0].set(title="Test ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
    axes[1].axhline(y_arr.mean(), linestyle="--", color=PALETTE_GRAY, linewidth=1, label=f"Baseline ({y_arr.mean():.3f})")
    axes[1].set(title="Test Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    axes[2].plot([0, 1], [0, 1], "--", color=PALETTE_GRAY, linewidth=1, label="Perfectly Calibrated")
    axes[2].set(title="Probability Calibration (Test Set)", xlabel="Mean Predicted Probability", ylabel="Observed Churn Fraction")

    for ax in axes:
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("v2 Model Discrimination & Calibration Evaluation (Test Set, n=1,409)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = out_dir / "03_model_evaluation_curves.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_confusion_and_topk(
    y_test: pd.Series,
    best_name: str,
    best_prob: np.ndarray,
    best_threshold: float,
    topk_rows: list[dict[str, Any]],
    out_dir: Path = FIGURES_DIR,
) -> Path:
    """图 4：最优模型混淆矩阵与 Top-K 预算捕获对比。"""
    _apply_style()
    y_arr = y_test.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    cm = confusion_matrix(y_arr, (best_prob >= best_threshold).astype(int), labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=["Pred: No", "Pred: Yes"], yticklabels=["Actual: No", "Actual: Yes"])
    axes[0].set_title(f"{best_name} Confusion Matrix\n(Valid-F1 Threshold={best_threshold:.3f})", fontweight="bold")

    tk = pd.DataFrame(topk_rows)
    axes[1].plot(tk["top_pct"], tk["recall_at_k"] * 100, marker="o", linewidth=2.2, color=PALETTE_PRIMARY, label="Model Top-K Recall (%)")
    axes[1].plot(tk["top_pct"], tk["top_pct"], "--", color=PALETTE_GRAY, linewidth=1.5, label="Random Targeting Baseline (%)")
    for _, r in tk.iterrows():
        axes[1].annotate(f"{r['recall_at_k']:.1%}\n(Lift {r['lift']:.2f}x)", (r["top_pct"], r["recall_at_k"] * 100),
                         textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    axes[1].set(title=f"{best_name} Top-K Churn Capture Curve", xlabel="Contacted Population (%)", ylabel="Churners Captured (%)", ylim=(0, 100))
    axes[1].legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    out_path = out_dir / "04_confusion_and_topk.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_explainability(explain_res: dict[str, Any], out_dir: Path = FIGURES_DIR) -> Path:
    """图 5：XGBoost 原生 TreeSHAP 全局重要度 + Logistic Regression 标准化系数方向。"""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    shap_df = pd.DataFrame(explain_res["xgboost_treeshap"]["raw_feature_shap"]).head(10)
    y_pos1 = np.arange(len(shap_df))
    axes[0].barh(y_pos1, shap_df["mean_abs_shap"], color=PALETTE_PRIMARY, height=0.6)
    axes[0].set_yticks(y_pos1, labels=shap_df["feature"])
    axes[0].invert_yaxis()
    axes[0].set(title="XGBoost TreeSHAP Importance (Raw Features)", xlabel="Mean |SHAP| (Log-Odds)")

    pos = pd.DataFrame(explain_res["logistic_regression"]["top_positive_drivers"]).head(5)
    neg = pd.DataFrame(explain_res["logistic_regression"]["top_negative_drivers"]).head(5)
    lr_df = pd.concat([pos, neg], ignore_index=True).sort_values("coef", ascending=True)
    lr_df["short_name"] = lr_df["feature"].str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False)
    bar_colors = [PALETTE_ACCENT if c > 0 else PALETTE_TEAL for c in lr_df["coef"]]
    axes[1].barh(np.arange(len(lr_df)), lr_df["coef"], color=bar_colors, height=0.6)
    axes[1].set_yticks(np.arange(len(lr_df)), labels=lr_df["short_name"])
    axes[1].axvline(0, color="black", linewidth=0.9)
    axes[1].set(title="Logistic Regression Top Drivers (Log-Odds Coef)", xlabel="Coefficient (+ Increases Churn, - Reduces Churn)")

    fig.suptitle("Model Explainability: Global Feature Importance & Directionality", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = out_dir / "05_feature_explainability.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_budget_scenarios(biz_res: dict[str, Any], out_dir: Path = FIGURES_DIR) -> Path:
    """图 6：不同触达成本与假设挽回率下的 Top-K 预算敏感性曲线（清晰标注假设条件）。"""
    _apply_style()
    sc_df = pd.DataFrame(biz_res["scenario_grid"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：固定假设挽回率 = 25%，对比不同单客触达成本 ($10 / $20 / $30)
    sub_save = sc_df[sc_df["assumed_save_rate"] == 0.25]
    for cost, grp in sub_save.groupby("contact_cost_per_user"):
        axes[0].plot(grp["top_pct"], grp["assumed_net_benefit"], marker="o", linewidth=2, label=f"Cost=${int(cost)}/user")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set(title="Assumed Net Benefit vs Top-K (Assumed Save Rate=25%)", xlabel="Target Top-K (%)", ylabel="Assumed Net Benefit ($)")
    axes[0].legend(fontsize=9)

    # 右图：固定单客触达成本 = $20，对比不同假设挽回率 (15% / 25% / 35%)
    sub_cost = sc_df[sc_df["contact_cost_per_user"] == 20.0]
    for sr, grp in sub_cost.groupby("assumed_save_rate"):
        axes[1].plot(grp["top_pct"], grp["assumed_net_benefit"], marker="s", linewidth=2, label=f"Assumed Save Rate={sr:.0%}")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set(title="Assumed Net Benefit vs Top-K (Contact Cost=$20/user)", xlabel="Target Top-K (%)", ylabel="Assumed Net Benefit ($)")
    axes[1].legend(fontsize=9)

    fig.suptitle("Hypothetical Retention Budget Sensitivity Analysis (Not Observed ROI)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = out_dir / "06_budget_scenarios.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path
