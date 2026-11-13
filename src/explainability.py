"""第四阶段：模型可解释性（LR 优势比、原始特征置换重要性、XGBoost 原生 TreeSHAP、典型画像与误判分析）。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE


def extract_lr_coefficients(lr_pipe: Pipeline, top_n: int = 12) -> dict[str, Any]:
    """提取 Logistic Regression 标准化系数与优势比（Odds Ratio）。"""
    pre = lr_pipe.named_steps["preprocess"]
    model = lr_pipe.named_steps["model"]
    feat_names = pre.get_feature_names_out()
    coefs = model.coef_[0]
    df_coef = pd.DataFrame({
        "feature": feat_names,
        "coef": coefs,
        "odds_ratio": np.exp(coefs),
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False)

    pos_drivers = df_coef[df_coef["coef"] > 0].head(top_n).drop(columns=["abs_coef"])
    neg_drivers = df_coef[df_coef["coef"] < 0].head(top_n).drop(columns=["abs_coef"])
    return {
        "intercept": float(model.intercept_[0]),
        "top_positive_drivers": pos_drivers.to_dict(orient="records"),
        "top_negative_drivers": neg_drivers.to_dict(orient="records"),
        "all_coefficients": df_coef.drop(columns=["abs_coef"]).to_dict(orient="records"),
    }


def compute_raw_permutation_importance(
    pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
) -> list[dict[str, Any]]:
    """在原始业务字段粒度上计算基于 PR-AUC 的置换重要性，避免 OneHot 拆分掩盖整体贡献。"""
    perm = permutation_importance(
        pipe,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rows = []
    for col, mean_imp, std_imp in zip(X_test.columns, perm.importances_mean, perm.importances_std):
        rows.append({
            "feature": col,
            "importance_mean": float(mean_imp),
            "importance_std": float(std_imp),
        })
    return sorted(rows, key=lambda r: r["importance_mean"], reverse=True)


def _map_encoded_to_raw(encoded_name: str, raw_cols: list[str]) -> str:
    """将 ColumnTransformer 输出名映射回原始业务字段名。"""
    clean = encoded_name.split("__", 1)[-1]
    # 优先精确匹配数值列
    if clean in raw_cols:
        return clean
    # OneHot 列通常为 "<col>_<category>"
    for col in sorted(raw_cols, key=len, reverse=True):
        if clean.startswith(f"{col}_"):
            return col
    return clean


def compute_xgb_treeshap(
    xgb_pipe: Pipeline,
    X_test: pd.DataFrame,
) -> dict[str, Any]:
    """利用 XGBoost 原生 pred_contribs=True 计算精确 TreeSHAP 贡献（无需外部 shap 包）。"""
    pre = xgb_pipe.named_steps["preprocess"]
    model = xgb_pipe.named_steps["model"]
    X_enc = pre.transform(X_test)
    enc_names = list(pre.get_feature_names_out())

    booster = model.get_booster()
    dmat = xgb.DMatrix(X_enc, feature_names=enc_names)
    contribs = booster.predict(dmat, pred_contribs=True)  # shape: (n_samples, n_features + 1)
    shap_vals = contribs[:, :-1]
    base_val = float(contribs[0, -1])

    # 按编码后特征统计平均绝对 SHAP
    enc_summary = pd.DataFrame({
        "encoded_feature": [n.split("__", 1)[-1] for n in enc_names],
        "mean_abs_shap": np.mean(np.abs(shap_vals), axis=0),
        "mean_shap": np.mean(shap_vals, axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    # 聚合回原始业务字段（可加性：同一原始字段的 OneHot 虚拟列 SHAP 直接相加）
    raw_cols = list(X_test.columns)
    raw_map = [_map_encoded_to_raw(n, raw_cols) for n in enc_names]
    raw_shap = np.zeros((len(X_test), len(raw_cols)), dtype=float)
    col_to_idx = {c: i for i, c in enumerate(raw_cols)}
    for j, r_col in enumerate(raw_map):
        raw_shap[:, col_to_idx[r_col]] += shap_vals[:, j]

    raw_summary = pd.DataFrame({
        "feature": raw_cols,
        "mean_abs_shap": np.mean(np.abs(raw_shap), axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    return {
        "base_value_logit": base_val,
        "top_encoded_shap": enc_summary.head(15).to_dict(orient="records"),
        "raw_feature_shap": raw_summary.to_dict(orient="records"),
        "_raw_shap_matrix": raw_shap,
        "_raw_cols": raw_cols,
    }


def extract_case_studies(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    prob: np.ndarray,
    raw_shap_matrix: np.ndarray,
    raw_cols: list[str],
    threshold: float,
) -> dict[str, Any]:
    """抽取匿名高风险客户画像以及典型误判（FP / FN）案例及 TreeSHAP 局部归因。"""
    y_arr = y_test.to_numpy()
    df_work = X_test.reset_index(drop=True).copy()

    def _build_profile(idx: int, label: str) -> dict[str, Any]:
        row = df_work.iloc[idx]
        contribs = sorted(
            zip(raw_cols, raw_shap_matrix[idx]),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:5]
        return {
            "case_label": label,
            "actual_churn": int(y_arr[idx]),
            "predicted_prob": float(prob[idx]),
            "profile": {
                "tenure_months": int(row["tenure"]),
                "Contract": str(row["Contract"]),
                "InternetService": str(row["InternetService"]),
                "MonthlyCharges": float(row["MonthlyCharges"]),
                "PaymentMethod": str(row["PaymentMethod"]),
                "TechSupport": str(row["TechSupport"]),
                "OnlineSecurity": str(row["OnlineSecurity"]),
                "PaperlessBilling": str(row["PaperlessBilling"]),
            },
            "top_shap_reasons": [
                {"feature": f, "shap_logit": float(v), "direction": "增加风险" if v > 0 else "降低风险"}
                for f, v in contribs
            ],
        }

    # 1. 典型高风险且确实流失的样本（按概率从高到低取前 2 个不同特征代表）
    tp_idx = np.where((y_arr == 1) & (prob >= threshold))[0]
    tp_sorted = tp_idx[np.argsort(-prob[tp_idx])] if len(tp_idx) > 0 else np.array([int(np.argmax(prob))])
    high_risk_profiles = [_build_profile(int(i), f"高风险典型客户 #{k+1}") for k, i in enumerate(tp_sorted[:2])]

    # 2. 典型误报（FP：模型预测高风险但实际未流失）
    fp_idx = np.where((y_arr == 0) & (prob >= threshold))[0]
    fp_cases = []
    if len(fp_idx) > 0:
        fp_top = int(fp_idx[np.argmax(prob[fp_idx])])
        fp_cases.append(_build_profile(fp_top, "典型误报 (FP：高预测风险但实际留存)"))

    # 3. 典型漏报（FN：模型预测低风险但实际流失）
    fn_idx = np.where((y_arr == 1) & (prob < threshold))[0]
    fn_cases = []
    if len(fn_idx) > 0:
        fn_low = int(fn_idx[np.argmin(prob[fn_idx])])
        fn_cases.append(_build_profile(fn_low, "典型漏报 (FN：低预测风险但实际流失)"))

    # 计算数值特征共线性提示（如 tenure vs TotalCharges）
    corr_tenure_total = float(df_work[["tenure", "TotalCharges"]].dropna().corr().iloc[0, 1])

    return {
        "high_risk_profiles": high_risk_profiles,
        "false_positive_cases": fp_cases,
        "false_negative_cases": fn_cases,
        "collinearity_note": {
            "tenure_vs_total_charges_corr": corr_tenure_total,
            "explanation": (
                f"在网时长 tenure 与累计消费 TotalCharges 在测试集上的皮尔逊相关系数为 {corr_tenure_total:.3f}，"
                "且 Fiber optic 网络与更高 MonthlyCharges 高度伴随；因此线性回归系数与树模型归因可能在相关特征间分摊权重，"
                "不得将单个特征重要性直接解读为独立因果效应。"
            ),
        },
    }


def run_explainability(
    fitted_models: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    xgb_prob: np.ndarray,
    xgb_threshold: float,
) -> dict[str, Any]:
    """运行第四阶段全部可解释性分析并输出可序列化结果。"""
    lr_res = extract_lr_coefficients(fitted_models["Logistic Regression"])
    lr_perm = compute_raw_permutation_importance(fitted_models["Logistic Regression"], X_test, y_test)
    rf_perm = compute_raw_permutation_importance(fitted_models["Random Forest"], X_test, y_test)
    xgb_perm = compute_raw_permutation_importance(fitted_models["XGBoost"], X_test, y_test)
    xgb_shap = compute_xgb_treeshap(fitted_models["XGBoost"], X_test)

    cases = extract_case_studies(
        X_test=X_test,
        y_test=y_test,
        prob=xgb_prob,
        raw_shap_matrix=xgb_shap.pop("_raw_shap_matrix"),
        raw_cols=xgb_shap.pop("_raw_cols"),
        threshold=xgb_threshold,
    )

    return {
        "logistic_regression": lr_res,
        "permutation_importance": {
            "Logistic Regression": lr_perm,
            "Random Forest": rf_perm,
            "XGBoost": xgb_perm,
        },
        "xgboost_treeshap": xgb_shap,
        "case_studies": cases,
    }
