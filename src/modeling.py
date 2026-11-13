"""第三阶段：严谨机器学习建模、CV 调优、不平衡对比、校准与稳定性评估模块。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .config import RANDOM_STATE
from .data_prep import get_feature_groups


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """构建仅在训练集拟合的预处理 ColumnTransformer。"""
    num_cols, cat_cols = get_feature_groups(X_train)
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


def select_f1_threshold(y_true: pd.Series, prob: np.ndarray) -> float:
    """在验证集上选择 F1 最大化分类阈值（禁止在测试集调用）。"""
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    if len(thresholds) == 0:
        return 0.5
    f1_scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1_scores))])


def bootstrap_metric_ci(
    y_true: np.ndarray,
    prob: np.ndarray,
    n_boot: int = 1000,
    seed: int = RANDOM_STATE,
) -> dict[str, dict[str, float]]:
    """对测试集预测概率进行分层 Bootstrap，计算 ROC-AUC 与 PR-AUC 的 95% CI。"""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    roc_vals, pr_vals = [], []
    for _ in range(n_boot):
        s_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        s_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([s_pos, s_neg])
        y_b, p_b = y_true[idx], prob[idx]
        roc_vals.append(roc_auc_score(y_b, p_b))
        pr_vals.append(average_precision_score(y_b, p_b))
    return {
        "roc_auc": {
            "mean": float(np.mean(roc_vals)),
            "std": float(np.std(roc_vals)),
            "ci_low": float(np.quantile(roc_vals, 0.025)),
            "ci_high": float(np.quantile(roc_vals, 0.975)),
        },
        "pr_auc": {
            "mean": float(np.mean(pr_vals)),
            "std": float(np.std(pr_vals)),
            "ci_low": float(np.quantile(pr_vals, 0.025)),
            "ci_high": float(np.quantile(pr_vals, 0.975)),
        },
    }


def evaluate_topk(y_true: np.ndarray, prob: np.ndarray, top_fractions: tuple[float, ...] = (0.10, 0.15, 0.20, 0.30, 0.40)) -> list[dict[str, Any]]:
    """评估不同触达预算（Top-K 比例）下的精确率、召回率与 Lift。"""
    n = len(y_true)
    total_pos = int(y_true.sum())
    base_rate = float(total_pos / n) if n > 0 else 0.0
    order = np.argsort(-prob)
    rows = []
    for frac in top_fractions:
        k = max(1, int(round(n * frac)))
        sel = order[:k]
        tp = int(y_true[sel].sum())
        prec = float(tp / k)
        rec = float(tp / total_pos) if total_pos > 0 else 0.0
        lift = float(prec / base_rate) if base_rate > 0 else 0.0
        cutoff_prob = float(prob[sel[-1]])
        rows.append({
            "top_pct": int(round(frac * 100)),
            "k_customers": k,
            "tp_captured": tp,
            "fp_wasted": k - tp,
            "precision_at_k": prec,
            "recall_at_k": rec,
            "lift": lift,
            "prob_cutoff": cutoff_prob,
        })
    return rows


def evaluate_threshold_sensitivity(y_true: np.ndarray, prob: np.ndarray) -> list[dict[str, Any]]:
    """计算 0.10 ~ 0.80 阈值区间内的精确率、召回率、F1 与预计触达人数。"""
    rows = []
    for thr in np.round(np.arange(0.10, 0.81, 0.05), 2):
        pred = (prob >= thr).astype(int)
        rows.append({
            "threshold": float(thr),
            "contacted": int(pred.sum()),
            "contacted_pct": float(pred.mean()),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
        })
    return rows


def _evaluate_imbalance_ablation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    pos_weight: float,
) -> list[dict[str, Any]]:
    """在训练集拟合并在验证集评估默认权重 vs 类别不平衡权重的真实收益。"""
    configs = [
        ("Logistic Regression", "default", LogisticRegression(max_iter=2000, C=0.5, random_state=RANDOM_STATE)),
        ("Logistic Regression", "balanced", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced", random_state=RANDOM_STATE)),
        ("Random Forest", "default", RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10, random_state=RANDOM_STATE, n_jobs=-1)),
        ("Random Forest", "balanced_subsample", RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)),
        ("XGBoost", "default", XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)),
        ("XGBoost", f"scale_pos_weight={pos_weight:.2f}", XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)),
    ]
    rows = []
    for model_name, weight_mode, est in configs:
        pipe = Pipeline([("preprocess", build_preprocessor(X_train)), ("model", est)])
        pipe.fit(X_train, y_train)
        v_prob = pipe.predict_proba(X_valid)[:, 1]
        thr = select_f1_threshold(y_valid, v_prob)
        v_pred = (v_prob >= thr).astype(int)
        rows.append({
            "model": model_name,
            "weighting": weight_mode,
            "valid_roc_auc": float(roc_auc_score(y_valid, v_prob)),
            "valid_pr_auc": float(average_precision_score(y_valid, v_prob)),
            "valid_brier": float(brier_score_loss(y_valid, v_prob)),
            "valid_f1": float(f1_score(y_valid, v_pred, zero_division=0)),
        })
    return rows


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """执行第三阶段完整建模流水线：基线、5 折 CV 超参搜索、不平衡消融、校准、Bootstrap 与 Top-K。"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    # 1. 类别不平衡处理在验证集上的消融实验
    imbalance_ablation = _evaluate_imbalance_ablation(X_train, y_train, X_valid, y_valid, pos_weight)

    # 2. 定义各模型搜索空间（在 X_train 上严格 5 折交叉验证优化 PR-AUC）
    search_specs: dict[str, tuple[Any, dict[str, list[Any]]]] = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {
                "model__C": [0.05, 0.1, 0.5, 1.0, 5.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [300],
                "model__max_depth": [6, 8, 12],
                "model__min_samples_leaf": [5, 15, 30],
                "model__class_weight": [None, "balanced_subsample"],
            },
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [150, 250],
                "model__max_depth": [2, 3, 4],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__subsample": [0.8],
                "model__colsample_bytree": [0.8],
                "model__reg_lambda": [1.0, 5.0],
                "model__scale_pos_weight": [1.0, round(pos_weight, 2)],
            },
        ),
    }

    fitted_models: dict[str, Pipeline] = {}
    cv_records: list[dict[str, Any]] = []
    valid_probs: dict[str, np.ndarray] = {}
    test_probs: dict[str, np.ndarray] = {}
    calibration_records: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    bootstrap_cis: dict[str, Any] = {}
    topk_results: dict[str, list[dict[str, Any]]] = {}
    threshold_sensitivity: dict[str, list[dict[str, Any]]] = {}
    calibration_curves: dict[str, dict[str, list[float]]] = {}

    # 3. 简单基线模型（分层随机基线，仅作下界参照）
    dummy_pipe = Pipeline([
        ("preprocess", build_preprocessor(X_train)),
        ("model", DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)),
    ])
    dummy_pipe.fit(X_train, y_train)
    dummy_test_prob = dummy_pipe.predict_proba(X_test)[:, 1]
    dummy_pred = dummy_pipe.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, dummy_pred, labels=[0, 1]).ravel()
    test_rows.append({
        "model": "Dummy Baseline (Stratified)",
        "validation_threshold": 0.5,
        "roc_auc": float(roc_auc_score(y_test, dummy_test_prob)),
        "pr_auc": float(average_precision_score(y_test, dummy_test_prob)),
        "brier_score": float(brier_score_loss(y_test, dummy_test_prob)),
        "precision": float(precision_score(y_test, dummy_pred, zero_division=0)),
        "recall": float(recall_score(y_test, dummy_pred, zero_division=0)),
        "f1": float(f1_score(y_test, dummy_pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "contacted": int(dummy_pred.sum()),
    })

    # 4. 三大主力模型：CV 调优 -> 验证集选阈值与校准检查 -> 测试集一次最终评估
    y_test_arr = y_test.to_numpy()
    for name, (estimator, param_grid) in search_specs.items():
        pipe = Pipeline([("preprocess", build_preprocessor(X_train)), ("model", estimator)])
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"},
            refit="pr_auc",
            cv=cv,
            n_jobs=-1,
        )
        gs.fit(X_train, y_train)
        best_pipe: Pipeline = gs.best_estimator_
        best_idx = int(gs.best_index_)
        cv_records.append({
            "model": name,
            "best_params": {k.replace("model__", ""): v for k, v in gs.best_params_.items()},
            "cv_pr_auc_mean": float(gs.cv_results_["mean_test_pr_auc"][best_idx]),
            "cv_pr_auc_std": float(gs.cv_results_["std_test_pr_auc"][best_idx]),
            "cv_roc_auc_mean": float(gs.cv_results_["mean_test_roc_auc"][best_idx]),
            "cv_roc_auc_std": float(gs.cv_results_["std_test_roc_auc"][best_idx]),
        })

        v_prob = best_pipe.predict_proba(X_valid)[:, 1]
        t_prob = best_pipe.predict_proba(X_test)[:, 1]
        thr = select_f1_threshold(y_valid, v_prob)

        # 检查 Sigmoid 校准（仅在训练集 CV 内拟合，避免泄露验证集或测试集）
        cal_pipe = CalibratedClassifierCV(best_pipe, method="sigmoid", cv=5)
        cal_pipe.fit(X_train, y_train)
        cal_v_prob = cal_pipe.predict_proba(X_valid)[:, 1]
        cal_t_prob = cal_pipe.predict_proba(X_test)[:, 1]

        raw_v_brier = float(brier_score_loss(y_valid, v_prob))
        cal_v_brier = float(brier_score_loss(y_valid, cal_v_prob))
        # 根据验证集 Brier Score 决定最终用于概率输出的模型（原始或校准后）
        use_calibrated = cal_v_brier < raw_v_brier - 1e-4
        final_pipe = cal_pipe if use_calibrated else best_pipe
        final_v_prob = cal_v_prob if use_calibrated else v_prob
        final_t_prob = cal_t_prob if use_calibrated else t_prob
        final_thr = select_f1_threshold(y_valid, final_v_prob)

        calibration_records.append({
            "model": name,
            "valid_brier_raw": raw_v_brier,
            "valid_brier_sigmoid": cal_v_brier,
            "selected_calibrated": bool(use_calibrated),
            "test_brier_final": float(brier_score_loss(y_test_arr, final_t_prob)),
        })

        t_pred = (final_t_prob >= final_thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test_arr, t_pred, labels=[0, 1]).ravel()
        test_rows.append({
            "model": name,
            "validation_threshold": float(final_thr),
            "roc_auc": float(roc_auc_score(y_test_arr, final_t_prob)),
            "pr_auc": float(average_precision_score(y_test_arr, final_t_prob)),
            "brier_score": float(brier_score_loss(y_test_arr, final_t_prob)),
            "precision": float(precision_score(y_test_arr, t_pred, zero_division=0)),
            "recall": float(recall_score(y_test_arr, t_pred, zero_division=0)),
            "f1": float(f1_score(y_test_arr, t_pred, zero_division=0)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "contacted": int(t_pred.sum()),
        })

        fitted_models[name] = best_pipe
        valid_probs[name] = final_v_prob
        test_probs[name] = final_t_prob
        bootstrap_cis[name] = bootstrap_metric_ci(y_test_arr, final_t_prob)
        topk_results[name] = evaluate_topk(y_test_arr, final_t_prob)
        threshold_sensitivity[name] = evaluate_threshold_sensitivity(y_test_arr, final_t_prob)

        frac_pos, mean_pred = calibration_curve(y_test_arr, final_t_prob, n_bins=10, strategy="quantile")
        calibration_curves[name] = {
            "mean_predicted_prob": [float(x) for x in mean_pred],
            "fraction_of_positives": [float(x) for x in frac_pos],
        }

    metrics_df = pd.DataFrame(test_rows).set_index("model").sort_values("pr_auc", ascending=False)
    return {
        "metrics_df": metrics_df,
        "fitted_models": fitted_models,
        "valid_probs": valid_probs,
        "test_probs": test_probs,
        "cv_summary": cv_records,
        "imbalance_ablation": imbalance_ablation,
        "calibration_summary": calibration_records,
        "calibration_curves": calibration_curves,
        "bootstrap_cis": bootstrap_cis,
        "topk_results": topk_results,
        "threshold_sensitivity": threshold_sensitivity,
    }
