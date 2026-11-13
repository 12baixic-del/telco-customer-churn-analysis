"""端到端可复现分析与建模流水线入口。

运行方式（在项目根目录）：
    python scripts/run_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.business_value import run_business_scenarios
from src.config import FIGURES_DIR, LEGACY_V1_DIR, REPORTS_DIR
from src.data_prep import load_and_clean_data, split_train_valid_test
from src.eda_analysis import run_full_eda
from src.explainability import run_explainability
from src.modeling import train_and_evaluate_models
from src.visualization import (
    plot_budget_scenarios,
    plot_churn_overview,
    plot_confusion_and_topk,
    plot_cross_segments,
    plot_explainability,
    plot_model_curves,
)


def _save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 数据清洗与质量审计
    df, audit = load_and_clean_data(save_processed=True)
    print(f"[1/6] 数据清洗完成：{audit['raw_rows']:,} 行，流失率 {audit['overall_churn_rate']:.2%}")

    # 2. 第二阶段：统计检验与多变量交叉分群
    eda_summary = run_full_eda(df)
    eda_summary["data_audit"] = audit
    _save_json(eda_summary, REPORTS_DIR / "eda_stats_summary.json")
    print(f"[2/6] 统计检验与交叉分群完成：{len(eda_summary['categorical_tests'])} 个分类变量 + {len(eda_summary['numeric_tests'])} 个数值变量")

    # 3. 严格分层划分与第三阶段建模
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_train_valid_test(df)
    print(f"[3/6] 数据集划分：Train={len(X_train):,}, Valid={len(X_valid):,}, Test={len(X_test):,}")

    model_res = train_and_evaluate_models(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_df = model_res["metrics_df"]
    metrics_df.to_csv(REPORTS_DIR / "model_comparison.csv")

    # 汇总 CV、不平衡消融、校准、Bootstrap 与 Top-K
    v1_path = LEGACY_V1_DIR / "model_comparison_v1.csv"
    v1_records = []
    if v1_path.exists():
        import pandas as pd
        v1_records = pd.read_csv(v1_path).to_dict(orient="records")

    cv_payload = {
        "v1_baseline": v1_records,
        "cv_summary": model_res["cv_summary"],
        "imbalance_ablation": model_res["imbalance_ablation"],
        "calibration_summary": model_res["calibration_summary"],
        "calibration_curves": model_res["calibration_curves"],
        "bootstrap_cis": model_res["bootstrap_cis"],
        "topk_results": model_res["topk_results"],
        "threshold_sensitivity": model_res["threshold_sensitivity"],
    }
    _save_json(cv_payload, REPORTS_DIR / "model_cv_summary.json")
    print("[3/6] 模型评估完成（按 PR-AUC 排序）：")
    print(metrics_df[["roc_auc", "pr_auc", "brier_score", "precision", "recall", "f1"]].round(4))

    # 选出 PR-AUC 最高的主力模型用于 Top-K 与商业情景基准（排除 Dummy）
    candidate_models = metrics_df.drop(index=["Dummy Baseline (Stratified)"], errors="ignore")
    best_model_name = str(candidate_models.index[0])
    best_prob = model_res["test_probs"][best_model_name]
    best_thr = float(candidate_models.loc[best_model_name, "validation_threshold"])

    # 4. 第四阶段：可解释性（LR 优势比 + 置换重要性 + XGBoost TreeSHAP + 典型画像/误判）
    xgb_thr = float(metrics_df.loc["XGBoost", "validation_threshold"])
    explain_res = run_explainability(
        fitted_models=model_res["fitted_models"],
        X_test=X_test,
        y_test=y_test,
        xgb_prob=model_res["test_probs"]["XGBoost"],
        xgb_threshold=xgb_thr,
    )
    _save_json(explain_res, REPORTS_DIR / "explainability_summary.json")
    print("[4/6] 模型可解释性分析完成（LR 优势比 + 置换重要性 + XGBoost TreeSHAP）")

    # 5. 第五阶段：商业价值与 A/B 测试设计
    biz_res = run_business_scenarios(X_test, y_test, best_model_name=best_model_name, best_prob=best_prob)
    _save_json(biz_res, REPORTS_DIR / "business_scenarios.json")
    print(f"[5/6] 商业情景分析完成（基准排序模型：{best_model_name}）")

    # 6. 第六阶段：生成 6 张高质量静态图表
    plot_churn_overview(eda_summary, FIGURES_DIR)
    plot_cross_segments(eda_summary, FIGURES_DIR)
    plot_model_curves(y_test, model_res["test_probs"], model_res["calibration_curves"], FIGURES_DIR)
    plot_confusion_and_topk(
        y_test,
        best_name=best_model_name,
        best_prob=best_prob,
        best_threshold=best_thr,
        topk_rows=model_res["topk_results"][best_model_name],
        out_dir=FIGURES_DIR,
    )
    plot_explainability(explain_res, FIGURES_DIR)
    plot_budget_scenarios(biz_res, FIGURES_DIR)
    print(f"[6/6] 全部 6 张图表已保存至 {FIGURES_DIR}")


if __name__ == "__main__":
    main()
