"""Telco Customer Churn v2 自动化单元测试与产出一致性验证。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.business_value import compute_ab_test_sample_size, simulate_budget_scenario
from src.config import DATA_RAW_PATH, FIGURES_DIR, LEGACY_V1_DIR, REPORTS_DIR
from src.data_prep import load_and_clean_data, split_train_valid_test
from src.eda_analysis import benjamini_hochberg, wilson_ci
from src.modeling import build_preprocessor, evaluate_topk


class TestDataAndIsolation(unittest.TestCase):
    def test_load_and_clean_audit(self):
        """验证 7043 行清洗结果与 TotalCharges 11 个空白对应 tenure==0。"""
        df, audit = load_and_clean_data(DATA_RAW_PATH, save_processed=False)
        self.assertEqual(len(df), 7043)
        self.assertEqual(audit["duplicate_customer_ids"], 0)
        self.assertEqual(audit["missing_counts"], {"TotalCharges": 11})
        self.assertEqual(audit["blank_total_charges_tenures"], {0: 11})

    def test_strict_train_valid_test_isolation(self):
        """验证 60/20/20 分层划分完全互斥且无索引重叠。"""
        df, _ = load_and_clean_data(DATA_RAW_PATH, save_processed=False)
        X_train, X_valid, X_test, y_train, y_valid, y_test = split_train_valid_test(df)
        self.assertEqual((len(X_train), len(X_valid), len(X_test)), (4225, 1409, 1409))

        idx_tr, idx_va, idx_te = set(X_train.index), set(X_valid.index), set(X_test.index)
        self.assertTrue(idx_tr.isdisjoint(idx_va))
        self.assertTrue(idx_tr.isdisjoint(idx_te))
        self.assertTrue(idx_va.isdisjoint(idx_te))
        self.assertAlmostEqual(y_train.mean(), y_test.mean(), delta=0.005)

    def test_preprocessor_fits_only_on_train(self):
        """验证预处理中位数仅使用训练集，不受验证/测试集污染。"""
        df, _ = load_and_clean_data(DATA_RAW_PATH, save_processed=False)
        X_train, _, X_test, _, _, _ = split_train_valid_test(df)
        pre = build_preprocessor(X_train)
        pre.fit(X_train)
        train_medians = pre.named_transformers_["num"].named_steps["impute"].statistics_
        expected_total_med = float(X_train["TotalCharges"].median())
        test_total_med = float(X_test["TotalCharges"].median())
        self.assertAlmostEqual(float(train_medians[2]), expected_total_med, places=6)
        self.assertNotAlmostEqual(expected_total_med, test_total_med, places=2)


class TestStatsAndBusinessLogic(unittest.TestCase):
    def test_wilson_ci_and_bh_fdr(self):
        """验证 Wilson 95% CI 边界及 Benjamini-Hochberg 单调性。"""
        lo, hi = wilson_ci(0, 50)
        self.assertAlmostEqual(lo, 0.0, places=12)
        self.assertGreater(hi, 0.0)
        self.assertLess(hi, 0.15)

        q = benjamini_hochberg([0.001, 0.02, 0.80])
        self.assertLess(q[0], q[1])
        self.assertLessEqual(q[1], q[2])
        self.assertLessEqual(q[2], 1.0)

    def test_topk_and_budget_simulation(self):
        """验证 Top-K 与预算模拟的精确率、Lift 及净收益计算一致性。"""
        y_true = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
        prob = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        mc = np.full(10, 100.0)

        topk = evaluate_topk(y_true, prob, top_fractions=(0.20,))
        self.assertEqual(topk[0]["k_customers"], 2)
        self.assertEqual(topk[0]["tp_captured"], 2)
        self.assertAlmostEqual(topk[0]["precision_at_k"], 1.0)

        sim = simulate_budget_scenario(y_true, mc, prob, top_pct=20, contact_cost=10.0, assumed_save_rate=0.5)
        self.assertEqual(sim["contacted_customers"], 2)
        self.assertEqual(sim["captured_churners"], 2)
        self.assertAlmostEqual(sim["total_contact_cost"], 20.0)
        # 2 * 100 * 0.5 * 6 * 0.6 - 20 = 360 - 20 = 340
        self.assertAlmostEqual(sim["assumed_net_benefit"], 340.0)

    def test_ab_sample_size_monotonicity(self):
        """探测越小的相对提升需要越大的每组样本量。"""
        n_15 = compute_ab_test_sample_size(0.50, 0.15)
        n_25 = compute_ab_test_sample_size(0.50, 0.25)
        self.assertGreater(n_15, n_25)


class TestGeneratedArtifactsConsistency(unittest.TestCase):
    def test_reports_and_figures_exist_and_valid(self):
        """验证离线生成的报告、JSON、图表及 v1 历史基线均完备一致。"""
        self.assertTrue((LEGACY_V1_DIR / "model_comparison_v1.csv").exists())
        comp = pd.read_csv(REPORTS_DIR / "model_comparison.csv")
        self.assertGreaterEqual(len(comp), 4)
        self.assertTrue({"XGBoost", "Random Forest", "Logistic Regression"}.issubset(set(comp["model"])))

        for fname in (
            "eda_stats_summary.json",
            "model_cv_summary.json",
            "explainability_summary.json",
            "business_scenarios.json",
        ):
            p = REPORTS_DIR / fname
            self.assertTrue(p.exists(), f"缺少产出文件: {p}")
            data = json.loads(p.read_text(encoding="utf-8"))
            self.assertIsInstance(data, dict)

        for fig_name in (
            "01_churn_overview.png",
            "02_cross_segments_heatmap.png",
            "03_model_evaluation_curves.png",
            "04_confusion_and_topk.png",
            "05_feature_explainability.png",
            "06_budget_scenarios.png",
        ):
            fig_path = FIGURES_DIR / fig_name
            self.assertTrue(fig_path.exists() and fig_path.stat().st_size > 10_000, f"图表异常: {fig_path}")


if __name__ == "__main__":
    unittest.main()
