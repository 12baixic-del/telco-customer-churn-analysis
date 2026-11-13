"""Telco Customer Churn v2 交互式作品集展示页（直接读取 reports/ 预生成结果）。

本地运行：
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

st.set_page_config(page_title="Telco Churn v2 Portfolio", layout="wide")
st.title("电信客户流失分析与留存资源分配（v2 作品集展示）")
st.caption("注：本页面直接加载已验证的离线计算产出（reports/），不在演示时重训模型。")

eda = json.loads((REPORTS / "eda_stats_summary.json").read_text(encoding="utf-8"))
cv_sum = json.loads((REPORTS / "model_cv_summary.json").read_text(encoding="utf-8"))
exp = json.loads((REPORTS / "explainability_summary.json").read_text(encoding="utf-8"))
biz = json.loads((REPORTS / "business_scenarios.json").read_text(encoding="utf-8"))
comp_df = pd.read_csv(REPORTS / "model_comparison.csv")

tab1, tab2, tab3, tab4 = st.tabs(["1. 业务探索与统计检验", "2. 模型对比与校准", "3. 可解释性与典型画像", "4. 预算情景与 A/B 测试"])

with tab1:
    ov = eda["overall"]
    c1, c2, c3 = st.columns(3)
    c1.metric("总客户数", f"{ov['customers']:,}")
    c2.metric("流失客户数", f"{ov['churned']:,}")
    c3.metric("整体流失率 (95% Wilson CI)", f"{ov['churn_rate']:.1%}", f"[{ov['ci_low']:.1%}, {ov['ci_high']:.1%}]")
    st.image(str(FIGURES / "01_churn_overview.png"), use_container_width=True)
    st.image(str(FIGURES / "02_cross_segments_heatmap.png"), use_container_width=True)

with tab2:
    st.subheader("测试集模型指标对比（n=1,409，严格隔离）")
    st.dataframe(comp_df.round(4), use_container_width=True)
    st.image(str(FIGURES / "03_model_evaluation_curves.png"), use_container_width=True)
    st.image(str(FIGURES / "04_confusion_and_topk.png"), use_container_width=True)

with tab3:
    st.subheader("XGBoost 原生 TreeSHAP 与 Logistic Regression 方向性")
    st.image(str(FIGURES / "05_feature_explainability.png"), use_container_width=True)
    st.info(exp["case_studies"]["collinearity_note"]["explanation"])
    st.subheader("典型高风险客户匿名画像（TreeSHAP 局部归因）")
    st.json(exp["case_studies"]["high_risk_profiles"][0])

with tab4:
    st.warning(biz["disclaimer"])
    st.subheader("风险分层与触达预算情景模拟")
    st.dataframe(pd.DataFrame(biz["risk_tiers"]).round(4), use_container_width=True)

    col_a, col_b = st.columns(2)
    sel_cost = col_a.selectbox("单客触达成本 ($)", [10.0, 20.0, 30.0], index=1)
    sel_save = col_b.selectbox("假设挽回成功率", [0.15, 0.25, 0.35], index=1)
    sc_df = pd.DataFrame(biz["scenario_grid"])
    filtered = sc_df[(sc_df["contact_cost_per_user"] == sel_cost) & (sc_df["assumed_save_rate"] == sel_save)]
    st.dataframe(filtered[["top_pct", "contacted_customers", "captured_churners", "precision_at_k", "total_contact_cost", "assumed_saved_customers", "assumed_net_benefit"]].round(2), use_container_width=True)
    st.image(str(FIGURES / "06_budget_scenarios.png"), use_container_width=True)
