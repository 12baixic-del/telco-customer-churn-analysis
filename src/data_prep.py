"""数据清洗、质量审计与严格分层划分模块。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    DATA_PROCESSED_PATH,
    DATA_RAW_PATH,
    ID_COL,
    NUMERIC_COLS,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE_TEMP,
    TEST_SPLIT_WITHIN_TEMP,
)


def load_and_clean_data(raw_path: Path = DATA_RAW_PATH, save_processed: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """读取原始 Telco CSV，执行数据质量审计并输出清洗后的 DataFrame 与审计报告。"""
    raw = pd.read_csv(raw_path)
    df = raw.copy()

    # 去除字符串前后空白并将空字符串统一转为 NaN
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)

    # 显式保留 SeniorCitizen 的业务可读标签副本（原始为 0/1 整数）
    original_total = df["TotalCharges"].copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 统一数值列类型
    df["tenure"] = pd.to_numeric(df["tenure"], errors="raise").astype("int64")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="raise").astype("float64")
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="raise").astype("int64")

    # 质量审计统计
    blank_total_mask = df["TotalCharges"].isna()
    blank_total_tenures = df.loc[blank_total_mask, "tenure"].value_counts().to_dict()
    blank_total_churn = df.loc[blank_total_mask, "Churn"].value_counts().to_dict()

    audit: dict[str, Any] = {
        "raw_rows": int(len(raw)),
        "raw_cols": int(raw.shape[1]),
        "duplicate_full_rows": int(df.duplicated().sum()),
        "duplicate_customer_ids": int(df[ID_COL].duplicated().sum()),
        "missing_customer_ids": int(df[ID_COL].isna().sum()),
        "invalid_nonblank_total_charges": int((original_total.notna() & df["TotalCharges"].isna()).sum()),
        "missing_counts": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
        "blank_total_charges_tenures": {int(k): int(v) for k, v in blank_total_tenures.items()},
        "blank_total_charges_churn": {str(k): int(v) for k, v in blank_total_churn.items()},
        "tenure_min": int(df["tenure"].min()),
        "tenure_max": int(df["tenure"].max()),
        "monthly_charges_min": float(df["MonthlyCharges"].min()),
        "monthly_charges_max": float(df["MonthlyCharges"].max()),
        "overall_churn_count": int((df[TARGET_COL] == "Yes").sum()),
        "overall_churn_rate": float((df[TARGET_COL] == "Yes").mean()),
    }

    assert df[ID_COL].notna().all() and df[ID_COL].is_unique, "customerID 必须非空且唯一"
    assert df[TARGET_COL].isin(["Yes", "No"]).all(), "Churn 只能包含 Yes/No"
    assert (df["tenure"] >= 0).all() and (df["MonthlyCharges"] > 0).all(), "数值字段存在异常越界"

    if save_processed:
        DATA_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PROCESSED_PATH, index=False)

    return df, audit


def split_train_valid_test(
    df: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """严格分层划分 60% 训练集 / 20% 验证集 / 20% 测试集（不修改原始列，不进行全量拟合）。"""
    X = df.drop(columns=[ID_COL, TARGET_COL]).copy()
    # SeniorCitizen 转为类别语义，避免被当作连续数值缩放
    X["SeniorCitizen"] = X["SeniorCitizen"].map({0: "No", 1: "Yes"})
    y = df[TARGET_COL].eq("Yes").astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE_TEMP, stratify=y, random_state=random_state
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=TEST_SPLIT_WITHIN_TEMP, stratify=y_temp, random_state=random_state
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def get_feature_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """返回数值特征列名与类别特征列名。"""
    num_cols = [c for c in NUMERIC_COLS if c in X.columns]
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols
