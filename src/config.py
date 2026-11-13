"""全局配置与路径管理模块。"""

from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE_TEMP = 0.40  # 60% train, 40% temp (valid 20% + test 20%)
TEST_SPLIT_WITHIN_TEMP = 0.50

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_PATH = ROOT_DIR / "data" / "raw" / "Telco-Customer-Churn.csv"
DATA_PROCESSED_PATH = ROOT_DIR / "data" / "processed" / "telco_cleaned.csv"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LEGACY_V1_DIR = REPORTS_DIR / "legacy_v1"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
ID_COL = "customerID"
TARGET_COL = "Churn"

# 关键业务分组变量（用于第二阶段统计检验与分群）
KEY_CATEGORICAL_VARS = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "PaperlessBilling",
    "OnlineSecurity",
    "TechSupport",
    "SeniorCitizen",
    "Partner",
    "Dependents",
]
