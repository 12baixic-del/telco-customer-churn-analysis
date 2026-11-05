# Telco Customer Churn

电信客户流失分析作品集：数据质量、业务分群和三个模型的评估。

- [业务报告](reports/business_report.md)（含已核实的探索图）
- [可运行 Notebook](notebooks/01_telco_churn.ipynb)

## 数据来源

使用 IBM 公开的 [Telco Customer Churn 示例数据集](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)，共 7,043 行、21 列；本地原始文件 SHA-256 为 `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`。数据与原始文件保留在仓库的 `data/` 目录；使用时请查阅 [IBM 原始项目](https://github.com/IBM/telco-customer-churn-on-icp4d)的许可和条款。

## 运行

在项目根目录启动 Jupyter，打开 Notebook 并选择 Run All：

```powershell
& 'C:\Users\13557\AppData\Local\Microsoft\WindowsApps\python.exe' -m jupyter notebook
```

需要 pandas、numpy、matplotlib、seaborn、scikit-learn、xgboost、jupyter。原始数据位于 data/raw/Telco-Customer-Churn.csv。运行后生成 data/processed/telco_cleaned.csv、reports/model_comparison.csv 和 reports/figures 下的 PNG。Notebook 将覆盖同名生成文件；原始数据不会被修改。

Notebook 已由用户在本机运行，报告中的模型指标来自该次运行的测试集结果。
