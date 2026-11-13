# 电信客户流失预测 — 建模与评估技术报告（MODEL_REPORT）

## 1. 数据划分与防泄漏设计

- **数据规模**：`7,043` 行、`19` 个有效输入特征（移除 `customerID` 与目标列 `Churn`，并将 `SeniorCitizen` 显式视为二元类别变量而非连续数值）。
- **严格分层划分**（`RANDOM_STATE = 42`）：
  - **训练集（60%，`4,225` 人）**：仅在此集合上拟合 `SimpleImputer(median/most_frequent)`、`StandardScaler`、`OneHotEncoder`，并执行 5 折分层交叉验证（`StratifiedKFold(5)`）与 `GridSearchCV` 超参数搜索。
  - **验证集（20%，`1,409` 人）**：用于对比类别不平衡加权策略、比较原始概率与 Sigmoid 校准的 Brier Score、以及挑选 F1 最大化演示阈值。
  - **测试集（20%，`1,409` 人）**：全流程隔离，仅在模型锁定后做**一次性最终评估**与 Bootstrap 不确定性分析。

---

## 2. v1 与 v2 模型对比（同一测试集 `n = 1,409`）

| 模型 | v1 ROC-AUC | **v2 ROC-AUC (95% Bootstrap CI)** | v1 PR-AUC | **v2 PR-AUC (95% Bootstrap CI)** | v2 Brier Score | v2 精确率 / 召回率 / F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | 0.786 | **0.8362** `[0.814, 0.858]` | 0.558 | **0.6442** `[0.601, 0.686]` | **0.1392** | `0.585 / 0.615 / 0.600` |
| **Random Forest** | 0.806 | **0.8323** `[0.810, 0.854]` | 0.591 | **0.6283** `[0.583, 0.675]` | 0.1407 | `0.549 / 0.733 / 0.628` |
| **Logistic Regression** | 0.832 | **0.8324** `[0.809, 0.854]` | 0.622 | **0.6221** `[0.578, 0.668]` | 0.1414 | `0.614 / 0.532 / 0.570` |
| Dummy Baseline (Stratified) | — | 0.4981 | — | 0.2647 | 0.3925 | `0.263 / 0.265 / 0.264` |

### 关键技术发现（客观对比）
1. **为什么 v2 XGBoost 提升最显著？**
   - v1 使用默认参数（`max_depth=6, learning_rate=0.3, n_estimators=300`），在 `4,225` 样本的中小数据集上严重过拟合，测试集 `PR-AUC` 仅 `0.558`。
   - v2 通过 5 折 CV 搜索到强正则化浅树组合（`max_depth=2, learning_rate=0.03, n_estimators=250, reg_lambda=5.0, subsample=0.8, colsample_bytree=0.8`），测试集 `PR-AUC` 提升至 **`0.6442`**（+0.086），`ROC-AUC` 提升至 **`0.8362`**（+0.050）。
2. **Logistic Regression 为什么依然具有极高性价比？**
   - 在 `C=1.0` 下，LR 测试集 `ROC-AUC = 0.8324` 与 XGBoost（`0.8362`）的 95% Bootstrap CI 高度重叠；在需要极简部署、强可解释性或法规合规审核场景下，LR 是极佳基线。

---

## 3. 类别不平衡处理消融实验（验证集实测）

在验证集对比默认权重 vs 平衡权重（`class_weight="balanced"` / `scale_pos_weight=2.77`）：

- **Logistic Regression**：默认 `valid_pr_auc = 0.6776`, `valid_brier = 0.1304` vs `balanced` `valid_pr_auc = 0.6773`, `valid_brier = 0.1640`。
- **Random Forest**：默认 `valid_pr_auc = 0.6725`, `valid_brier = 0.1320` vs `balanced_subsample` `valid_pr_auc = 0.6744`, `valid_brier = 0.1539`。
- **XGBoost**：默认 `valid_pr_auc = 0.6781`, `valid_brier = 0.1303` vs `scale_pos_weight=2.77` `valid_pr_auc = 0.6777`, `valid_brier = 0.1582`。

**实验结论**：由于原始正样本比例为 `26.5%`（并非 1:100 极度不平衡），**开启类别加权并未显著提升排序指标（PR-AUC/ROC-AUC），反而因人为推高预测概率导致 Brier Score 明显恶化（`0.130 -> 0.158~0.164`）**。因此，三个主力模型均保留原始概率权重，通过**调整业务决策阈值 / Top-K 排序**来满足不同召回或预算需求，而不是盲目加权扭曲概率校准。

---

## 4. 概率校准与不确定性评估

- **校准评估**：在验证集对比原始概率与 5 折 Sigmoid 校准后的 Brier Score：LR（`0.1303` vs `0.1305`）、RF（`0.1313` vs `0.1313`）、XGBoost（`0.1301` vs `0.1301`）。三模型在未加权下本身校准已接近最优（测试集 XGBoost Brier = `0.1392`），无需额外套壳校准。
- **Bootstrap 95% 置信区间**：在测试集（`n=1,409`）分层重采样 1,000 次，XGBoost `PR-AUC` 95% CI 为 `[0.601, 0.686]`，显著优于随机基线 `0.265`。

---

## 5. 可解释性与共线性偏差提示

1. **XGBoost 原生 TreeSHAP（原始字段聚合平均 $|\text{SHAP}|$）Top-5**：
   `Contract (0.666)` > `tenure (0.436)` > `MonthlyCharges (0.236)` > `OnlineSecurity (0.234)` > `InternetService (0.230)`。
2. **多重共线性偏差（面试关键点）**：
   测试集中 `tenure` 与 `TotalCharges` 的皮尔逊相关系数高达 `r` 正强相关（`TotalCharges ≈ tenure × MonthlyCharges`）。在逻辑回归中，`num__tenure` 系数为 `-1.257`（强势负向），而 `num__TotalCharges` 系数却反转为 `+0.577`（正向）。这属于典型的**共线性符号反转**，说明在单变量层面 `TotalCharges` 越小越易流失，但在控制 `tenure` 后它吸收了高月费交互残差，**绝不能将其误读为“消费越多越容易流失”**。

---

## 6. 最终模型选择建议

| 业务场景 | 推荐模型 | 依据 |
| :--- | :--- | :--- |
| **预算有限的 Top-K 精准排序触达** | **XGBoost (CV 调优)** | `PR-AUC (0.6442)`、`ROC-AUC (0.8362)` 与 `Brier (0.1392)` 均最优，Top 20% 精确率 `64.9%` (Lift `2.44x`) |
| **高覆盖优先（漏报代价极高）** | **Random Forest (CV 调优)** | 在验证集 F1 阈值下测试集 `Recall` 达 `73.3%`（`F1 = 0.6277`） |
| **强解释性 / 极简上线基线** | **Logistic Regression** | `ROC-AUC = 0.8324` 接近树模型，优势比（Odds Ratio）直观透明 |
