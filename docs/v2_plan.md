# Telco v2 升级计划（XGBoost 调参与特征重要性）

基于 GitHub Issue #1 与 v1 基线（`reports/business_report.md` 及 `notebooks/01_telco_churn.ipynb`），制定 **Telco v2 升级计划** 如下。

---

## 一、 基线回顾与核心约束

1. **v1 基线表现（测试集 $n=1,409$，正例率 $26.5\%$）**
   - `Logistic Regression`：ROC-AUC `0.832` | PR-AUC `0.622` | Precision `0.614` | Recall `0.532` | F1 `0.570`（验证集阈值 `0.495`）
   - `Random Forest`：ROC-AUC `0.806` | PR-AUC `0.591` | Precision `0.521` | Recall `0.698` | F1 `0.597`（验证集阈值 `0.321`）
   - `XGBoost`（v1 默认参数 `n_estimators=300`）：ROC-AUC `0.786` | PR-AUC `0.558` | Precision `0.484` | Recall `0.676` | F1 `0.564`（验证集阈值 `0.170`）
   - **症结诊断**：v1 中 `XGBoost` 采用较深的默认树结构（`max_depth=6`）与较高学习率（`learning_rate=0.3`）且无正则化约束，在 4,225 行的中小型表格数据集上过拟合明显，导致排序指标（ROC-AUC / PR-AUC）与分类指标（F1）均落后于线性模型与随机森林。

2. **严格约束与文件隔离规范**
   - **数据划分完全一致**：严格沿用 `notebooks/01_telco_churn.ipynb` 的两阶段分层切分（`random_state=42`，`stratify=y`），保持训练集 60%（4,225 行）、验证集 20%（1,409 行）、测试集 20%（1,409 行）。
   - **预处理流水线完全一致**：排除 `customerID`，沿用相同的 `ColumnTransformer`（数值特征 `SimpleImputer(strategy='median')` + `StandardScaler()`；类别特征 `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore')`），且仅在训练集或交叉验证训练折内拟合，杜绝数据泄露。
   - **评估指标**：核心评估指标为 **ROC-AUC**、**PR-AUC（Average Precision）**、**F1**（辅以 Precision、Recall、混淆矩阵与预计触达人数 `contacted`）。
   - **不修改 v1 文件**：保留 `notebooks/01_telco_churn.ipynb`、`reports/business_report.md` 及 v1 生成的图表与 CSV 原样不动；v2 产出独立存放于新文件（如 `notebooks/02_telco_churn_v2_xgboost.ipynb`、`reports/business_report_v2.md`、`reports/model_comparison_v2.csv` 及 `reports/figures/v2_*.png`）。

---

## 二、 XGBoost 超参数调优方案

### 1. 搜索空间设计（针对中小样本高维独热特征的抗过拟合策略）
聚焦控制树复杂度、步长收缩、行列采样与正则化强度：

| 参数类别 | 超参数名称 | 候选搜索范围 | 调优目的与物理意义 |
| :--- | :--- | :--- | :--- |
| **树结构复杂度** | `max_depth` | `[2, 3, 4, 5, 6]` | 限制树深，防止对独热编码稀疏特征产生高阶噪声交互 |
| | `min_child_weight` | `[1, 3, 5, 10, 15]` | 提升叶节点最小样本海森权重和，抑制小样本叶节点分裂 |
| **步长与迭代** | `learning_rate` (`eta`) | `[0.01, 0.03, 0.05, 0.1, 0.2]` | 降低学习率，配合子树累加平滑预测概率分布 |
| | `n_estimators` | `[100, 200, 300, 500, 800]` | 配合低学习率与早停机制确定最优树数量 |
| **随机子采样** | `subsample` | `[0.6, 0.7, 0.8, 0.9, 1.0]` | 行采样引入随机性，降低方差 |
| | `colsample_bytree` | `[0.5, 0.6, 0.7, 0.8, 1.0]` | 列采样避免单一强相关特征（如月付合同）垄断所有树分裂 |
| **分裂与正则化** | `gamma` (`min_split_loss`) | `[0, 0.1, 0.5, 1.0, 2.0]` | 设定节点分裂所需的最小损失下降阈值（后剪枝效果） |
| | `reg_alpha` (L1) | `[0, 0.01, 0.1, 1.0, 5.0]` | 稀疏化叶节点权重，适应独热编码后的高维特征空间 |
| | `reg_lambda` (L2) | `[0.5, 1.0, 2.0, 5.0, 10.0]` | 平滑叶节点输出权重，防止预测概率极端化（改善校准度与阈值稳定性） |
| **类别不平衡** | `scale_pos_weight` | `[1.0, 1.5, 2.77]` | 对照默认权重（`1.0`）与正负样本比（$\approx 73.5/26.5 \approx 2.77$）对 PR-AUC 与 F1 的影响 |

### 2. 验证与选型方法（严格防泄露双层设计）
- **超参数选择阶段（基于训练/验证集，严禁触碰测试集）**：
  - **主方案（与 v1 留出验证集对齐）**：在 `X_train`（60%）上拟合预处理与不同超参数组合，在独立的 `X_valid`（20%）上评估 **PR-AUC（Average Precision）** 作为第一主目标（兼顾 ROC-AUC），并利用 `X_valid` 配合 `early_stopping_rounds=30`（监控验证集 `aucpr` / `logloss`）防止过拟合；或采用在 `X_train` 内部进行 5 折分层交叉验证（`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`，将 `Pipeline` 整体放入 `RandomizedSearchCV`）选出超参数后，统一在 `X_valid` 上验证排序表现。
  - **两步搜索策略**：先通过 `RandomizedSearchCV`（固定 `random_state=42`，迭代 60~100 组参数）快速定位高潜力参数区间，再围绕 Top 参数组合做局部细粒度网格搜索（`GridSearchCV`）。
- **分类阈值确定阶段**：
  - 与 v1 完全一致：锁定最优超参数的 `XGBoost (Tuned)` 模型后，在验证集 `X_valid` 上根据预测概率计算 `precision_recall_curve`，选择使验证集 **F1 最大化** 的分类阈值 `validation_threshold`。
- **最终盲测阶段**：
  - 将锁定超参数与验证集阈值的模型应用于测试集 `X_test`（20%），一次性产出测试集 ROC-AUC、PR-AUC、Precision、Recall、F1、混淆矩阵及触达人数，并与 v1 的三个基线模型（`Logistic Regression`、`Random Forest`、`XGBoost v1 Default`）并列对比。

---

## 三、 特征重要性分析方案

为避免单一特征重要性指标的偏倚（如基尼/分裂次数偏向连续变量或高基数特征），采用**多维度互补分析法**，并确保与业务解释严格对齐：

1. **多口径重要性度量对比**
   - **XGBoost 内置重要性（Gain / Weight / Cover）**：重点提取 `importance_type='gain'`（平均信息增益贡献），辅以 `weight`（分裂频次）与 `cover`（覆盖样本量）进行交叉核对，识别模型分裂时最依赖的特征。
   - **置换重要性（Permutation Importance）**：在验证集/测试集上使用 `sklearn.inspection.permutation_importance`（以 `average_precision` 和 `roc_auc` 为评分准则，重复打乱 $n=30$ 次，固定随机种子）。直接衡量**原始特征**（在进入 `ColumnTransformer` 前打乱，或对编码后特征聚合）对模型泛化排序能力的真实贡献及置信区间。
   - **SHAP 值分析（或树模型局部可解释性对比，如环境支持）**：提取全局平均绝对贡献度（Mean $|SHAP|$）及特征正负作用方向，区分特征是“推高流失风险”还是“降低流失风险”。

2. **特征层级聚合与跨模型一致性校验**
   - **独热特征映射与汇总**：从 `ColumnTransformer` 的 `get_feature_names_out()` 提取编码后特征名（如 `cat__Contract_Month-to-month`、`cat__InternetService_Fiber optic`、`num__tenure`），同时提供**细粒度编码特征 Top 15** 与**按原始业务字段聚合的重要性排名**。
   - **跨模型横向互证**：将调优后 `XGBoost` 的 Top 特征与 `Logistic Regression`（标准化回归系数）及 `Random Forest`（置换重要性）进行对比，检验模型发现的驱动因子是否与 v1 探索性分析（EDA）中的高流失群体（月付合同、0-12个月新客、光纤网络服务、电子支票等）一致。

3. **可视化与相关性边界声明**
   - 输出验证/测试集 Permutation Importance（含误差棒）与 XGBoost Gain Importance 双联对比图（`reports/figures/v2_05_feature_importance.png`）。
   - **方法论约束**：明确标注特征重要性仅反映**预测关联强度与模型依赖度**，受共线性影响（如 `tenure`、`MonthlyCharges` 与 `TotalCharges` 高度共线，或网龄与合同期重叠），**绝不等同于业务因果效应**。

---

## 四、 报告润色要点（`reports/business_report_v2.md`）

在延续 v1 报告严谨克制、数据实证风格的基础上，重点升级以下四个部分：

1. **模型评估与调优前后对比重构**
   - 增设 **“XGBoost 调优前后及多模型对比表”**（包含 `Logistic Regression`、`Random Forest`、`XGBoost (v1 Default)`、`XGBoost (v2 Tuned)` 四行），完整披露验证集选定阈值、ROC-AUC、PR-AUC、精确率、召回率、F1 及预计触达客户数（`contacted`）。
   - 补充调优增益解读：说明通过降低树深、收缩学习率与引入正则化，如何修正了 v1 默认参数的过拟合问题，以及验证集阈值变化对精确率/召回率权衡的具体影响。
2. **新增“核心预测特征与驱动因素分析”章节**
   - 将特征重要性图表与业务分群联动解读：区分**结构性/门槛特征**（如合同类型 `Contract`、在网时长 `tenure`）与**体验/服务特征**（如光纤网络 `InternetService`、无在线安全/技术支持附加服务 `OnlineSecurity`/`TechSupport`、月费水平 `MonthlyCharges`）。
   - 解释共线性对特征排名的影响（例如 `TotalCharges` 为月费与在网时长的乘积近似，模型间权重分配可能存在替代效应）。
3. **深化业务落地建议与运营可执行性**
   - **从“按 F1 演示阈值”走向“容量与成本约束决策”**：结合测试集 `contacted`（触达人数）、`tp`（命中流失数）与 `fp`（误报打扰数），在报告中提供基于不同外呼预算/触达容量（例如 Top 10%、Top 20%、Top 30% 高危人群）的分档覆盖率参考框架。
   - **分层干预实验设计**：结合特征重要性，针对“新客月付+光纤且无技术支持”等高危交集细分群提出差异化留存假设（如首年体验排查、年约转换激励），并重申需通过随机对照试验（RCT / Uplift 实验）区分“自然高流失”与“可挽回客户”。
4. **严谨性、复现性与局限性规范**
   - 明确声明所有报告数值 100% 源自 Notebook 实际运行输出（`reports/model_comparison_v2.csv`），无人工估算。
   - 保持并强化 v1 的风险提示：静态截面数据缺乏时间戳与观察窗口、存在特征时点可用性风险、单次随机划分存在抽样波动，上线前须补充时间外验证（Out-of-Time Validation）与模型校准度检查。

---

## 五、 里程碑与推进时间表（截止日期：2026-11-20）

当前日期为 **2026-11-06**，距离截止日共 14 天，分四个阶段推进：

| 阶段 | 时间节点 | 核心任务 | 交付物（独立新文件，不改动 v1） |
| :--- | :--- | :--- | :--- |
| **M1：环境复核与调优框架搭建** | 11-06 ~ 11-09 | 复核 v1 数据切分与预处理复现性；搭建 v2 Notebook 框架，配置 XGBoost 搜索空间与防泄露验证流程 | `notebooks/02_telco_churn_v2_xgboost.ipynb`（初稿框架） |
| **M2：超参数搜索与评估实验** | 11-10 ~ 11-13 | 执行超参数搜索与早停验证；锁定最优模型并在验证集选择 F1 阈值；完成测试集盲测及四模型指标汇总与曲线绘制 | `reports/model_comparison_v2.csv`<br>`reports/figures/v2_04_model_curves.png` |
| **M3：特征重要性多维分析** | 11-14 ~ 11-16 | 计算 Gain / Permutation Importance（及可选 SHAP）；完成独热特征向原始字段的聚合映射与跨模型对比可视化 | `reports/feature_importance_v2.csv`<br>`reports/figures/v2_05_feature_importance.png` |
| **M4：报告撰写、润色与终审交付** | 11-17 ~ 11-20 | 撰写并润色 v2 业务报告；交叉核验所有正文数字、表格与图表完全一致；验证 `git status` 确保 v1 文件零修改 | `reports/business_report_v2.md`<br>最终全量复现校验通过 |
