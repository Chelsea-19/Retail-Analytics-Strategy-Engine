<div align="center">
  <h1>Retail Analytics Strategy Engine</h1>
  <p>
    <b>
      <a href="#-english-version">🇺🇸 English Version</a> | 
      <a href="#-中文版">🇨🇳 中文版</a>
    </b>
  </p>
</div>

---

<div id="-english-version"></div>

## 🇺🇸 English Version

**Retail Analytics Strategy Engine** is a comprehensive data analysis solution designed to uncover customer value, predict churn risk, and improve cross-selling rates through machine learning algorithms. The project includes three core analysis modules: Customer Segmentation, Propensity Scoring, and Recommendation System.

### 📋 Key Features

#### 1. Customer Segmentation
- **File**: `scripts/customer_segmentation.py`
- **Algorithm**: K-Means Clustering + PCA (Principal Component Analysis)
- **Functionality**:
  - Based on RFM (Recency, Frequency, Monetary) model and channel preferences.
  - Automatically segments customers into **5 strategic groups** (n_clusters=5).
  - Generates profile reports to identify high-value customers and potential churners.

#### 2. Propensity Scoring & Churn Modeling
- **File**: `scripts/propensity_score.py`
- **Algorithm**: XGBoost Classifier + SHAP (Explainability) + SMOTE (Oversampling)
- **Functionality**:
  - **Churn Prediction**: Identifies customers at risk of leaving.
  - **Loyalty Prediction**: Discovers high-potential loyal customers.
  - Provides SHAP value analysis to explain key drivers (e.g., recency, total spend).

#### 3. Recommendation Engine
- **File**: `scripts/recommendation.py`
- **Algorithm**: Apriori (Market Basket Analysis)
- **Functionality**:
  - Mines strong Association Rules between products.
  - Generates "bundle suggestions" based on Support, Confidence, and Lift.
  - Ideal for designing Cross-sell and Bundling strategies (e.g., QLab Tea case).

### 🛠️ Tech Stack
- **Data Processing**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`, `xgboost`, `mlxtend`, `imbalanced-learn`
- **Visualization**: `matplotlib`, `seaborn`, `shap`

<br>
<div align="right">
  <a href="#-中文版">Go to Chinese Version 🇨🇳</a>
</div>
<br>

---

<div id="-中文版"></div>

## 🇨🇳 中文版

**零售分析策略引擎 (Retail Analytics Strategy Engine)** 这是一个综合性的零售数据分析解决方案，旨在通过机器学习算法挖掘客户价值、预测流失风险并提升交叉销售率。项目包含三个核心分析模块：客户分群、倾向性评分和商品推荐系统。

### 📋 核心功能 (Key Features)

#### 1. 客户分群 (Customer Segmentation)
- **文件**: `scripts/customer_segmentation.py`
- **算法**: K-Means Clustering + PCA (主成分分析)
- **功能**:
  - 基于 RFM (Recency, Frequency, Monetary) 模型及渠道偏好特征。
  - 自动将客户划分为 **5 个战略群体** (n_clusters=5)。
  - 生成分群画像报告，识别高价值客户与潜在流失客户。

#### 2. 倾向性评分与流失预测 (Propensity & Churn Modeling)
- **文件**: `scripts/propensity_score.py`
- **算法**: XGBoost Classifier + SHAP (可解释性分析) + SMOTE (过采样)
- **功能**:
  - **流失预测 (Churn Risk)**: 识别即将流失的客户。
  - **忠诚度预测 (Loyalty)**: 挖掘高潜力的忠诚客户。
  - 提供 SHAP 值分析，解释影响模型决策的关键因子（如：最近一次购买时间、消费金额等）。

#### 3. 智能推荐引擎 (Recommendation Engine)
- **文件**: `scripts/recommendation.py`
- **算法**: Apriori (购物篮分析 / Market Basket Analysis)
- **功能**:
  - 挖掘商品间的强关联规则 (Association Rules)。
  - 基于支持度 (Support)、置信度 (Confidence) 和提升度 (Lift) 生成“购买组合建议”。
  - 适用于设计交叉销售 (Cross-sell) 和捆绑销售 (Bundling) 策略（例如：QLab Tea 案例）。

### 🛠️ 技术栈 (Tech Stack)
- **数据处理**: `pandas`, `numpy`
- **机器学习**: `scikit-learn`, `xgboost`, `mlxtend`, `imbalanced-learn`
- **可视化**: `matplotlib`, `seaborn`, `shap`

<br>
<div align="right">
  <a href="#-english-version">回到英文版本 🇺🇸</a>
</div>

## Executive Summary
A leading tea retailer in Southeast Asia (Singapore, Malaysia, Hong Kong) faces increasing customer acquisition costs and stagnant Average Order Value (AOV). 

This project implements a **CRISP-DM** based analytical framework to solve two critical business problems:
1.  **Who is leaving?** (Churn Prediction & Prevention)
2.  **What else will they buy?** (Basket Optimization & Cross-selling)

**Compliance Note:** To adhere to Data Privacy regulations (GDPR/PDPA) and Non-Disclosure Agreements (NDA), the original proprietary dataset has been replaced. This repository utilizes a custom **Synthetic Data Generator (`data_generator.py`)** that statistically replicates the client's data distribution, seasonal patterns, and churn behaviors without exposing sensitive information.

---

## Project Architecture

The solution is divided into three strategic modules:

| Module | Business Goal | Technical Approach |
| :--- | :--- | :--- |
| **1. Data Simulation** | Ensure compliance & reproducibility. | `Faker`, `NumPy` for probabilistic generation of Demographics & Transactions. |
| **2. Customer Segmentation** | Identify high-value personas. | **K-Means Clustering** + RFM Analysis (Recency, Frequency, Monetary). |
| **3. Churn Prediction** | Proactive retention. | **XGBoost Classifier** + **SMOTE** (Imbalance handling) + **SHAP** (Explainability). |
| **4. Market Basket Analysis** | Increase wallet share (AOV). | **Apriori Algorithm** for Association Rule Mining. |

---

## Key Insights & Business Translation

### 1. The "Why" Behind Churn (Explainable AI)
Using SHAP (SHapley Additive exPlanations), we deconstructed the black-box model to understand driver behaviors.

![SHAP Summary](results/shap_feature_importance.png)
*(Note: Visual based on mock data replication)*

* **Insight:** `Recency` (days since last purchase) is the #1 predictor of churn. The risk creates a "hockey stick" curve after **60 days** of inactivity.
* **Action:** Implement an automated "We Miss You" email trigger specifically at **Day 45** (preventative) and **Day 60** (reactive) with a time-bound discount.

### 2. Strategic Product Bundling
By analyzing co-occurrence patterns in transaction history:

![Correlation Heatmap](results/product_cross_sell_matrix.png)

* **Insight:** Strong correlation found between **"Premium Pu-erh"** and **"Tea Accessories"**.
* **Action:** Create a "Gong Fu Cha Starter Set" bundle. When a user adds Pu-erh to the cart, recommend the accessory kit for 15% off.

### 3. Customer Personas (Clustering)
* **"Loyal Connoisseurs":** High Frequency, High Monetary, Low Recency. -> *Strategy: Exclusive tastings (Gold Tier).*
* **"Dormant Big Spenders":** Low Frequency, High Monetary, High Recency. -> *Strategy: High-value win-back campaigns.*

---

## Repository Structure

```bash
Retail-Analytics-Strategy-Engine/
├── data/                     # Folder for generated CSVs (excluded from git)
├── scripts/
│   ├── data_generator.py     # Simulates realistic retail data (Privacy Safe)
│   ├── segmentation.py       # RFM Analysis & K-Means Clustering
│   ├── churn_prediction.py   # XGBoost Training & SHAP Interpretation
│   └── market_basket.py      # Association Rules (Apriori) engine
├── results/
│   ├── shap_feature_importance.png
│   ├── product_cross_sell_matrix.png
│   └── k_selection_report.csv
├── requirements.txt
└── README.md
```

## Getting Started

Follow these instructions to set up the environment and replicate the analysis.

### 1. Prerequisites
Ensure you have Python 3.8+ installed. The project relies on the following key libraries:
* `pandas` & `numpy` (Data Manipulation)
* `scikit-learn` & `xgboost` (Modeling)
* `shap` (Model Interpretability)
* `mlxtend` (Association Rules)
* `faker` (Data Simulation)

### 2. Installation
Clone the repository and install dependencies:

```bash
git clone [https://github.com/YourUsername/Retail-Analytics-Strategy-Engine.git](https://github.com/YourUsername/Retail-Analytics-Strategy-Engine.git)
cd Retail-Analytics-Strategy-Engine

# It is recommended to create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Usage Guide

**Step 1: Generate Synthetic Data**
First, run the generator to create the mock datasets. This ensures you have the necessary CSV files (`mock_customer_profile.csv` and `mock_transaction_history.csv`) in your local environment.

```bash
python scripts/data_generator.py
```
Output: Files generated in the project root or specified data folder.

**Step 2: Run Customer Segmentation**
Identify customer clusters based on purchasing behavior.

```bash
python scripts/segmentation.py
```
Output: Generates `customer_with_segments.csv` and K-Selection reports.

**Step 3: Train Churn Model & View Explainability**
Train the XGBoost model and generate SHAP plots to understand churn drivers.

```bash
python scripts/churn_prediction.py
```
Output: Displays/Saves SHAP summary plots and prints model accuracy metrics.

**Step 4: Generate Cross-sell Rules**
Run the Market Basket Analysis to find product associations.

```bash
python scripts/market_basket.py
```
Output: Prints top association rules (e.g., "If Buy A -> Then Buy B") and saves the correlation heatmap.

---

## Future Roadmap (Scalability)

To transition this project from a prototype to a production-ready system, the following improvements are proposed based on the **Data Science Lifecycle**:

### 1. Model Deployment (MLOps)
* **Current State:** Scripts run locally via CLI.
* **Future State:** Wrap the XGBoost model in a **FastAPI** microservice. Build a **Streamlit** dashboard for the Marketing Team to input a Customer ID and get real-time "Churn Risk Score" and "Next Best Action".

### 2. Advanced Metrics (CLV)
* **Current State:** Focus on Churn Probability.
* **Future State:** Implement **Customer Lifetime Value (CLV)** prediction using the *BG/NBD model (Lifetimes library)*. This will allow the business to prioritize retention budget on high-value customers rather than just high-risk ones.

### 3. Automated Pipeline
* **Current State:** Manual execution.
* **Future State:** Use **Apache Airflow** to schedule weekly retraining of the model as new transaction data arrives, ensuring the model adapts to changing market trends (Concept Drift).

---

## Contribution

Contributions are welcome! This project follows the standard "Fork & Pull" open-source model.

1.  **Fork** the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a **Pull Request**

---

## License

* **License:** Distributed under the MIT License. See `LICENSE` for more information.

---

*If you found this project useful for understanding Retail Analytics, please give it a ⭐️!*
