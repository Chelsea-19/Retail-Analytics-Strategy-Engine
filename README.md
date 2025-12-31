# Retail-Analytics-Strategy-Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Privacy%20Compliant-green?style=for-the-badge)

> **Objective:** Transforming raw transactional data into actionable retention and cross-sell strategies.

## Executive Summary
A leading tea retailer in Southeast Asia (Singapore, Malaysia, Hong Kong) faces increasing customer acquisition costs and stagnant Average Order Value (AOV). 

This project implements a **CRISP-DM** based analytical framework to solve two critical business problems:
1.  **Who is leaving?** (Churn Prediction & Prevention)
2.  **What else will they buy?** (Basket Optimization & Cross-selling)

**Compliance Note:** To adhere to Data Privacy regulations (GDPR/PDPA) and Non-Disclosure Agreements (NDA), the original proprietary dataset has been replaced. This repository utilizes a custom **Synthetic Data Generator (`data_generator.py`)** that statistically replicates the client's data distribution, seasonal patterns, and churn behaviors without exposing sensitive information.

---

## Analytical Modules

### 1. Customer Segmentation Engine
* **Objective:** Identify distinct customer groups and personas to tailor marketing efforts.

* **Feature Engineering:**
    * **RFM Analysis:** Recency, Frequency, Monetary value.
    * **O2O Identification:** Automatically detects offline transactions (keyword: "Funan") to distinguish physical store behavior from online activity.
    * **Log Transformation:** Handles skewed monetary data for better clustering performance.
* **Visualization:** Produces a **Strategic Value Matrix** (Consulting-style Bubble Chart) that maps "Recency" against "Monetary Value" to visually identify segments like "Champions" or "At Risk".

### 2. Propensity Prediction Engine
* **Objective:** Proactively manage customer lifecycle by predicting future states.

* **Modeling Approach:**
    * Trains two separate **XGBoost Classifiers**: one for **Churn Risk** and one for **Loyalty Potential**.
    * Utilizes **SMOTE** (Synthetic Minority Over-sampling Technique) within a pipeline to handle class imbalance.
* **Strategic Output:**
    * Generates an **Action Matrix** segmenting customers into actionable categories (e.g., *VIP Rescue*, *Upsell/Growth*) to guide retention strategies.

### 3. Recommendation System
* **Objective:** Increase Average Order Value (AOV) through data-driven product bundling and cross-selling.

* **Methodology:**
    * **Market Basket Analysis:** Implements the **Apriori algorithm** to find frequent itemsets.
    * **Churn Filtering:** Excludes churned customers to ensure recommendations reflect active user preferences.
* **Output:**
    * Identifies strong association rules (Lift > 1.0).
    * Auto-generates marketing copy for bundles (e.g., *"Customers who bought Oolong also loved Premium Pu-erh"*).

---

## Project Architecture

The solution is divided into three strategic modules:

| Module | Business Goal | Technical Approach |
| :--- | :--- | :--- |
| **1. Data Simulation** | Ensure compliance & reproducibility. | `Faker`, `NumPy` for probabilistic generation of Demographics & Transactions. |
| **2. Customer Segmentation** | Identify high-value personas. | **K-Means Clustering** + RFM Analysis (Recency, Frequency, Monetary). |
| **3. Churn Prediction** | Proactive retention. | **XGBoost Classifier** + **SMOTE** (Imbalance handling) + **SHAP** (Explainability). |
| **4. Market Basket Analysis** | Increase wallet share (AOV). | **Apriori Algorithm** for Association Rule Mining. |

## Key Insights & Business Translation

### 1. Strategic Portfolio Management (The "3A" Framework)
Instead of a "one-size-fits-all" marketing approach, we applied the **Propensity Engine** to map the entire customer base onto a risk-value plane. This visualizes our **"Anticipate"** strategy.

![Strategic Customer Portfolio Matrix](bubble_figure.png)

* **VIP Rescue (High Risk, High Value):** Identified top-tier customers (e.g., **Customer #918**, LTV $1,516) facing a **77% churn risk**.
    * *Action:* Immediate concierge outreach and unconditional coupons, as their retention value outweighs the cost.
* **Let Go (High Risk, Low Value):** Identified segments (e.g., **Customer #621**) where the cost of retention exceeds future value .
    * *Action:* Cease ad spend and reallocate budget to "Upsell" segments.

### 2. The "Why" Behind Churn (Explainable AI)
Using **SHAP (SHapley Additive exPlanations)**, we deconstructed the model to validate business logic and understand driver behaviors.

![SHAP Summary - Churn Drivers](SHAP.png)

* **Insight:** The model identifies **Frequency** (represented as the top feature) as the **#1 predictive factor** for retention.
* **Operational Trigger:** High-value customers showing a drop in frequency trigger an automated "We Miss You" campaign before they hit the critical churn window.

### 3. Unlocking the "Second Growth Curve" (Cross-Sell)
Market Basket Analysis revealed a specific consumption upgrade path, debunking the assumption that all tea drinkers are the same.

![Product Co-purchase Correlation Matrix](heat_map.png)

* **The "One-Way" Upgrade:** Analysis shows a strong asymmetry. **"Floral Blend"** users (Cluster 3) are **63.6% likely** to upgrade to **"Premium Pu-erh"**, whereas the reverse flow is only 11.7%.
* **Strategy:** Launch the **"Connoisseur's Journey"** bundle. Instead of generic cross-selling, explicitly target Cluster 3 users with a Pu-erh sampler add-on to drive ARPU uplift.

### 4. The O2O Blind Spot (Data Gap)
Our clustering analysis exposed a critical infrastructure gap in the data collection process regarding our physical store at Funan Mall.

![Customer Segments & Offline Preference](PCA.png)

* **Insight:** As seen in the *Business View* (right panel), the `Offline_Ratio` is **0.0%** for all user clusters.
* **Implication:** We are currently unable to attribute Funan Mall offline sampling to online retention, creating a "blind spot" in identifying true omnichannel users.
* **Action (Phase 1 Align):** The immediate next step is integrating **Funan POS data** to merge offline transaction history before expanding to Malaysia/HK.

## Repository Structure

```bash
Retail-Analytics-Strategy-Engine/
├── data/                     # Folder for generated CSVs (excluded from git)
├── scripts/
│   ├── data_generator.py              # Simulates realistic retail data (Privacy Safe)
│   ├── customer_segmentation.py       # RFM Analysis & K-Means Clustering
│   ├── churn_prediction.py            # XGBoost Training & SHAP Interpretation
│   └── market_basket.py               # Association Rules (Apriori) engine
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
