# Retail-Analytics-Strategy-Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Privacy%20Compliant-green?style=for-the-badge)

> **Context:** Experiential Data Science Project
> **Objective:** Transforming raw transactional data into actionable retention and cross-sell strategies.

---

## 📖 Executive Summary
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
