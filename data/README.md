# Data Dictionary & Privacy Compliance

## Compliance Notice
To strictly adhere to **Data Privacy Regulations (GDPR/PDPA)** and **Non-Disclosure Agreements (NDA)**, this repository **does NOT contain real proprietary client data**.

The CSV files in this directory are **synthetically generated** using the `scripts/data_generator.py` module. They statistically replicate the patterns, distributions, and relationships of the original dataset to demonstrate the validity of the analytical pipeline without exposing sensitive information.

---

## 1. File Overview

| File Name | Description | Rows (Approx) |
| :--- | :--- | :--- |
| **`mock_customer_profile.csv`** | Contains demographic and membership details for each unique customer. | ~1,000 |
| **`mock_transaction_history.csv`** | Granular transaction logs, including product details and channel information. | ~5,000 |

---

## 2. Data Schema

### A. `mock_customer_profile.csv` (User Dimension)
Primary Key: `customer_id`

| Column | Type | Description |
| :--- | :--- | :--- |
| `customer_id` | String | Unique identifier (e.g., `CUST_00123`). |
| `join_date` | Date | The date the customer registered as a member. |
| `age` | Integer | Customer age (Range: 18-65). |
| `gender` | String | Gender (M/F/Other) for demographic profiling. |
| `member_tier` | String | Loyalty status: `Bronze`, `Silver`, `Gold`. |
| `region` | String | Market location: `Singapore`, `Malaysia`, `Hong Kong`. |

### B. `mock_transaction_history.csv` (Fact Table)
Foreign Key: `customer_id`

| Column | Type | Description |
| :--- | :--- | :--- |
| `transaction_id` | String | Unique transaction reference (e.g., `TXN_00045`). |
| `customer_id` | String | Links to `mock_customer_profile.csv`. |
| `transaction_date`| Date | Date of purchase (Simulates seasonality over 2 years). |
| `product_category`| String | Item purchased (e.g., `Oolong Tea`, `Premium Pu-erh`, `Tea Accessories`). |
| `quantity` | Integer | Number of units purchased. |
| `unit_price` | Float | Price per unit (SGD). |
| `total_amount` | Float | Calculated as `quantity * unit_price`. |
| `channel` | String | Sales channel: `Online Store` vs `Funan Mall Outlet`. |
| `is_churned` | Integer | **Target Label for Prediction**. <br> `1` = Churned (Simulated risk), `0` = Active. |

---

## 3. Entity-Relationship Context

The datasets are designed to be joined as follows for analysis:

```sql
-- Conceptual SQL Join
SELECT 
    t.*, 
    c.age, 
    c.member_tier, 
    c.region
FROM mock_transaction_history t
LEFT JOIN mock_customer_profile c 
    ON t.customer_id = c.customer_id;
```

## 4. Note on "Churn" Label
In a real-world scenario, the `is_churned` label is derived from historical inactivity (e.g., "No purchase in last 90 days"). In this synthetic dataset, the label is probabilistically generated within `data_generator.py` based on behavioral patterns to allow for supervised learning (XGBoost) demonstration.
