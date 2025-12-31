import logging
from typing import Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PropensityEngine:
    """
    Train and apply propensity models for churn and loyalty.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: dict[str, Any] = {}
        self.preprocessor: ColumnTransformer | None = None

    def load_data(self, cust_path: str, trans_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw CSVs and normalize date columns."""
        logger.info("Loading data from %s and %s...", cust_path, trans_path)

        df_cust = pd.read_csv(cust_path)
        df_trans = pd.read_csv(trans_path)

        df_cust["join_date"] = pd.to_datetime(df_cust["join_date"], errors="coerce")
        df_trans["transaction_date"] = pd.to_datetime(df_trans["transaction_date"], errors="coerce")
        return df_cust, df_trans

    def feature_engineering(self, df_cust: pd.DataFrame, df_trans: pd.DataFrame) -> pd.DataFrame:
        """
        Build customer-level features and binary labels.

        Outputs:
        - Label_Churn: churn indicator (from transaction table aggregation)
        - Label_Loyal: high-value/loyal proxy label (top quartile in frequency or monetary)
        """
        logger.info("Starting feature engineering...")

        ref_date = df_trans["transaction_date"].max() + pd.Timedelta(days=1)

        cust_agg = (
            df_trans.groupby("customer_id")
            .agg(
                {
                    "transaction_date": lambda x: (ref_date - x.max()).days,  # Recency
                    "transaction_id": "count",  # Frequency
                    "total_amount": "sum",  # Monetary
                    "quantity": "mean",  # Avg basket size
                    "product_category": lambda x: x.mode()[0] if not x.mode().empty else "Unknown",
                    "channel": "nunique",  # Omni-channel check
                    "is_churned": "max",  # Source churn flag
                }
            )
            .reset_index()
        )

        cust_agg.rename(
            columns={
                "transaction_date": "Recency",
                "transaction_id": "Frequency",
                "total_amount": "Monetary",
                "quantity": "Avg_Basket_Size",
                "product_category": "Fav_Category",
                "channel": "Channel_Count",
                "is_churned": "Label_Churn",
            },
            inplace=True,
        )

        # Derived features
        cust_agg["AOV"] = cust_agg["Monetary"] / cust_agg["Frequency"]
        cust_agg["Is_Omnichannel"] = (cust_agg["Channel_Count"] > 1).astype(int)

        # Loyalty proxy: top quartile in frequency OR monetary
        f_thresh = cust_agg["Frequency"].quantile(0.75)
        m_thresh = cust_agg["Monetary"].quantile(0.75)
        cust_agg["Label_Loyal"] = ((cust_agg["Frequency"] >= f_thresh) | (cust_agg["Monetary"] >= m_thresh)).astype(int)

        df_final = pd.merge(df_cust, cust_agg, on="customer_id", how="left")

        fill_map = {
            "Recency": 365 * 2,
            "Frequency": 0,
            "Monetary": 0,
            "AOV": 0,
            "Avg_Basket_Size": 0,
            "Channel_Count": 0,
            "Label_Churn": 0,
            "Label_Loyal": 0,
            "Fav_Category": "None",
            "Is_Omnichannel": 0,
        }
        df_final.fillna(value=fill_map, inplace=True)

        df_final["Tenure_Days"] = (ref_date - df_final["join_date"]).dt.days
        df_final["Tenure_Days"].fillna(0, inplace=True)

        logger.info("Feature engineering done. Shape: %s", df_final.shape)
        logger.info(
            "Class distribution - Churn: %.2f%%, Loyal: %.2f%%",
            df_final["Label_Churn"].mean() * 100,
            df_final["Label_Loyal"].mean() * 100,
        )

        # Drop identifiers not used for modeling
        return df_final.drop(columns=["customer_id", "join_date"])

    def _build_pipeline(self, scale_pos_weight: float) -> ImbPipeline:
        """Preprocess -> SMOTE -> XGBoost (binary classification)."""
        num_features = ["age", "Recency", "Frequency", "Monetary", "AOV", "Tenure_Days", "Avg_Basket_Size"]
        cat_features = ["gender", "member_tier", "region", "Fav_Category", "Is_Omnichannel"]

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ]
        )

        xgb_clf = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=self.random_state,
        )

        return ImbPipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("smote", SMOTE(random_state=self.random_state)),
                ("classifier", xgb_clf),
            ]
        )

    def train_model(self, X: pd.DataFrame, y: pd.Series, task_name: str) -> Any:
        """
        Train a model for a given task.

        task_name should be either "churn" or "loyalty" (stored under self.models[task_name]).
        """
        logger.info("Training %s model...", task_name)

        pos = float(y.sum())
        neg = float(len(y) - pos)
        scale_weight = (neg / pos) if pos > 0 else 1.0

        pipeline = self._build_pipeline(scale_weight)

        # Keep grid small for runtime; expand for real projects.
        param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.05, 0.1],
        }

        grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
        grid.fit(X, y)

        logger.info("Best %s CV AUC: %.4f", task_name, grid.best_score_)
        self.models[task_name] = grid.best_estimator_
        return grid.best_estimator_

    def generate_strategy_matrix(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Combine churn and loyalty probabilities into a simple action framework.

        Output columns:
        - Prob_Churn, Prob_Loyal
        - Segment, Action
        """
        if "churn" not in self.models or "loyalty" not in self.models:
            raise ValueError("Models not trained yet.")

        logger.info("Generating strategy matrix...")

        prob_churn = self.models["churn"].predict_proba(df_input)[:, 1]
        prob_loyal = self.models["loyalty"].predict_proba(df_input)[:, 1]

        results = df_input.copy()
        results["Prob_Churn"] = prob_churn
        results["Prob_Loyal"] = prob_loyal

        def assign_segment(row):
            # Thresholds can be tuned based on business constraints.
            if row["Prob_Churn"] > 0.6:
                if row["Prob_Loyal"] > 0.6:
                    return "VIP Rescue", "High value at-risk: personal outreach + targeted incentives."
                return "Let Go", "Low value at-risk: minimize retention spend."
            if row["Prob_Loyal"] > 0.6:
                return "Upsell/Cross-sell", "High value loyal: bundles, subscription, cross-sell."
            return "Nurture", "Potential: onboarding nudges and education."

        results[["Segment", "Action"]] = results.apply(lambda r: pd.Series(assign_segment(r)), axis=1)
        return results

    def explain_model_shap(self, X_test: pd.DataFrame, task_name: str = "churn"):
        """
        SHAP explanation for the XGBoost classifier in the pipeline.
        This assumes SHAP is compatible with the installed numpy and xgboost versions.
        """
        logger.info("Generating SHAP plots for %s...", task_name)

        model_pipeline = self.models[task_name]
        preprocessor = model_pipeline.named_steps["preprocessor"]
        X_trans = preprocessor.transform(X_test)

        xgb_model = model_pipeline.named_steps["classifier"]
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_trans)

        plt.figure()
        plt.title(f"SHAP Summary - {task_name.capitalize()} Drivers")
        shap.summary_plot(shap_values, X_trans, plot_type="bar", show=False)
        plt.tight_layout()
        plt.show()


def main():
    # Local file paths (edit as needed)
    cust_path = "your_path\mock_customer_profile.csv"
    trans_path = "your_path\mock_transaction_history.csv"

    print(f"Using Customer Path: {cust_path}")
    print(f"Using Transaction Path: {trans_path}")

    engine = PropensityEngine()

    # 1) Load and engineer features
    df_cust, df_trans = engine.load_data(cust_path, trans_path)
    df_features = engine.feature_engineering(df_cust, df_trans)

    # 2) Prepare modeling data
    X = df_features.drop(columns=["Label_Churn", "Label_Loyal"])
    y_churn = df_features["Label_Churn"]
    y_loyal = df_features["Label_Loyal"]

    # Stratify on churn to preserve class balance in the split
    X_train, X_test, y_c_train, y_c_test, y_l_train, y_l_test = train_test_split(
        X,
        y_churn,
        y_loyal,
        test_size=0.2,
        random_state=42,
        stratify=y_churn,
    )

    # 3) Train models
    engine.train_model(X_train, y_c_train, task_name="churn")
    engine.train_model(X_train, y_l_train, task_name="loyalty")

    # 4) Evaluate churn model
    print("\n=== Churn Model Evaluation ===")
    y_c_pred = engine.models["churn"].predict(X_test)
    print(classification_report(y_c_test, y_c_pred))

    # 5) Optional SHAP explanation (kept in a guarded block)
    try:
        engine.explain_model_shap(X_test, task_name="churn")
    except Exception as e:
        print(f"Skipping SHAP visualization due to error: {e}")

    # 6) Strategy matrix on the test set
    strategy_df = engine.generate_strategy_matrix(X_test)

    print("\n=== Recommended Strategy Matrix (First 5 Customers) ===")
    cols_show = ["Recency", "Frequency", "Monetary", "Prob_Churn", "Prob_Loyal", "Segment", "Action"]
    print(strategy_df[cols_show].head(5).to_string())

    strategy_df.to_csv("customer_strategy_action_plan.csv", index=False)


if __name__ == "__main__":
    main()
