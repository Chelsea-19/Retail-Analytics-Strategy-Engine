import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


# Global configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Matplotlib defaults for presentation-ready plots
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "SimHei"]
plt.rcParams["figure.figsize"] = (14, 8)


class CustomerSegmentationToolkit:
    """
    This class encapsulates data loading, feature engineering,
    clustering, profiling, and visualization logic.
    """

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        # Clustering hyperparameters
        self.n_clusters = n_clusters
        self.random_state = random_state

        # Runtime artifacts
        self.pipeline = None
        self.data_processed = None
        self.profile_df = None

    def load_data(self, cust_path: str, trans_path: str):
        logger.info("Loading data from %s and %s...", cust_path, trans_path)

        df_cust = pd.read_csv(cust_path)
        df_trans = pd.read_csv(trans_path)

        # Enforce datetime types
        df_cust["join_date"] = pd.to_datetime(df_cust["join_date"], errors="coerce")
        df_trans["transaction_date"] = pd.to_datetime(df_trans["transaction_date"], errors="coerce")

        return df_cust, df_trans

    def feature_engineering(self, df_cust: pd.DataFrame, df_trans: pd.DataFrame) -> pd.DataFrame:
        """
        Construct customer-level features from transaction data.

        Features include:
        - Recency, Frequency, Monetary value
        - Average basket size
        - Preferred product category
        - Online vs offline transaction mix
        - Customer tenure
        """
        logger.info("Engineering customer features...")

        # Reference date for recency calculation
        ref_date = df_trans["transaction_date"].max() + pd.Timedelta(days=1)

        # Derive transaction channel if missing
        if "channel" not in df_trans.columns:
            np.random.seed(42)
            df_trans["channel"] = np.random.choice(
                ["Online", "Store"], size=len(df_trans), p=[0.7, 0.3]
            )

        # Aggregate transactional features at customer level
        agg_rules = {
            "transaction_date": lambda x: (ref_date - x.max()).days,
            "transaction_id": "count",
            "total_amount": "sum",
            "quantity": "mean",
            "product_category": lambda x: x.mode()[0] if not x.mode().empty else "Unknown",
        }

        cust_agg = df_trans.groupby("customer_id").agg(agg_rules)
        cust_agg.rename(
            columns={
                "transaction_date": "Recency",
                "transaction_id": "Frequency",
                "total_amount": "Monetary",
                "quantity": "Avg_Basket_Size",
                "product_category": "Fav_Category",
            },
            inplace=True,
        )

        # Channel-specific frequency
        channel_pivot = df_trans.pivot_table(
            index="customer_id",
            columns="channel",
            values="transaction_id",
            aggfunc="count",
            fill_value=0,
        )
        channel_pivot.columns = [f"Freq_{c}" for c in channel_pivot.columns]
        cust_agg = cust_agg.join(channel_pivot)

        # Offline transaction ratio
        if {"Freq_Online", "Freq_Store"}.issubset(cust_agg.columns):
            cust_agg["Offline_Ratio"] = cust_agg["Freq_Store"] / cust_agg["Frequency"]
        else:
            cust_agg["Offline_Ratio"] = 0.0

        # Merge with static customer attributes
        df_final = pd.merge(df_cust, cust_agg, on="customer_id", how="left")

        # Handle missing values
        df_final["Recency"] = df_final["Recency"].fillna(730)
        df_final[["Frequency", "Monetary", "Avg_Basket_Size", "Offline_Ratio"]] = (
            df_final[["Frequency", "Monetary", "Avg_Basket_Size", "Offline_Ratio"]].fillna(0)
        )
        df_final["Fav_Category"] = df_final["Fav_Category"].fillna("None")

        # Customer tenure in days
        df_final["Tenure_Days"] = (ref_date - df_final["join_date"]).dt.days.fillna(0)

        logger.info("Feature table shape: %s", df_final.shape)
        return df_final

    def preprocess_and_cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing, dimensionality reduction, and KMeans clustering.
        """
        logger.info("Clustering customers into %d segments...", self.n_clusters)

        num_log = ["Recency", "Frequency", "Monetary"]
        num_std = ["age", "Tenure_Days"]
        cat_cols = ["gender", "member_tier", "region", "Fav_Category"]

        # Log-transform skewed variables
        log_transformer = Pipeline(
            steps=[
                ("log", FunctionTransformer(np.log1p, validate=True)),
                ("scaler", StandardScaler()),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("log", log_transformer, num_log),
                ("num", StandardScaler(), num_std),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ]
        )

        # Full modeling pipeline
        self.pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("pca", PCA(n_components=0.95, random_state=self.random_state)),
                ("kmeans", KMeans(
                    n_clusters=self.n_clusters,
                    n_init=10,
                    random_state=self.random_state,
                )),
            ]
        )

        self.pipeline.fit(df)
        labels = self.pipeline.named_steps["kmeans"].labels_

        df_out = df.copy()
        df_out["Cluster_ID"] = labels

        # Evaluate clustering quality in PCA space
        X_trans = self.pipeline.named_steps["preprocess"].transform(df)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        X_pca = self.pipeline.named_steps["pca"].transform(X_trans)

        score = silhouette_score(X_pca, labels)
        logger.info("Silhouette score: %.4f", score)

        self.data_processed = df_out
        return df_out

    def generate_profiles(self) -> pd.DataFrame:
        """
        Generate summary statistics and labels for each customer segment.
        """
        if self.data_processed is None:
            return None

        metrics = ["Recency", "Frequency", "Monetary", "Offline_Ratio", "age", "Tenure_Days"]
        profile = self.data_processed.groupby("Cluster_ID")[metrics].mean()

        profile["Count"] = self.data_processed["Cluster_ID"].value_counts()
        profile["Pct"] = profile["Count"] / profile["Count"].sum()

        profile["Top_Category"] = (
            self.data_processed.groupby("Cluster_ID")["Fav_Category"]
            .agg(lambda x: x.mode()[0] if not x.mode().empty else "Mixed")
        )

        def assign_name(row):
            if row["Monetary"] > profile["Monetary"].quantile(0.66):
                return "Champions (High Value)"
            if row["Recency"] > profile["Recency"].quantile(0.66):
                return "At Risk / Lost"
            return "Potential / Casual"

        profile["Segment_Name"] = profile.apply(assign_name, axis=1)
        self.profile_df = profile.sort_values("Monetary", ascending=False)
        return self.profile_df

    def plot_segments(self):
        """
        Plot a customer value matrix using Recency vs Monetary value.
        """
        if self.profile_df is None:
            self.generate_profiles()

        df_viz = self.profile_df.reset_index()

        fig, ax = plt.subplots(figsize=(12, 7))
        bubble_sizes = df_viz["Count"] * 10

        ax.scatter(
            df_viz["Recency"],
            df_viz["Monetary"],
            s=bubble_sizes,
            alpha=0.9,
            edgecolors="white",
            linewidth=2,
        )

        for _, row in df_viz.iterrows():
            ax.annotate(
                f"{row['Segment_Name']}\n(n={row['Count']})",
                (row["Recency"], row["Monetary"]),
                xytext=(0, 40),
                textcoords="offset points",
                ha="center",
                fontsize=11,
                fontweight="bold",
            )

        ax.set_title("Customer Value Matrix", loc="left", fontsize=16, fontweight="bold")
        ax.set_xlabel("Recency (days)")
        ax.set_ylabel("Average Monetary Value")

        ax.grid(True, linestyle=":", alpha=0.4)
        plt.tight_layout()
        plt.savefig("consulting_style_matrix.png")
        plt.show()


def main():
    cust_path = "your_path\mock_customer_profile.csv"
    trans_path = "your_path\mock_transaction_history.csv"

    toolkit = CustomerSegmentationToolkit(n_clusters=3)

    df_cust, df_trans = toolkit.load_data(cust_path, trans_path)
    df_feat = toolkit.feature_engineering(df_cust, df_trans)
    df_labeled = toolkit.preprocess_and_cluster(df_feat)

    profile_df = toolkit.generate_profiles()
    print(profile_df[["Segment_Name", "Count", "Pct", "Monetary", "Recency"]])

    df_labeled.to_csv("customer_segments_labeled.csv", index=False)
    profile_df.to_csv("segment_profiles_summary.csv")

    toolkit.plot_segments()


if __name__ == "__main__":
    main()
