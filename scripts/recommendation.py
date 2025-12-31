import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 8)


class MarketBasketRecommender:
    def __init__(self, min_support: float = 0.01, min_lift: float = 1.1, min_confidence: float = 0.2):
        # min_support: frequency threshold for itemsets
        # min_lift: association strength threshold (>1 indicates positive association)
        # min_confidence: conditional probability threshold P(B|A)
        self.min_support = min_support
        self.min_lift = min_lift
        self.min_confidence = min_confidence
        self.rules = None
        self.basket_matrix = None

    def load_data(self, trans_path: str) -> pd.DataFrame:
        """Load transactions and apply basic cleaning/filters."""
        logger.info("Loading transaction data from %s...", trans_path)

        df = pd.read_csv(trans_path)
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")

        # Keep only valid sales records (exclude refunds/negative totals)
        if "total_amount" in df.columns:
            df = df[df["total_amount"] > 0]

        # Focus on active customers to learn bundles that support retention/upsell
        if "is_churned" in df.columns:
            before = len(df)
            df = df[df["is_churned"] == 0]
            logger.info("Filtered out churned customers: %d -> %d records", before, len(df))

        df["product_category"] = df["product_category"].astype(str).str.strip()
        return df

    def create_basket(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a customer-by-category boolean basket matrix.

        This is a lifetime basket view per customer (not per single checkout),
        used to answer: "Customers who ever bought A, did they also buy B?"
        """
        logger.info("Creating basket matrix...")

        basket = (
            df.groupby(["customer_id", "product_category"])["transaction_id"]
            .count()
            .unstack()
            .reset_index()
            .fillna(0)
            .set_index("customer_id")
        )

        self.basket_matrix = basket.map(lambda x: x >= 1).astype(bool)

        # Keep customers with at least 2 distinct categories (required for associations)
        self.basket_matrix = self.basket_matrix[self.basket_matrix.sum(axis=1) >= 2]

        logger.info("Basket matrix ready: %s", self.basket_matrix.shape)
        return self.basket_matrix

    def run_analysis(self) -> pd.DataFrame | None:
        """Run Apriori + association rules mining."""
        if self.basket_matrix is None or self.basket_matrix.empty:
            logger.warning("Basket matrix is empty. Cannot run analysis.")
            return None

        logger.info("Mining frequent itemsets (min_support=%.4f)...", self.min_support)

        try:
            frequent_itemsets = apriori(self.basket_matrix, min_support=self.min_support, use_colnames=True)
            if frequent_itemsets.empty:
                logger.warning("No frequent itemsets found. Consider lowering min_support.")
                return None

            logger.info("Generating rules (min_lift=%.4f)...", self.min_lift)
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=self.min_lift)

            rules = rules[rules["confidence"] >= self.min_confidence]
            rules = rules.sort_values(by="lift", ascending=False)

            # Normalize set-valued columns for CSV readability (single-item rules)
            rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0] if len(x) else "")
            rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0] if len(x) else "")

            self.rules = rules
            logger.info("Analysis complete: %d rules.", len(rules))
            return rules

        except Exception as e:
            logger.error("Apriori / rule mining failed: %s", e)
            return None

    def generate_marketing_bundles(self, top_n: int = 10) -> pd.DataFrame:
        """Convert top association rules into bundle recommendations."""
        if self.rules is None or self.rules.empty:
            return pd.DataFrame()

        bundles = []
        for _, row in self.rules.head(top_n).iterrows():
            item_a = row["antecedents"]
            item_b = row["consequents"]
            lift = row["lift"]
            conf = row["confidence"]

            bundles.append(
                {
                    "Bundle Name": f"{item_a} + {item_b} Combo",
                    "Trigger Product": item_a,
                    "Upsell Product": item_b,
                    "Strength (Lift)": f"{lift:.2f}x",
                    "Success Prob (Conf)": f"{conf:.1%}",
                    "Marketing Copy": (
                        f"Love {item_a}? Customers who bought it also loved {item_b}. "
                        "Try them together for 10% off!"
                    ),
                }
            )

        return pd.DataFrame(bundles)

    def plot_heatmap(self):
        """Plot a simple product-category correlation heatmap from the basket matrix."""
        if self.basket_matrix is None or self.basket_matrix.empty:
            return

        logger.info("Generating product correlation heatmap...")
        corr = self.basket_matrix.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", center=0)
        plt.title("Cross-Sell Opportunity Heatmap (Correlation)", fontsize=14)
        plt.tight_layout()
        plt.show()


def main():
    trans_path = "your_path\mock_transaction_history.csv"
    print(f"Target Transaction File: {trans_path}")

    recommender = MarketBasketRecommender(min_support=0.005, min_lift=0.99, min_confidence=0.05)

    try:
        df_trans = recommender.load_data(trans_path)
    except Exception:
        print("Error: Files not found. Please check paths.")
        return

    recommender.create_basket(df_trans)
    rules = recommender.run_analysis()

    if rules is not None and not rules.empty:
        marketing_df = recommender.generate_marketing_bundles(top_n=10)

        print("=== Top 5 Cross-Sell Recommendations (Strategy Menu) ===")
        print(marketing_df[["Bundle Name", "Success Prob (Conf)", "Marketing Copy"]].head(5).to_string(index=False))

        rules.to_csv("technical_rules_raw.csv", index=False)
        marketing_df.to_csv("marketing_bundles_recommendation.csv", index=False)
        print("Files saved: 'technical_rules_raw.csv', 'marketing_bundles_recommendation.csv'")

        recommender.plot_heatmap()
    else:
        print("No strong associations found with current thresholds.")
        print("Tip: Try lowering min_support or min_lift in main().")


if __name__ == "__main__":
    main()
