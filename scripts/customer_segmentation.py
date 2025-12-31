import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import argparse
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class CustomerSegmentationToolkit:
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.pipeline = None
        self.model = None
        self.data_processed = None
        self.feature_names = None

    def load_data(self, cust_path: str, trans_path: str):
        """Load & Merge"""
        logger.info(f"Loading data from {cust_path} and {trans_path}...")
        try:
            df_cust = pd.read_csv(cust_path)
            df_trans = pd.read_csv(trans_path)
            
            df_cust['join_date'] = pd.to_datetime(df_cust['join_date'], errors='coerce')
            df_trans['transaction_date'] = pd.to_datetime(df_trans['transaction_date'], errors='coerce')
            
            return df_cust, df_trans
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise

    def feature_engineering(self, df_cust: pd.DataFrame, df_trans: pd.DataFrame) -> pd.DataFrame:
        """
        Conduct feature engineering to create RFM + O2O features.
        """
        logger.info("Engineering features (RFM + O2O)...")
        
        ref_date = df_trans['transaction_date'].max() + pd.Timedelta(days=1)
        
        # RFM
        agg_rules = {
            'transaction_date': lambda x: (ref_date - x.max()).days, # Recency
            'transaction_id': 'count',                               # Frequency
            'total_amount': 'sum',                                   # Monetary
            'quantity': 'mean',                                      # Avg Basket Size
            'product_category': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
        }
        
        # O2O Features
        if 'channel' not in df_trans.columns:
            logger.warning("'channel' column missing. Simulating data for O2O demonstration.")
            np.random.seed(42)
            df_trans['channel'] = np.random.choice(['Online', 'Store'], size=len(df_trans), p=[0.7, 0.3])

        # Perference
        channel_pivot = df_trans.pivot_table(
            index='customer_id', 
            columns='channel', 
            values='transaction_id', 
            aggfunc='count', 
            fill_value=0
        )
        channel_pivot.columns = [f'Freq_{c}' for c in channel_pivot.columns]
        
        # Aggregate RFM
        cust_agg = df_trans.groupby('customer_id').agg(agg_rules)
        cust_agg.rename(columns={
            'transaction_date': 'Recency',
            'transaction_id': 'Frequency',
            'total_amount': 'Monetary',
            'quantity': 'Avg_Basket_Size',
            'product_category': 'Fav_Category'
        }, inplace=True)
        
        # Merge O2O Features
        cust_agg = cust_agg.join(channel_pivot)
        
        # Calculate O2O Features
        if 'Freq_Online' in cust_agg.columns and 'Freq_Store' in cust_agg.columns:
            cust_agg['Is_Omnichannel'] = ((cust_agg['Freq_Online'] > 0) & (cust_agg['Freq_Store'] > 0)).astype(int)
            cust_agg['Offline_Ratio'] = cust_agg['Freq_Store'] / cust_agg['Frequency']
        else:
            cust_agg['Is_Omnichannel'] = 0
            cust_agg['Offline_Ratio'] = 0

        # Merge back to customer profile
        df_final = pd.merge(df_cust, cust_agg, on='customer_id', how='left')
        
        # Fulfill missing values
        df_final['Recency'] = df_final['Recency'].fillna(365*2)
        df_final[['Frequency', 'Monetary', 'Avg_Basket_Size', 'Offline_Ratio']] = \
            df_final[['Frequency', 'Monetary', 'Avg_Basket_Size', 'Offline_Ratio']].fillna(0)
        df_final['Fav_Category'] = df_final['Fav_Category'].fillna('None')
        df_final['Tenure_Days'] = (ref_date - df_final['join_date']).dt.days
        df_final['Tenure_Days'] = df_final['Tenure_Days'].fillna(0)

        logger.info(f"Features engineered. Shape: {df_final.shape}")
        return df_final

    def preprocess_and_cluster(self, df: pd.DataFrame):
        logger.info(f"Running Clustering with K={self.n_clusters}...")
        
        # Feature selection
        num_cols = ['age', 'Tenure_Days', 'Recency', 'Frequency', 'Monetary', 'Offline_Ratio']
        cat_cols = ['gender', 'member_tier', 'region', 'Fav_Category', 'Is_Omnichannel']
        
        # Preprocessing
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])
        
        # Clustering Pipeline
        self.pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('pca', PCA(n_components=0.95, random_state=self.random_state)), 
            ('kmeans', KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state))
        ])
        
        # Training
        self.pipeline.fit(df)
        labels = self.pipeline.named_steps['kmeans'].labels_

        df_result = df.copy()
        df_result['Cluster_ID'] = labels
        
        # Calculate Silhouette Score
        X_trans = self.pipeline.named_steps['preprocess'].transform(df)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        X_pca = self.pipeline.named_steps['pca'].transform(X_trans)
        score = silhouette_score(X_pca, labels)
        logger.info(f"Clustering completed. Silhouette Score: {score:.4f}")
        
        self.data_processed = df_result
        return df_result

    def generate_profiles(self) -> pd.DataFrame:
        """Portrait Generation"""
        if self.data_processed is None:
            return None
            
        logger.info("Generating Segment Profiles...")
        metrics = ['Recency', 'Frequency', 'Monetary', 'Offline_Ratio', 'age', 'Tenure_Days']
        
        # Aggregate Metrics
        profile = self.data_processed.groupby('Cluster_ID')[metrics].mean()
        profile['Count'] = self.data_processed['Cluster_ID'].value_counts()
        profile['Pct'] = profile['Count'] / profile['Count'].sum()
        
        # Find Top Preferred Category and Region for each cluster
        profile['Top_Category'] = self.data_processed.groupby('Cluster_ID')['Fav_Category'].agg(
            lambda x: x.mode()[0] if not x.mode().empty else 'Mixed'
        )
        profile['Top_Region'] = self.data_processed.groupby('Cluster_ID')['region'].agg(
            lambda x: x.mode()[0] if not x.mode().empty else 'Mixed'
        )
        
        # Heuristic Naming
        def name_segment(row):
            if row['Monetary'] > profile['Monetary'].quantile(0.7):
                if row['Offline_Ratio'] > 0.4: return "High-Value Store VIP"
                return "High-Value Online VIP"
            if row['Recency'] > profile['Recency'].quantile(0.7):
                return "Dormant/Churned"
            if row['Frequency'] > profile['Frequency'].quantile(0.6):
                return "Loyal Regulars"
            return "Casual/New"

        profile['Segment_Name'] = profile.apply(name_segment, axis=1)
        
        return profile.sort_values('Monetary', ascending=False)

    def plot_segments(self):
        if self.data_processed is None: return
        
        logger.info("Plotting visualizations...")

        pre_step = self.pipeline.named_steps['preprocess']
        pca_step = self.pipeline.named_steps['pca']
        X_trans = pre_step.transform(self.data_processed)
        if hasattr(X_trans, "toarray"): X_trans = X_trans.toarray()
        X_pca = pca_step.transform(X_trans)
        
        df_plot = self.data_processed.copy()
        df_plot['PCA1'] = X_pca[:, 0]
        df_plot['PCA2'] = X_pca[:, 1]
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # Figure 1: PCA 2D Projection
        sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster_ID', data=df_plot, palette='viridis', ax=axes[0], s=60)
        axes[0].set_title('Customer Segments (2D PCA Projection)')
        
        # Figure 2: Business View - Monetary vs. Offline Ratio
        sns.scatterplot(x='Offline_Ratio', y='Monetary', hue='Cluster_ID', data=df_plot, palette='viridis', ax=axes[1], s=60)
        axes[1].set_title('Business View: Spend vs. Offline Preference')
        axes[1].set_xlabel('Offline Ratio (0=Pure Online, 1=Pure Store)')
        axes[1].set_ylabel('Total Monetary Value')
        
        plt.tight_layout()
        plt.show()
        logger.info("Plots displayed.")


# Main Execution
def main():

    cust_path = "your_path\mock_customer_profile.csv"
    trans_path = "your_path\mock_transaction_history.csv"
    
    print(f"Target Customer File: {cust_path}")
    print(f"Target Transaction File: {trans_path}")

    toolkit = CustomerSegmentationToolkit(n_clusters=5) 
    
    # Loading
    try:
        df_cust, df_trans = toolkit.load_data(cust_path, trans_path)
    except Exception:
        print("Error: Files not found. Please check paths.")
        return

    # Feature Engineering
    df_features = toolkit.feature_engineering(df_cust, df_trans)
    
    # Clustering
    df_result = toolkit.preprocess_and_cluster(df_features)
    
    # Portrait segments
    profile_df = toolkit.generate_profiles()
    print("\n=== Segment Profiles (Strategy Table) ===")
    cols_to_show = ['Segment_Name', 'Count', 'Pct', 'Monetary', 'Recency', 'Offline_Ratio', 'Top_Category', 'Top_Region']
    print(profile_df[cols_to_show].to_string(formatters={
        'Monetary': '${:,.2f}'.format,
        'Pct': '{:.1%}'.format,
        'Offline_Ratio': '{:.1%}'.format,
        'Recency': '{:.0f} days'.format
    }))
    
    # Save results
    df_result.to_csv("customer_segments_labeled.csv", index=False)
    profile_df.to_csv("segment_profiles_summary.csv")
    print("\nFiles saved: 'customer_segments_labeled.csv', 'segment_profiles_summary.csv'")
    
    toolkit.plot_segments()

if __name__ == "__main__":
    main()