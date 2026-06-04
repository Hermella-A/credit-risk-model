"""
Feature engineering and proxy target variable creation for credit risk model.
Uses RFM clustering to label high‑risk customers.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

def compute_rfm(df, reference_date=None):
    """
    Compute Recency, Frequency, Monetary for each customer.
    """
    df = df.copy()
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    if reference_date is None:
        reference_date = df['TransactionStartTime'].max()
    
    rfm = df.groupby('CustomerId').agg(
        recency=('TransactionStartTime', lambda x: (reference_date - x.max()).days),
        frequency=('TransactionId', 'count'),
        monetary=('Amount', lambda x: x[x > 0].sum())   # only positive amounts (spending)
    ).reset_index()
    
    # Handle zero monetary values (customers who never spent? but transactions may have negative)
    # For customers with no positive amount, monetary could be 0; we keep it.
    return rfm

def assign_risk_cluster(rfm_df, n_clusters=3, random_state=42):
    """
    Scale RFM features, apply KMeans clustering, and return cluster labels.
    Also returns the cluster centers to identify the high‑risk cluster.
    """
    # Select RFM features
    X = rfm_df[['recency', 'frequency', 'monetary']].copy()
    
    # Handle missing values (should be none, but safe)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Determine which cluster is the least engaged (high‑risk)
    # Typically high risk = low frequency, low monetary, high recency (old)
    # We'll compute mean of each cluster on original scale
    cluster_summary = rfm_df.copy()
    cluster_summary['cluster'] = clusters
    cluster_means = cluster_summary.groupby('cluster')[['recency', 'frequency', 'monetary']].mean()
    
    # High‑risk cluster: highest recency (oldest) and lowest frequency & monetary
    # We'll simply take the cluster with the lowest frequency (or lowest monetary)
    high_risk_cluster = cluster_means['frequency'].idxmin()  # cluster with smallest average frequency
    # Alternative: use a score (e.g., recency* - frequency - monetary) but simple works.
    
    return clusters, high_risk_cluster

def process_data(raw_df):
    """
    Main function: processes raw transaction data, computes RFM, clusters customers,
    assigns high‑risk label, and returns a model‑ready dataset with features and target.
    """
    df = raw_df.copy()
    
    # 1. Compute RFM
    rfm = compute_rfm(df)
    
    # 2. Cluster customers
    clusters, high_risk_cluster = assign_risk_cluster(rfm)
    rfm['cluster'] = clusters
    rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)
    
    # 3. Build additional features (optional – you can add more)
    # For example, average transaction amount, refund ratio, etc.
    # We'll compute some basic aggregations.
    customer_agg = df.groupby('CustomerId').agg(
        total_transactions=('TransactionId', 'count'),
        total_spent=('Amount', lambda x: x[x>0].sum()),
        total_refund=('Amount', lambda x: -x[x<0].sum()),
        avg_transaction=('Amount', 'mean'),
        std_transaction=('Amount', 'std'),
        unique_categories=('ProductCategory', 'nunique')
    ).reset_index()
    
    # Fill NaN std
    customer_agg['std_transaction'] = customer_agg['std_transaction'].fillna(0)
    
    # Compute refund ratio
    customer_agg['refund_ratio'] = customer_agg['total_refund'] / (customer_agg['total_spent'] + 1e-6)
    
    # 4. Merge RFM with other features
    final_df = rfm.merge(customer_agg, on='CustomerId', how='left')
    
    # 5. Drop original RFM components if you want to keep only scaled versions? We'll keep for now.
    # But we need to scale numerical features before modeling (will be done in training pipeline).
    # We'll just return the raw features plus target.
    
    # Keep only useful columns (excluding CustomerId for modeling)
    # We'll keep CustomerId for merging but drop later.
    feature_cols = ['recency', 'frequency', 'monetary', 'total_transactions', 'total_spent',
                    'total_refund', 'avg_transaction', 'std_transaction', 'unique_categories', 'refund_ratio']
    
    X = final_df[['CustomerId'] + feature_cols]
    y = final_df['is_high_risk']
    
    # Handle any remaining missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_numeric = X[feature_cols]
    X_imputed = imputer.fit_transform(X_numeric)
    X_imputed_df = pd.DataFrame(X_imputed, columns=feature_cols)
    X_imputed_df['CustomerId'] = X['CustomerId'].values
    
    # Combine features and target
    result = X_imputed_df
    result['is_high_risk'] = y.values
    
    return result

if __name__ == "__main__":
    # Example usage
    raw = pd.read_csv('data/raw/data.csv')
    processed = process_data(raw)
    print("Processed data shape:", processed.shape)
    print("Target distribution:\n", processed['is_high_risk'].value_counts())
    processed.to_csv('data/processed/customer_features.csv', index=False)
    print("Saved to ../data/processed/customer_features.csv")