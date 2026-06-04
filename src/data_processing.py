"""
Feature engineering pipeline with sklearn Pipeline, categorical encoding, and manual WoE/IV.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class CustomerAggregator(BaseEstimator, TransformerMixin):
    """Aggregates raw transactions to customer-level features."""
    def __init__(self, reference_date=None):
        self.reference_date = reference_date

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        if self.reference_date is None:
            self.reference_date = df['TransactionStartTime'].max()
        
        # Basic RFM and refund metrics
        agg = df.groupby('CustomerId').agg(
            recency=('TransactionStartTime', lambda x: (self.reference_date - x.max()).days),
            frequency=('TransactionId', 'count'),
            monetary=('Amount', lambda x: x[x > 0].sum()),
            refund_amount=('Amount', lambda x: -x[x < 0].sum()),
            refund_count=('Amount', lambda x: (x < 0).sum()),
            avg_amount=('Amount', 'mean'),
            std_amount=('Amount', 'std')
        ).reset_index()
        
        # Fix future warning: use assignment instead of inplace fillna
        agg['std_amount'] = agg['std_amount'].fillna(0)
        agg['refund_ratio'] = agg['refund_amount'] / (agg['monetary'] + 1e-6)
        
        # Proportions of product categories (one-hot encoding)
        cat_dummies = pd.get_dummies(df[['CustomerId', 'ProductCategory']], columns=['ProductCategory'], prefix='cat')
        cat_agg = cat_dummies.groupby('CustomerId').mean().reset_index()
        
        # Merge
        final = agg.merge(cat_agg, on='CustomerId', how='left').fillna(0)
        final.set_index('CustomerId', inplace=True)
        return final

def compute_woe_iv_manual(X, y, n_bins=10):
    """
    Manual Weight of Evidence and Information Value calculation.
    For each numeric feature, bin it, compute WoE and IV.
    Uses integer bin labels to avoid Categorical issues.
    """
    iv_dict = {}
    X_woe = pd.DataFrame(index=X.index)
    
    for col in X.columns:
        # Bin the feature
        try:
            # Try quantile binning (equal frequency)
            X_bin = pd.qcut(X[col], q=n_bins, duplicates='drop', labels=False)
        except:
            # Fallback to equal width binning
            X_bin = pd.cut(X[col], bins=n_bins, labels=False, include_lowest=True)
        
        # Cross-tabulation with target
        cross = pd.crosstab(X_bin, y, margins=False)
        
        # Ensure both good (1) and bad (0) exist
        if 1 not in cross.columns:
            cross[1] = 0
        if 0 not in cross.columns:
            cross[0] = 0
        
        total_good = y.sum()
        total_bad = len(y) - total_good
        
        cross['good_pct'] = cross[1] / total_good
        cross['bad_pct'] = cross[0] / total_bad
        
        # Avoid division by zero
        cross['good_pct'] = cross['good_pct'].replace(0, 1e-6)
        cross['bad_pct'] = cross['bad_pct'].replace(0, 1e-6)
        
        cross['woe'] = np.log(cross['good_pct'] / cross['bad_pct'])
        cross['iv_component'] = (cross['good_pct'] - cross['bad_pct']) * cross['woe']
        iv = cross['iv_component'].sum()
        iv_dict[col] = iv
        
        # Map WoE values back to each row (using integer bin labels)
        woe_map = cross['woe'].to_dict()
        X_woe[col] = X_bin.map(woe_map).fillna(0)
    
    iv_df = pd.DataFrame({
        'feature': list(iv_dict.keys()),
        'iv': list(iv_dict.values())
    }).sort_values('iv', ascending=False)
    return iv_df, X_woe

def create_processed_data(raw_df, target_df=None):
    """
    Transforms raw transactions to model-ready dataset.
    If target_df is given (with 'CustomerId' and 'is_high_risk'), returns (X, y).
    Otherwise returns X only.
    """
    aggregator = CustomerAggregator()
    X = aggregator.transform(raw_df)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    
    if target_df is not None:
        # Ensure target alignment
        y = target_df.set_index('CustomerId')['is_high_risk'].loc[X.index]
        return X_scaled, y
    return X_scaled

if __name__ == "__main__":
    raw = pd.read_csv('data/raw/data.csv')
    # Load previously created target (from clustering)
    proc = pd.read_csv('data/processed/customer_features.csv')
    X, y = create_processed_data(raw, proc[['CustomerId', 'is_high_risk']])
    print("Features shape:", X.shape)
    # Compute WoE/IV manually
    iv_df, X_woe = compute_woe_iv_manual(X, y)
    print("Information Values:\n", iv_df)
    # Optionally save WoE-transformed dataset
    X_woe.to_csv('data/processed/X_woe.csv', index=True)