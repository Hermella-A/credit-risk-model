"""
Unit tests for data_processing module.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from src.data_processing import compute_rfm, assign_risk_cluster

def test_compute_rfm_returns_expected_columns():
    # Create a minimal DataFrame
    data = {
        'CustomerId': ['C1', 'C1', 'C2'],
        'TransactionId': ['T1', 'T2', 'T3'],
        'TransactionStartTime': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'Amount': [100, -10, 200]
    }
    df = pd.DataFrame(data)
    rfm = compute_rfm(df)
    expected_cols = ['CustomerId', 'recency', 'frequency', 'monetary']
    assert all(col in rfm.columns for col in expected_cols), "Missing expected columns"
    assert len(rfm) == 2  # two customers

def test_assign_risk_cluster_returns_two_values():
    # Use the same rfm from above
    data = {
        'CustomerId': ['C1', 'C2', 'C3'],
        'TransactionId': ['T1', 'T2', 'T3'],
        'TransactionStartTime': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'Amount': [100, 200, 300]
    }
    df = pd.DataFrame(data)
    rfm = compute_rfm(df)
    clusters, high_risk_cluster = assign_risk_cluster(rfm, n_clusters=2)
    assert len(clusters) == len(rfm), "Clusters length mismatch"
    assert high_risk_cluster in [0, 1], "High risk cluster index out of range"