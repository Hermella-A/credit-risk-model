# src/predict.py
import pandas as pd
import joblib
import mlflow
from mlflow.tracking import MlflowClient
import os
import logging

logger = logging.getLogger(__name__)

def load_model_from_registry(model_name="CreditRiskBestModel", stage="Production", fallback_local=True):
    """Load model from MLflow registry, with fallback to local file."""
    try:
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            raise Exception(f"No model version found for {model_name} in stage {stage}")
        model_uri = versions[0].source
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Loaded model from registry: {model_uri}")
        return model
    except Exception as e:
        logger.warning(f"Registry load failed: {e}")
        if fallback_local:
            local_path = "models/best_model.pkl"
            if os.path.exists(local_path):
                model = joblib.load(local_path)
                logger.info(f"Loaded local model from {local_path}")
                return model
            else:
                raise FileNotFoundError(f"Local model not found at {local_path}")
        else:
            raise

def predict(model, X: pd.DataFrame):
    """Return risk probabilities and binary classes."""
    proba = model.predict_proba(X)[:, 1]
    pred_class = (proba >= 0.5).astype(int)
    return proba, pred_class