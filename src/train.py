"""
Model training and tracking with MLflow.
Trains multiple classifiers, tunes hyperparameters, logs experiments, and registers the best model.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib

# Set random seed for reproducibility
RANDOM_STATE = 42

def load_data(filepath='data/processed/customer_features.csv'):
    """Load processed dataset, drop non-numeric columns, split features/target."""
    df = pd.read_csv(filepath)
    # Drop identifier columns (CustomerId) and any other non-numeric
    drop_cols = ['CustomerId'] if 'CustomerId' in df.columns else []
    X = df.drop(columns=['is_high_risk'] + drop_cols, axis=1)
    y = df['is_high_risk']
    # Ensure all features are numeric (convert if needed)
    X = X.apply(pd.to_numeric, errors='coerce')
    # Drop rows with NaNs (if any)
    valid_idx = X.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]
    return X, y

def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)

def train_and_log_model(model, param_grid, model_name, X_train, y_train, X_test, y_test):
    """Perform grid search, log to MLflow, return best model and metrics."""
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, error_score='raise')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    # Log to MLflow
    with mlflow.start_run(run_name=model_name, nested=True):
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, model_name)
        # Log feature importance if available
        if hasattr(best_model, 'feature_importances_'):
            importance = best_model.feature_importances_
            for i, col in enumerate(X_train.columns):
                mlflow.log_metric(f"importance_{col}", importance[i])
    
    return best_model, metrics

def main():
    # Load data
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Define models and hyperparameter grids (without deprecated penalty for LogisticRegression)
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, solver='lbfgs'),
            'param_grid': {
                'C': [0.01, 0.1, 1, 10],
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE),
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        },
        'XGBoost': {
            'model': xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'),
            'param_grid': {
                'n_estimators': [50, 100],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1]
            }
        }
    }
    
    # Set MLflow experiment
    mlflow.set_experiment("Credit Risk Modeling")
    
    best_models = {}
    best_score = 0
    best_model_name = None
    best_model_obj = None
    
    for name, config in models.items():
        print(f"Training {name}...")
        model, metrics = train_and_log_model(
            config['model'], config['param_grid'], name,
            X_train, y_train, X_test, y_test
        )
        best_models[name] = (model, metrics)
        print(f"{name} ROC-AUC: {metrics['roc_auc']:.4f}")
        
        if metrics['roc_auc'] > best_score:
            best_score = metrics['roc_auc']
            best_model_name = name
            best_model_obj = model
    
    # Register the best model in MLflow Model Registry
    print(f"\nBest model: {best_model_name} with ROC-AUC = {best_score:.4f}")
    
    # Log the best model as a separate run and register
    with mlflow.start_run(run_name="Best_Model") as run:
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_roc_auc", best_score)
        # Log the model
        if best_model_name == 'XGBoost':
            mlflow.xgboost.log_model(best_model_obj, "best_model")
        else:
            mlflow.sklearn.log_model(best_model_obj, "best_model")
        
        # Register the model
        model_uri = f"runs:/{run.info.run_id}/best_model"
        mlflow.register_model(model_uri, "CreditRiskBestModel")
    
    # Save best model locally as joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model_obj, 'models/best_model.pkl')
    print("Best model saved to models/best_model.pkl")
    
    # Optionally, save all models
    for name, (model, _) in best_models.items():
        joblib.dump(model, f'models/{name}.pkl')
    
    print("All models saved.")

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    main()