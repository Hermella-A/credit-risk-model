# Credit Risk Model – Bati Bank & eCommerce Partner

[![CI](https://github.com/Hermella-A/credit-risk-model/actions/workflows/ci.yml/badge.svg)](https://github.com/Hermella-A/credit-risk-model/actions/workflows/ci.yml)

## Project Overview

This project builds an end-to-end credit risk scoring model for Bati Bank's buy-now-pay-later service. Using transaction data from an eCommerce partner (no historical default labels), we:

- Engineered a proxy target variable via RFM clustering.
- Built customer-level features (Recency, Frequency, Monetary, refund ratios, product category proportions).
- Trained and compared Logistic Regression, Random Forest, and XGBoost models.
- Tracked experiments with MLflow and registered the best model.
- Deployed the model as a FastAPI REST API, containerized with Docker.
- Set up CI/CD (linting + tests) with GitHub Actions.

**Best model:** XGBoost (ROC-AUC ≈ 0.85) – registered in MLflow as `CreditRiskBestModel`.

---

## Business Understanding (Basel II, Proxy Risk, Model Trade-offs)

Basel II requires interpretable, well-documented risk models. Since the raw data contains no default flag, we used a **proxy target** derived from customer transaction behaviour (low frequency, low monetary, high recency). Risks include proxy bias and regulatory scrutiny, which we mitigate by transparent documentation and SHAP explanations.

| Aspect | Simple (Logistic Regression) | High‑performance (XGBoost) |
|--------|------------------------------|----------------------------|
| Interpretability | Very high | Low (requires SHAP) |
| Regulatory acceptance | High | Moderate |
| Predictive power | Moderate | High |
| Our choice | Baseline | Champion (with SHAP) |

---

## Repository Structure
credit-risk-model/
├── .github/workflows/ci.yml # CI/CD pipeline (flake8 + pytest)
├── data/ # ignored by Git
│ ├── raw/ # raw transaction data
│ └── processed/ # customer_features.csv, X_woe.csv
├── notebooks/
│ ├── eda.ipynb # exploratory analysis
│ └── test_rfm_clustering.ipynb # RFM clustering and proxy target
├── src/
│ ├── data_processing.py # feature engineering, RFM, WoE/IV
│ ├── train.py # model training, MLflow, hyperparameter tuning
│ ├── predict.py # inference helper
│ └── api/
│ ├── main.py # FastAPI app
│ └── pydantic_models.py # request/response schemas
├── tests/
│ └── test_data_processing.py # unit tests
├── models/ # saved models (ignored)
├── mlruns/ # MLflow experiments (ignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md

## Setup & Run
```bash
git clone https://github.com/Hermella-A/credit-risk-model.git
cd credit-risk-model
python -m venv venv && source venv/bin/activate  # or venv\Scripts\activate
pip install -r requirements.txt
python src/data_processing.py   # feature engineering + WoE/IV
python src/train.py             # train models, MLflow
uvicorn src.api.main:app --reload   # API
docker-compose up --build       # or run with Docker
pytest tests/                   # tests

Key Results
Best model: XGBoost (ROC‑AUC 0.85)

Top features: Monetary, Frequency, Recency

Deployment: FastAPI + Docker, CI/CD with GitHub Actions

Git Practices
Feature branches (task-1…task-5, improve-feature-engineering) merged via Pull Requests.

Conventional commits, CI badge shows passing build.

Limitations & Future Work
Proxy target may not perfectly reflect true default risk.

Low recall (0.19) – consider threshold tuning or oversampling.

Next: integrate external credit data, monitor model drift.

Author
Hermella-Amha – GitHub