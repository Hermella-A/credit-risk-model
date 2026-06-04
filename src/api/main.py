# src/api/main.py
import sys
import os
import logging
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic_models import PredictionRequest, PredictionResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.predict import load_model_from_registry, predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Credit Risk API", description="Predicts high-risk customers", version="1.0")

model = None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = load_model_from_registry()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    try:
        input_df = pd.DataFrame([request.features])
        proba, pred_class = predict(model, input_df)
        return PredictionResponse(risk_probability=float(proba[0]), risk_class=int(pred_class[0]))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))