# src/api/pydantic_models.py
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class PredictionRequest(BaseModel):
    """Input features for a single customer."""
    features: Dict[str, float] = Field(..., example={"recency": 10, "frequency": 5, "monetary": 1000})

class PredictionResponse(BaseModel):
    """Risk probability and binary class."""
    risk_probability: float = Field(..., example=0.85)
    risk_class: int = Field(..., example=1)   # 1 = high risk, 0 = low risk