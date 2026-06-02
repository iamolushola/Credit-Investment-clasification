"""Minimal FastAPI service for the classifier."""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference import predict_records
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)
app = FastAPI(title="Borrower/Investor Classifier", version="1.0.0")
class PredictionRequest(BaseModel):
    records: Union[Dict[str, Any], List[Dict[str, Any]]]
@app.get("/health")
def health(): return {"status": "ok"}
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        records = request.records if isinstance(request.records, list) else [request.records]
        return {"predictions": predict_records(records)}
    except Exception as exc:
        LOGGER.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
