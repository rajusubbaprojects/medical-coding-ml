"""FastAPI app for ICD-10 code prediction."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from serving.app.predict import predict, available_models

app = FastAPI(
    title="Medical Coding ML API",
    description="Predict ICD-10 codes from clinical discharge notes.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    note: str = Field(..., min_length=10, description="Discharge note text")
    model: str = Field("encounter", description="Model to use: encounter or patient")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Confidence threshold")


class PredictResponse(BaseModel):
    predictions: list[dict]
    model: str
    n_codes_evaluated: int
    note_length: int


@app.get("/health")
def health():
    return {"status": "ok", "available_models": available_models()}


@app.post("/predict", response_model=PredictResponse)
def predict_codes(req: PredictRequest):
    if req.model not in ("encounter", "patient"):
        raise HTTPException(status_code=400, detail="model must be 'encounter' or 'patient'")
    try:
        predictions = predict(req.note, model_name=req.model, threshold=req.threshold)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    from serving.app.predict import _load_bundle
    bundle = _load_bundle(req.model)

    return PredictResponse(
        predictions=predictions,
        model=req.model,
        n_codes_evaluated=len(bundle["label_names"]),
        note_length=len(req.note),
    )
