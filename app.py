"""
FastAPI webapp for the webnovel tag/genre predictor.

Run with:
    uvicorn app:app --reload

Then open http://127.0.0.1:8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.registry import list_models, get_model
from src.predict import predict_tags

app = FastAPI(title="Webnovel Tag Predictor")
app.mount("/static", StaticFiles(directory="static"), name="static")


class PredictRequest(BaseModel):
    model: str
    title: str
    description: str
    threshold: float = 0.30


@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html") as f:
        return f.read()


@app.get("/api/models")
def get_models():
    models = list_models()
    if not models:
        raise HTTPException(status_code=404, detail="No models found in models/ folder")
    return {"models": models}


@app.post("/api/predict")
def predict(req: PredictRequest):
    try:
        bundle = get_model(req.model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not req.title.strip() and not req.description.strip():
        raise HTTPException(status_code=400, detail="Provide a title and/or description")

    predictions = predict_tags(bundle, req.title, req.description, req.threshold)
    return {"predictions": predictions}
