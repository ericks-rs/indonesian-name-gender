from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from inference import Predictor

HERE = Path(__file__).parent
RESULTS_DIR = HERE.parent / "results"

app = FastAPI(
    title="Riset Nama Gender - Demo API",
    description="Klasifikasi gender berdasarkan nama Indonesia.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading models...")
predictor = Predictor(RESULTS_DIR)
print("Models loaded.")

class PredictRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, description="Nama lengkap")
    model: str = Field(default="CharBiLSTM", description="Model name")

class CompareRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)

class AttentionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    model: str = Field(default="CharBiLSTM")

@app.get("/api/models")
def list_models():
    return {
        "models": predictor.available_models,
        "default": "CharBiLSTM",
        "device": str(predictor.device),
    }

@app.post("/api/predict")
def predict_single(req: PredictRequest):
    if req.model not in predictor.available_models:
        raise HTTPException(400, f"Unknown model. Available: {predictor.available_models}")
    return predictor.predict_single(req.name, req.model)

@app.post("/api/compare")
def predict_compare(req: CompareRequest):
    results = predictor.predict_all(req.name)
    char_models = [r for r in results if "Char" in r["model"]]
    word_models = [r for r in results if "Word" in r["model"]]
    return {"name": req.name, "predictions": char_models + word_models}

@app.post("/api/attention")
def predict_attention(req: AttentionRequest):
    if req.model not in predictor.available_models:
        raise HTTPException(400, f"Unknown model. Available: {predictor.available_models}")
    return predictor.predict_with_attention(req.name, req.model)

STATIC_DIR = HERE / "static"

@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
