# --========================================-- #
#   * Author  : NTheme - All rights reserved
#   * Created : 20 May 2025, 1:25 AM
#   * File    : main.py
#   * Project : Python
# --========================================-- #

from fastapi import FastAPI, HTTPException
from .schemas import (
    FitRequest, FitResponse,
    LoadRequest, GenericResponse,
    PredictRequest, PredictResponse,
    UnloadRequest, RemoveRequest
)
from .manager import (
    start_training, load_model, unload_model,
    predict, remove_model, remove_all, available_cores
)
from .config import settings

app = FastAPI()


@app.post("/fit", response_model=FitResponse)
def fit(req: FitRequest):
    try:
        pid = start_training(
            req.config.model_name,
            req.config.model_type,
            req.X, req.y, req.config.hyperparams or {}
        )
    except FileExistsError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(429, str(e))
    return FitResponse(status="training_started", pid=pid)


@app.post("/load", response_model=GenericResponse)
def load(req: LoadRequest):
    try:
        load_model(req.model_name)
    except FileNotFoundError:
        raise HTTPException(404, "Model not found")
    except RuntimeError as e:
        raise HTTPException(429, str(e))
    return GenericResponse(status="loaded")


@app.post("/unload", response_model=GenericResponse)
def unload(req: UnloadRequest):
    try:
        unload_model(req.model_name)
    except KeyError:
        raise HTTPException(404, "Model not loaded")
    return GenericResponse(status="unloaded")


@app.post("/predict", response_model=PredictResponse)
def do_predict(req: PredictRequest):
    try:
        preds = predict(req.model_name, req.X)
    except KeyError:
        raise HTTPException(404, "Model not loaded")
    return PredictResponse(predictions=preds)


@app.post("/remove", response_model=GenericResponse)
def remove(req: RemoveRequest):
    try:
        remove_model(req.model_name)
    except FileNotFoundError:
        raise HTTPException(404, "Model not found")
    return GenericResponse(status="removed")


@app.post("/remove_all", response_model=GenericResponse)
def removeall():
    remove_all()
    return GenericResponse(status="all_removed")


@app.get("/cores")
def get_cores():
    return {
        "free_cores": available_cores(),
        "total_cores": settings.num_cores
    }
