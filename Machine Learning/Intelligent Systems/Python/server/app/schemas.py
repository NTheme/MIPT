# --========================================-- #
#   * Author  : NTheme - All rights reserved
#   * Created : 20 May 2025, 1:26 AM
#   * File    : schemas.py
#   * Project : Python
# --========================================-- #

from pydantic import BaseModel
from typing import List, Optional, Dict


class ModelConfig(BaseModel):
    model_name: str
    model_type: str
    hyperparams: Optional[Dict] = {}


class FitRequest(BaseModel):
    X: List[List[float]]
    y: List[int]
    config: ModelConfig


class FitResponse(BaseModel):
    status: str
    pid: int


class LoadRequest(BaseModel):
    model_name: str


class UnloadRequest(LoadRequest): pass


class RemoveRequest(LoadRequest): pass


class GenericResponse(BaseModel):
    status: str


class PredictRequest(BaseModel):
    model_name: str
    X: List[List[float]]


class PredictResponse(BaseModel):
    predictions: List[int]
