# --========================================-- #
#   * Author  : NTheme - All rights reserved
#   * Created : 20 May 2025, 1:26 AM
#   * File    : manager.py
#   * Project : Python
# --========================================-- #

import os
import threading
import joblib

from multiprocessing import get_context
from typing import Dict

from .config import settings

ctx = get_context("fork")
Process = ctx.Process

_active_trains = ctx.Value('i', 0)
_active_lock = ctx.Lock()

_loaded_models: Dict[str, object] = {}
_loaded_lock = threading.Lock()


def _train_and_save(model_name: str, model_cls, X, y, hyperparams: dict):
    try:
        model = model_cls(**hyperparams)
        model.fit(X, y)
        path = os.path.join(settings.model_dir, f"{model_name}.joblib")
        joblib.dump(model, path)
    finally:
        with _active_lock:
            _active_trains.value -= 1


def start_training(model_name: str, model_type: str, X, y, hyperparams: dict):
    path = os.path.join(settings.model_dir, f"{model_name}.joblib")
    if os.path.exists(path):
        raise FileExistsError(f"Model '{model_name}' already exists")

    with _active_lock:
        if _active_trains.value >= settings.num_cores:
            raise RuntimeError("No free cores for training")
        _active_trains.value += 1

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    cls_map = {
        "logreg": LogisticRegression,
        "rf": RandomForestClassifier,
        "svm": SVC,
    }
    model_cls = cls_map.get(model_type)
    if model_cls is None:
        with _active_lock:
            _active_trains.value -= 1
        raise ValueError(f"Unknown model type '{model_type}'")

    proc = Process(
        target=_train_and_save,
        args=(model_name, model_cls, X, y, hyperparams),
    )
    proc.start()
    return proc.pid


def available_cores() -> int:
    with _active_lock:
        return settings.num_cores - _active_trains.value


def load_model(model_name: str):
    with _loaded_lock:
        if model_name in _loaded_models:
            return
        if len(_loaded_models) >= settings.max_loaded_models:
            raise RuntimeError("Loaded models limit reached")
        path = os.path.join(settings.model_dir, f"{model_name}.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model '{model_name}' not found on disk")
        _loaded_models[model_name] = joblib.load(path)


def unload_model(model_name: str):
    with _loaded_lock:
        if model_name not in _loaded_models:
            raise KeyError(f"Model '{model_name}' is not loaded")
        del _loaded_models[model_name]


def predict(model_name: str, X):
    with _loaded_lock:
        if model_name not in _loaded_models:
            raise KeyError(f"Model '{model_name}' is not loaded")
        return _loaded_models[model_name].predict(X).tolist()


def remove_model(model_name: str):
    with _loaded_lock:
        _loaded_models.pop(model_name, None)
    path = os.path.join(settings.model_dir, f"{model_name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model '{model_name}' not found on disk")
    os.remove(path)


def remove_all():
    with _loaded_lock:
        _loaded_models.clear()
    for fname in os.listdir(settings.model_dir):
        if fname.endswith(".joblib"):
            os.remove(os.path.join(settings.model_dir, fname))
