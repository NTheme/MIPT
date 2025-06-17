# --========================================-- #
#   * Author  : NTheme - All rights reserved
#   * Created : 21 May 2025, 7:54 PM
#   * File    : sync_client.py
#   * Project : Python
# --========================================-- #

import requests
import time
import asyncio
import aiohttp
import numpy as np

URL = "http://localhost:8000"


def wait_for_model(name, timeout=3600, poll=5):
    import os

    start = time.time()
    path = f"./app/models/{name}.joblib"
    while time.time() - start < timeout:
        if os.path.exists(path):
            print(f"Model {name} is ready")
            return True
        time.sleep(poll)
    raise TimeoutError(f"Model {name} not ready in {timeout}s")


def train_model(name, model_type):
    X = np.random.randn(10000, 50).tolist()
    y = np.random.randint(0, 2, size=10000).tolist()
    payload = {"X": X, "y": y, "config": {"model_name": name, "model_type": model_type}}
    resp = requests.post(f"{URL}/fit", json=payload)
    print(name, resp.json())
    if resp.status_code == 200:
        wait_for_model(name)


def sequential_training():
    start = time.time()
    train_model("m1", "rf")
    train_model("m2", "svm")
    train_model("m3", "logreg")
    train_model("m4", "svm")
    train_model("m5", "svm")
    print("Sequential total:", time.time() - start)


if __name__ == "__main__":
    sequential_training()
