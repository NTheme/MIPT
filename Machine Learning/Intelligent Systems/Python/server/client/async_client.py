# --========================================-- #
#   * Author  : NTheme - All rights reserved
#   * Created : 21 May 2025, 7:54 PM
#   * File    : async_client.py
#   * Project : Python
# --========================================-- #

import time
import asyncio
import aiohttp
import numpy as np

URL = "http://localhost:8000"


async def train(session, name, model_type):
    X = np.random.randn(10000, 50).tolist()
    y = np.random.randint(0, 2, size=10000).tolist()
    payload = {"X": X, "y": y, "config": {"model_name": name, "model_type": model_type}}
    async with session.post(f"{URL}/fit", json=payload) as resp:
        print(name, await resp.json())


async def async_training():
    start = time.time()
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(
            train(session, "a1", "rf"),
            train(session, "a2", "svm"),
            train(session, "a3", "logreg"),
            train(session, "a4", "svm"),
            train(session, "a5", "svm"),
        )
    print("Async total:", time.time() - start)


if __name__ == "__main__":
    asyncio.run(async_training())
