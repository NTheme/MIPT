import time
import functools


def profiler(func):
    rec = 0

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal rec
        if rec == 0:
            wrapper.calls = 0
        wrapper.calls += 1

        rec += 1
        start = time.time()
        res = func(*args, **kwargs)
        wrapper.last_time_taken = time.time() - start
        rec -= 1
        return res

    return wrapper
