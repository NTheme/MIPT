import functools
from collections import deque


def cache(size):
    dict = {}
    arr = deque()

    def cache(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            all = args + tuple(kwargs.values())
            if all in dict:
                return dict[all]
            else:
                if len(arr) == size:
                    del dict[arr.popleft()]
                res = func(*args, **kwargs)
                arr.append(all)
                dict[all] = res
                return dict[all]
        return wrapper
    return cache
