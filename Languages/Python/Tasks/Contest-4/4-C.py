import sys
import functools


def takes(*args):
    gv = args

    def wraps(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(min(len(gv), len(args))):
                if not isinstance(args[i], gv[i]):
                    raise TypeError
            return func(*args, **kwargs)
        return wrapper
    return wraps


exec(sys.stdin.read())
