from collections.abc import Container
from app import VeryImportantClass as Cls
from app import decorator
from numbers import Number


class HackedClass(Cls):
    def __init__(self):
        super().__init__()
        for name in dir(self):
            attr = Cls.__getattribute__(self, name)
            if not name[0] == '_' and callable(attr):
                Cls.__setattr__(self, name, decorator(attr))

    def __getattribute__(self, name):
        if name[0] == '_':
            return Cls.__getattribute__(self, name)
        val = Cls.__getattribute__(self, name)
        if isinstance(val, Number):
            return val * 2
        if isinstance(val, Container):
            return type(val)()
        return val
