import sys
import contextlib


@contextlib.contextmanager
def dumper(stream):
    try:
        yield
    except Exception as err:
        stream.write(str(err))


@contextlib.contextmanager
def supresser(*types):
    try:
        yield
    except Exception as err:
        for exc in types:
            if err.__class__.__name__ == exc.__name__:
                return
        raise (err)


@contextlib.contextmanager
def retyper(type_from, type_to):
    try:
        yield
    except type_from as err:
        exception = type_to()
        exception.args = err.args
        raise exception
