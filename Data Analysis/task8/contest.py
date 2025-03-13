import numpy

from coin import coin


def get_shapes(a, b):
    a = (a,) if isinstance(a, int) else a
    b = (b,) if isinstance(b, int) else b
    return a + b


def uniform(size=1, precision=30):
    var = coin(get_shapes(size, precision))
    return (var * numpy.logspace(1, precision, base=0.5, num=precision)).sum(axis=-1)


def normal(size=1, loc=0.0, scale=1.0, precision=30):
    var = 1 - uniform(get_shapes(2, size), precision)
    return loc + scale * numpy.cos(2 * numpy.pi * var[1]) * numpy.sqrt(-2 * numpy.log(var[0]))


def expon(size=1, lambd=1.0, precision=30):
    var = 1 - uniform(size, precision)
    return -numpy.log(var) / lambd
