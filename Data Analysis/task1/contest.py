import numpy

def matrix_multiplication(A: numpy.ndarray, B: numpy.ndarray) -> numpy.ndarray: 
    return (A[:,  numpy.newaxis, :] * B.T[numpy.newaxis, :, :]).sum(axis=2)

def find_nearest_points(A: numpy.ndarray, B: numpy.ndarray, k: int) -> numpy.ndarray:    
    return numpy.argsort(((A[:, numpy.newaxis, :] - B[numpy.newaxis, :, :]) ** 2).sum(axis=2), axis=0)[:k].T + 1
