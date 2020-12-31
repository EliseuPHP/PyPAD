import time
import numba
from numba import jit, njit, prange
import numpy as np

y = 100
w = 100
v = 100

# input matrices
aux = np.zeros(shape=(y, v))
rmatrix = np.zeros(shape=(y, 1))

matrizA = np.loadtxt("arqA.dat")
matrizA = np.reshape(matrizA, (y, -1))

matrizB = np.loadtxt("arqB.dat")
matrizB = np.reshape(matrizB, (w, -1))

matrizC = np.loadtxt("arqC.dat")
matrizC = np.reshape(matrizC, (v, -1))

# multiplication function
@jit("void(double[:,:],double[:,:],double[:,:])")
def matmul(matrix1, matrix2, rmatrix):
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                rmatrix[i][j] += matrix1[i][k] * matrix2[k][j]


@jit(parallel=True, forceobj=True)
def parallel_sum(A):
    sum = 0.0
    for i in prange(A.shape[0]):
        sum += A[i]

    return sum


# Calculate running time
t1 = time.perf_counter()

matmul(matrizA, matrizB, aux)

matmul(aux, matrizC, rmatrix)

reduc = parallel_sum(rmatrix)

t2 = time.perf_counter()

# print results
print(f"Result: {reduc}. Finished in {t2-t1} seconds")