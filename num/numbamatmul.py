import time
import numba
from numba import njit, prange
import numpy as np

y = 997
w = 981
v = 991

# input matrices
aux = np.array(np.zeros(shape=(y, v)), dtype=np.float64)
rmatrix = np.array(np.zeros(shape=(y, 1)), dtype=np.float64)

matrizA = np.array(np.loadtxt("arqA.dat"), dtype=np.float64)
matrizA = np.array(np.reshape(matrizA, (y, -1)), dtype=np.float64)

matrizB = np.array(np.loadtxt("arqB.dat"), dtype=np.float64)
matrizB = np.array(np.reshape(matrizB, (w, -1)), dtype=np.float64)

matrizC = np.array(np.loadtxt("arqC.dat"), dtype=np.float64)
matrizC = np.array(np.reshape(matrizC, (v, -1)), dtype=np.float64)

# multiplication function
@njit("void(float64[:,:],float64[:,:],float64[:,:])", parallel=True, nogil=True, fastmath=True)
def matmul(matrix1, matrix2, rmatrix):
    for i in prange(matrix1.shape[0]):
        for j in prange(matrix2[0].shape[0]):
            for k in prange(matrix2.shape[0]):
                rmatrix[i][j] += matrix1[i][k] * matrix2[k][j]

# reduction function
@njit(parallel=True, nogil=True, fastmath=True)
def prange_test(A):
    s = 0.0
    for i in prange(A.shape[0]):
        s += A[i]
    return s


# Calculate running time
t1 = time.perf_counter()

matmul(matrizA, matrizB, aux)

matmul(aux, matrizC, rmatrix)

t2 = time.perf_counter()

rmatrix = rmatrix.flatten()

t3 = time.perf_counter()

reduc = prange_test(rmatrix)

t4 = time.perf_counter()

# print results
print(f"Result: {reduc}. Finished in {(t2-t1)-(t3-t4)} seconds")