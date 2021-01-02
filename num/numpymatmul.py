import time
import numba
import numpy as np

y = 997
w = 981
v = 991

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
def matmul(matrix1, matrix2, rmatrix):
    np.dot(matrix1,matrix2,rmatrix)

# Calculate running time
t1 = time.perf_counter()

matmul(matrizA, matrizB, aux)

matmul(aux, matrizC, rmatrix)

reduc = np.sum(rmatrix)

t2 = time.perf_counter()

# print results
print(f"Result: {reduc}. Finished in {t2-t1} seconds")