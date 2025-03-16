import sys
sys.path.append("../build/Src/matrix_multiply/release")

import cu_matrix_multiply

import numpy as np
import time

A_rows = 1024
A_cols = 512
B_cols = 256

A = np.random.randint(10, size=(A_rows,A_cols))
B = np.random.randint(10, size=(A_cols,B_cols))

t0 = time.time()
#ref_C = A @ B
ref_C = []
for i in range(A_rows):
    row = []
    for j in range(B_cols):
        sum = 0
        for k in range(A_cols):
            sum += A[i][k] * B[k][j]
        row.append(sum)
    ref_C.append(row)      
    
t1 = time.time()
print("python matrix multiply time: ", t1-t0)


t0 = time.time()
np_C = np.matmul(A, B)
t1 = time.time()
print("numpy matrix multiply time: ", t1-t0)


t0 = time.time()
C = cu_matrix_multiply.mm(A, B)
t1 = time.time()
print("cuda matrix multiply time: ", t1-t0)

print("cuda matrix multiply matches python reference output: ", np.array_equal(ref_C, C))
