import sys
sys.path.append("../build/Src/matrix_multiply/release")

import cu_matrix_multiply

import numpy as np

A_rows = 1024
A_cols = 512
B_cols = 256

A = np.random.randint(10, size=(A_rows,A_cols))
B = np.random.randint(10, size=(A_cols,B_cols))

ref_C = A @ B

C = cu_matrix_multiply.mm(A, B)

print("cuda matrix multiply matches python reference output: ", np.array_equal(ref_C, C))
