import numpy as np


canon_matrices = np.load("devoir4/canon_biadj_matrices.npy")
i = 0
for mat in canon_matrices[:2]:
    d1 = np.sum(mat, axis=1)
    d2 = np.sum(mat, axis=0)

    if min(d1) == 0:
        i += 1
    elif min(d2) == 0:
        i += 1

print(i)