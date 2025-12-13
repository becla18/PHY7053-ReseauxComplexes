import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from metrics import nodf, remove_null_rows_cols


# empirical network
B = np.genfromtxt('devoir4/plant-pollinator-biadjacency-matrix.csv', delimiter=',')
nodf_emp = nodf(B)

# canonical ensemble PROBLEM WITH 0 DEGREE NODES?
canon_matrices = np.load("devoir4/canon_biadj_matrices.npy")
nodf_canon = []


print('Canonical ensemble...')
for B_can in tqdm(canon_matrices):
    B_can = remove_null_rows_cols(B_can)
    nodf_canon.append(nodf(B_can))

# microcanonical ensemble
microcan_matrices = np.load("devoir4/microcan_biadj_matrices.npy")
nodf_microcan = []
print('Microcanonical ensemble...')
for B_mic in tqdm(microcan_matrices):
    nodf_microcan.append(nodf(B_mic))

# compare assortativity coefficients
plt.figure()
plt.hist(nodf_canon, bins=50, label='Canonique', color="#1f63b066")
plt.hist(nodf_microcan, bins=50, label='Microcanonique', color="#e2873166")
plt.axvline(nodf_emp, label='Empirique', linestyle='--', color='#353535ff')
plt.ylabel('Nombre de graphes')
plt.xlabel('NODF')
plt.legend()
plt.savefig('devoir4/figures/nodf.pdf')
plt.show()
