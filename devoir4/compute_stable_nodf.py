import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from metrics import stable_nodf, remove_null_rows_cols


# empirical network
B = np.genfromtxt('devoir4/plant-pollinator-biadjacency-matrix.csv', delimiter=',')
stable_nodf_emp = stable_nodf(B)

# canonical ensemble PROBLEM WITH 0 DEGREE NODES?
canon_matrices = np.load("devoir4/canon_biadj_matrices.npy")
stable_nodf_canon = []
print('Canonical ensemble...')
for B_can in tqdm(canon_matrices):
    B_can = remove_null_rows_cols(B_can)
    stable_nodf_canon.append(stable_nodf(B_can))

# microcanonical ensemble
microcan_matrices = np.load("devoir4/microcan_biadj_matrices.npy")
stable_nodf_microcan = []
print('Microcanonical ensemble...')
for B_mic in tqdm(microcan_matrices):
    stable_nodf_microcan.append(stable_nodf(B_mic))

# compare assortativity coefficients
plt.figure()
plt.hist(stable_nodf_canon, bins=50, label='Canonique', color="#1f63b066")
plt.hist(stable_nodf_microcan, bins=50, label='Microcanonique', color="#e2873166")
plt.axvline(stable_nodf_emp, label='Empirique', linestyle='--', color='#353535ff')
plt.ylabel('Nombre de graphes')
plt.xlabel('Stable NODF')
plt.legend()
plt.savefig('devoir4/figures/stable_nodf.pdf')
plt.show()
