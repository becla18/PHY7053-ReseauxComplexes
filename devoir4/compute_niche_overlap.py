import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from metrics import *


# empirical network
B = np.genfromtxt('devoir4/plant-pollinator-biadjacency-matrix.csv', delimiter=',')
overlap_emp = niche_overlap(B)
print(overlap_emp)

# canonical ensemble
canon_matrices = np.load("devoir4/canon_biadj_matrices.npy")
overlap_canon = []
print('Canonical ensemble...')
# remove null degree nodes

for B_can in tqdm(canon_matrices):
    B_can = remove_null_rows_cols(B_can)
    overlap_canon.append(niche_overlap(B_can))

# microcanonical ensemble
microcan_matrices = np.load("devoir4/microcan_biadj_matrices.npy")
overlap_microcan = []
print('Microcanonical ensemble...')
for B_mic in tqdm(microcan_matrices):
    overlap_microcan.append(niche_overlap(B_mic))

# # compare niche overlap
plt.figure()
plt.hist(overlap_canon, bins=50, label='Canonique', color="#1f63b066")
plt.hist(overlap_microcan, bins=50, label='Microcanonique', color="#e2873166")
plt.axvline(overlap_emp, label='Empirique', linestyle='--', color='#353534ff')
plt.ylabel('Nombre de graphes')
plt.xlabel('Niche overlap')
plt.legend()
plt.savefig('devoir4/figures/niche_overlap.pdf')
plt.show()
