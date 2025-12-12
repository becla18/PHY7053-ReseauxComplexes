import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from metrics import *


# empirical network
B = np.genfromtxt('devoir4/plant-pollinator-biadjacency-matrix.csv', delimiter=',')
overlap_emp = niche_overlap(B)
print(overlap_emp)

# canonical ensemble PROBLEM WITH NULL DEGREES
# canon_matrices = np.load("devoir4/canon_biadj_matrices.npy")
# overlap_canon = []
# for B_can in canon_matrices:
#     overlap_canon.append(niche_overlap(B_can))

# microcanonical ensemble
microcan_matrices = np.load("devoir4/microcan_biadj_matrices.npy")
overlap_microcan = []
for B_mic in tqdm(microcan_matrices):
    overlap_microcan.append(niche_overlap(B_mic))

# # compare niche overlap
plt.figure()
# plt.hist(overlap_canon, bins=49, label='canon', color="#1f63b066")
plt.hist(overlap_microcan, bins=49, label='microcan', color="#e2873166")
plt.axvline(overlap_emp, label='empirical', linestyle='--', color='#353534ff')
plt.legend()
plt.show()
