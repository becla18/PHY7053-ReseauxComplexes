import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from metrics import *


# empirical network
B = np.genfromtxt('devoir4/plant-pollinator-biadjacency-matrix.csv', delimiter=',')
assort_emp = assortativity_degrees(B)

# canonical ensemble
canon_matrices = np.load("devoir4/canon_biadj_matrices.npy")
assort_canon = []
for B_can in tqdm(canon_matrices):
    assort_canon.append(assortativity_degrees(B_can))

# microcanonical ensemble
microcan_matrices = np.load("devoir4/microcan_biadj_matrices.npy")
assort_microcan = []
for B_mic in tqdm(microcan_matrices):
    assort_microcan.append(assortativity_degrees(B_mic))

# compare assortativity coefficients
plt.figure()
plt.hist(assort_canon, bins=50, label='Canonical', color="#1f63b066")
plt.hist(assort_microcan, bins=50, label='Microcanonical', color="#e2873166")
plt.axvline(assort_emp, label='empirical', linestyle='--', color='#353535ff')
plt.ylabel('Number of graphs')
plt.xlabel('Degree assortativity')
plt.legend()
plt.savefig('devoir4/figures/assortativity.pdf')
plt.show()
