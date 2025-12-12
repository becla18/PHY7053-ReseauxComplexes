import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from metrics import *


# empirical network
B = np.genfromtxt('devoir4/plant-pollinator-biadjacency-matrix.csv', delimiter=',')
specrad_emp = spectral_radius(B)

# canonical ensemble
canon_matrices = np.load("devoir4/canon_biadj_matrices.npy")
specrad_canon = []
print('Canonical ensemble...')
for B_can in tqdm(canon_matrices):
    specrad_canon.append(spectral_radius(B_can))

# microcanonical ensemble
microcan_matrices = np.load("devoir4/microcan_biadj_matrices.npy")
specrad_microcan = []
for B_mic in tqdm(microcan_matrices):
    specrad_microcan.append(spectral_radius(B_mic))

# compare the occurence of motif 5
plt.figure()
plt.hist(specrad_canon, bins=40, label='Canonical', color="#1f63b066")
plt.hist(specrad_microcan, bins=40, label='Microcanical', color="#e2873166")
plt.axvline(specrad_emp, label='Empirical', linestyle='--', color='#353535ff')
plt.ylabel('Number of graphs')
plt.xlabel('Spectral radius')
plt.legend()
plt.savefig('devoir4/figures/spectral_radius.pdf')
plt.show()
