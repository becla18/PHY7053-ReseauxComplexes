import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from metrics import *


""" Compute the occurence of motif 5 """

# empirical network
B = np.genfromtxt('devoir4/plant-pollinator-biadjacency-matrix.csv', delimiter=',')
motifs5_emp = motifs5(B)
print(np.sum(B))

# canonical ensemble
canon_matrices = np.load("devoir4/canon_biadj_matrices.npy")
motifs5_canon = []
for B_can in canon_matrices:
    motifs5_canon.append(motifs5(B_can))

# microcanonical ensemble
microcan_matrices = np.load("devoir4/microcan_biadj_matrices.npy")
motifs5_microcan = []
for B_mic in tqdm(microcan_matrices):
    motifs5_microcan.append(motifs5(B_mic))

# compare the occurence of motif 5
plt.figure()
plt.hist(motifs5_canon, bins=40, label='Canonique', color="#1f63b066")
plt.hist(motifs5_microcan, bins=40, label='Microcanonique', color="#e2873166")
plt.axvline(motifs5_emp, label='Empirique', linestyle='--', color='#353535ff')
plt.ylabel('Nombre de graphes')
plt.xlabel('Occurences du motif 5')
plt.legend()
plt.savefig('devoir4/figures/motif5.pdf')
plt.show()


""" Compute the number of motifs 6 """

# empirical network
B = np.genfromtxt('devoir4/plant-pollinator-biadjacency-matrix.csv', delimiter=',')
motifs6_emp = motifs6(B)

# canonical ensemble
canon_matrices = np.load("devoir4/canon_biadj_matrices.npy")
motifs6_canon = []
print('Canonical ensemble...')
for B_can in canon_matrices:
    motifs6_canon.append(motifs6(B_can))

# microcanonical ensemble
microcan_matrices = np.load("devoir4/microcan_biadj_matrices.npy")
motifs6_microcan = []
print('Microcanonical ensemble...')
for B_mic in tqdm(microcan_matrices):
    motifs6_microcan.append(motifs6(B_mic))

# compare the occurence of motif 6
plt.figure()
plt.hist(motifs6_canon, bins=40, label='Canonique', color="#1f63b066")
plt.hist(motifs6_microcan, bins=40, label='Microcanonique', color="#e2873166")
plt.axvline(motifs6_emp, label='Empirique', linestyle='--', color='#353535ff')
plt.ylabel('Nombre de graphes')
plt.xlabel('Occurences du motif 6')
plt.legend()
plt.savefig('devoir4/figures/motif6.pdf')
plt.show()
