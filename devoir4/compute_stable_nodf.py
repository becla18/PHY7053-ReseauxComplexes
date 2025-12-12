import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from metrics import stable_nodf


# empirical network
B = np.genfromtxt('devoir4/plant-pollinator-biadjacency-matrix.csv', delimiter=',')
stable_nodf_emp = stable_nodf(B)

# canonical ensemble PROBLEM WITH 0 DEGREE NODES?
# canon_matrices = np.load("devoir4/canon_biadj_matrices.npy")
# stable_nodf_canon = []
# for B_can in canon_matrices:
#     stable_nodf_canon.append(stable_nodf(B_can))

# microcanonical ensemble
microcan_matrices = np.load("devoir4/microcan_biadj_matrices.npy")
stable_nodf_microcan = []
for B_mic in tqdm(microcan_matrices):
    stable_nodf_microcan.append(stable_nodf(B_mic))

# plt.imshow(microcan_matrices[0])
# plt.show()
# plt.imshow(microcan_matrices[10])
# plt.show()
# plt.imshow(B)
# plt.show()

# compare assortativity coefficients
plt.figure()
# plt.hist(stable_nodf_canon, label='canon', color="#1f63b066")
plt.hist(stable_nodf_microcan, bins=50, label='microcan', color="#e2873166")
plt.axvline(stable_nodf_emp, label='empirical', linestyle='--', color='#353535ff')
plt.legend()
plt.show()
