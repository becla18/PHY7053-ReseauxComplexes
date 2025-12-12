import numpy as np
from tqdm import tqdm
from curveball import *
import matplotlib.pyplot as plt


# Generate a sample of the microcanonical configuration model for bipartite graphs
# using Curveball algorithm


original_B = np.genfromtxt('devoir4/plant-pollinator-biadjacency-matrix.csv', delimiter=',')

d1 = np.sum(original_B, axis=1)
d2 = np.sum(original_B, axis=0)

n_matrices = 10**4
microcan_matrices = []
for _ in tqdm(range(n_matrices)):
    B = original_B.copy()
    for _ in range(1000):
        B = curveball_iter(B)
    microcan_matrices.append(B)

d1_2 = np.sum(B, axis=1)
d2_2 = np.sum(B, axis=0)

# verify that the degree sequences are preserved
print((d1==d1_2).all(), (d2 == d2_2).all())

# save the sampled biadjacency matrices
np.save("devoir4/microcan_biadj_matrices.npy", microcan_matrices)