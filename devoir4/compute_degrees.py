import numpy as np


# import the empirical network of interest
B = np.genfromtxt('devoir4/plant-pollinator-biadjacency-matrix.csv', delimiter=',')

# compute the degree sequence for both types of nodes
degrees_type1 = np.sum(B, axis=1)
degrees_type2 = np.sum(B, axis=0)

# save degree sequences as numpy arrays
np.save("devoir4/sequence1.npy", degrees_type1)
np.save("devoir4/sequence2.npy", degrees_type2)