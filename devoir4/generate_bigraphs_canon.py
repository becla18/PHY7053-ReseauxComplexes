import numpy as np


# This script samples a given number of bipartite graphs from a canonical configuration model.

# import the desired lagrange multiplier values
gamma = np.load('devoir4/gamma.npy')
beta = np.load('devoir4/beta.npy')

# compute number of nodes of each type
N1 = np.size(gamma)
N2 = np.size(beta)

# compute the connexion probability matrix
p = []
for gamma_i in gamma:
    p_row = []
    for beta_a in beta:
        p_row.append(1 / (1 + np.exp(gamma_i + beta_a)))
    p.append(p_row)
p = np.array(p)

# sample a given number of graphs
n_graphs = 10**4
biadj_matrices = []
for _ in range(n_graphs):
    biadj_matrices.append((np.random.random((N1, N2)) < p).astype(int))

# verify that the probabilites are respected
print(np.max(np.abs(sum(biadj_matrices) / len(biadj_matrices) - p)))

# save the sampled biadjacency matrices
np.save("devoir4/canon_biadj_matrices.npy", biadj_matrices)