import numpy as np
import matplotlib.pyplot as plt


# This script computes the values of the lagrange multipliers in a canonical configuration
# model of bipartite graphs for a specific degree sequences

# first, import the degree sequences
degrees1 = np.load("devoir4/sequence1.npy")
degrees2 = np.load("devoir4/sequence2.npy")

# number of nodes of each type and total number of links
N1 = np.size(degrees1)
N2 = np.size(degrees2)
L = np.sum(degrees1)

# define the iteration function for the fixed-point method (from Vallarano et al., 2021, Scientific Reports)
def iteration_fixed_point_BiCM(gamma, beta, degrees1, degrees2, N1, N2):
    # compute the new gamma multipliers
    denom_gamma = []
    for gamma_i in gamma:
        denom_gamma_i = 0
        for beta_a in beta:
            denom_gamma_i += np.exp(-beta_a) / (1 + np.exp(-gamma_i - beta_a))
        denom_gamma.append(denom_gamma_i)
    new_gamma = -np.log(degrees1 / denom_gamma)

    # compute the new beta multipliers
    denom_beta = []
    for beta_a in beta:
        denom_beta_a = 0
        for gamma_i in gamma:
            denom_beta_a += np.exp(-gamma_i) / (1 + np.exp(-gamma_i - beta_a))
        denom_beta.append(denom_beta_a)
    new_beta = -np.log(degrees2 / denom_beta) 

    return new_gamma, new_beta

# initial conditions for lagrange multipliers
gamma_0 = -np.log(degrees1 / np.sqrt(L))
beta_0 = -np.log(degrees2 / np.sqrt(L))

# do the iterative fixed-point process
nb_iters = 1000
gamma_iters = [gamma_0]
beta_iters = [beta_0]
for _ in range(nb_iters):
    gamma, beta = gamma_iters[-1], beta_iters[-1]
    new_gamma, new_beta = iteration_fixed_point_BiCM(gamma, beta, degrees1, degrees2, N1, N2)
    gamma_iters.append(new_gamma)
    beta_iters.append(new_beta)

# save the values of the lagrange multipliers as numpy array
np.save("devoir4/gamma.npy", gamma_iters[-1])
np.save("devoir4/beta.npy", beta_iters[-1])

# plot the values
# plt.figure()
# plt.plot(range(N1), gamma_iters[-1], '.', label='gamma')
# plt.plot(range(N2), beta_iters[-1], '.', label='beta')
# plt.plot(range(N1), degrees1, '.', label='degrees type 1')
# plt.plot(range(N2), degrees2, '.', label='degrees type 2')
# plt.legend()
# plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

# --- Left subplot: gamma + degrees1 ---
ax = axes[0]
ax.plot(range(N1), gamma_iters[-1], '.', label='gamma')
ax.plot(range(N1), degrees1, '.', label='degrees type 1')
ax.set_title("Gamma and Degrees 1")
ax.legend()

# --- Right subplot: beta + degrees2 ---
ax = axes[1]
ax.plot(range(N2), beta_iters[-1], '.', label='beta')
ax.plot(range(N2), degrees2, '.', label='degrees type 2')
ax.set_title("Beta and Degrees 2")
ax.legend()

plt.tight_layout()
plt.show()