import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import factorial


# INTEGRATE MASS-ACTION SYSTEM

# dynamical parameters
R0_MA_list = [0.5, 2, 3]
t0, t1 = (0, 6)
time = np.linspace(t0, t1, 1000)
i0 = [0.01]

# dynamical equation
def i_dot(t, i, R0):
    return -i + R0*(1 - i)*i

i_t_list = []

for R0 in R0_MA_list:
    # integrate the equation
    sol = solve_ivp(i_dot, (t0, t1), i0, args=(R0,), t_eval = time)
    i_t_list.append(sol.y[0])


# INTEGRATE DBFM SYSTEM

# set max degree, expected degree and generate distribution
k_max = 10
k = np.arange(0, k_max+1, 1)
print(k)
exp_k = 2.
k_dist = [0 for _ in k]
k_dist[int(exp_k)] = 1.
k_dist = np.array(k_dist)
print(k_dist)
print(np.sum(k_dist))

# Theta function
def Theta(ik, exp_k, k):
    return 1/exp_k * np.sum(k * k_dist * ik)

# dynamical equations
def ik_dot(t, ik, R0, exp_k, k):
    return -ik + k * R0 * Theta(ik, exp_k, k) * (1 - ik)

dbmf_i_t_list = []

ik0 = np.array([0.01 for _ in k])

ik_trajectories = []

for R0_MA in R0_MA_list:
    # compute DBMF R0 parameter
    R0 = R0_MA / exp_k

    # integrate the equation
    sol = solve_ivp(ik_dot, (t0, t1), ik0, args=(R0, exp_k, k), t_eval = time, rtol=1e-10)
    # equilibrium equation verification
    eq_point_k = k * R0 * Theta(sol.y[:, -1], exp_k, k)/(1 + k*R0* Theta(sol.y[:, -1], exp_k, k))
    eq_point = np.dot(k_dist, k * R0 * Theta(sol.y[:, -1], exp_k, k)/(1 + k*R0* Theta(sol.y[:, -1], exp_k, k)))
    # print(np.max(np.abs(k * R0 * Theta(sol.y[:, -1], exp_k, k)/(1 + k*R0* Theta(sol.y[:, -1], exp_k, k)) - sol.y[:, -1])))
    ik_trajectories.append(sol.y)
    dbmf_i_t_list.append(k_dist @ sol.y)

ik_trajectories = np.array(ik_trajectories)

plt.figure()
for i, R0_MA in enumerate(R0_MA_list[::-1]):
    plt.plot(time, i_t_list[-1-i], label=f"Action de masse ($R_0 = {R0_MA})$")
    plt.plot(time, dbmf_i_t_list[-1-i], '--', color='#353535ff', label=f"Champ moyen hétérogène ($R_0 = {R0_MA / exp_k}$)")
plt.xlim(t0, t1)
plt.ylabel('$i$')
plt.xlabel('$t$')
plt.legend()
plt.savefig('1j_b.pdf')
plt.show()
