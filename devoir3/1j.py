import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import factorial


# INTEGRATE MASS-ACTION SYSTEM

# dynamical parameters
R0_MA_list = [3]
t0, t1 = (0, 10)
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
k_max = 500
k = np.arange(0, k_max+1, 1)
print(k)
exp_k = 3.
k_dist = np.exp(-exp_k) * exp_k**(k) / factorial(k)
print(np.sum(k_dist))

# Theta function
def Theta(ik, exp_k, k):
    return 1/exp_k * np.sum(k * k_dist * ik)
    # return np.exp(-exp_k) * np.sum(k * exp_k**(k - 1) * ik / factorial(k))

# dynamical equations
def ik_dot(t, ik, R0, exp_k, k):
    return -ik + k * R0 * Theta(ik, exp_k, k) * (1 - ik)

dbmf_i_t_list = []

ik0 = np.array([0.01 for _ in k])

# ik_trajectories = []

for R0_MA in R0_MA_list:
    # compute DBMF R0 parameter
    R0 = R0_MA / exp_k * 1.2

    # integrate the equation
    sol = solve_ivp(ik_dot, (t0, t1), ik0, args=(R0, exp_k, k), t_eval = time)
    # equilibrium equation verification
    print(np.max(np.abs(((1 - sol.y[:, -1]) * R0 * k * Theta(sol.y[:, -1], exp_k, k)) - sol.y[:, -1])))
    # ik_trajectories.append(sol.y)
    dbmf_i_t_list.append(k_dist @ sol.y)


plt.figure()
for i, R0_MA in enumerate(R0_MA_list[::-1]):
    plt.plot(time, i_t_list[-1-i], label=f"$R_0 = {R0_MA}$")
    plt.plot(time, dbmf_i_t_list[-1-i], label=f"DBMF $R_0 = {R0_MA / exp_k * 1.2}$")
plt.xlim(t0, t1)
plt.legend()
# plt.savefig('1e_a.pdf')
plt.show()
