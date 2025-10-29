import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# dynamical parameters
R0_list = [3, 3, 3, 3]
i0_list = [0.01, 0.05, 0.1, 0.2]
t0, t1 = (0, 20)
time = np.linspace(t0, t1, 1000)
# i0 = 0.01
# state0 = [1 - i0, i0]

i_t_list = []

for i, R0 in enumerate(R0_list):
    i0 = i0_list[i]
    state0 = [1 - i0, i0]
    # dynamical equation
    def sir(t, state, R0):
        return np.array([-R0*state[0]*state[1], R0*state[0]*state[1] - state[1]])
    # integrate the equation
    sol = solve_ivp(sir, (t0, t1), state0, args=(R0,), t_eval = time)

    i_t_list.append(sol.y)

plt.figure()
for i, R0 in enumerate(R0_list[::-1]):
    plt.plot(i_t_list[-1-i][0], i_t_list[-1-i][1], linewidth=1, label=f"$i_0 = {i0_list[-1-i]}$")
plt.plot(np.linspace(0, 1, 10), 1-np.linspace(0, 1, 10), '--', linewidth=1, color="#353535ff", label='$i = -s$')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('$s$')
plt.ylabel('$i$')
plt.legend()
plt.savefig('2b_i0.pdf')
plt.show()
