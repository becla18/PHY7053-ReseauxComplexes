import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# dynamical parameters
R0_list = [0.5, 1.5, 2, 2.5, 3]
t0, t1 = (0, 20)
time = np.linspace(t0, t1, 1000)
i0 = [0.01]

i_t_list = []

for R0 in R0_list:
    # dynamical equation
    def i_dot(t, i, R0):
        return -i + R0*(1 - i)*i

    # integrate the equation
    sol = solve_ivp(i_dot, (t0, t1), i0, args=(R0,), t_eval = time)

    i_t_list.append(sol.y[0])

plt.figure()
for i, R0 in enumerate(R0_list[::-1]):
    plt.plot(time, i_t_list[-1-i], label=f"$R_0 = {R0}$")
    if R0 > 1:
        plt.plot(time, time*0+(1 - 1/R0), '--', color='#353535ff', linewidth=1.2)
plt.xlim(t0, t1)
plt.legend()
# plt.savefig('1e_a.pdf')
plt.show()
