import numpy as np
import matplotlib.pyplot as plt


# fonction pour calculer S selon a
def calculer_S(a):
    u3 = (2 - a - np.sqrt(a*(4 - 3*a))) / (2*a)
    return 1 - (1-a) / (1 - a*u3)

# valeurs de a
a = np.linspace(0, 1, 1000)


# figure
plt.figure(figsize=(2.5, 2))
plt.plot(a, np.zeros(len(a)), color='#353535ff', linewidth=1)
plt.axvline(1/3, linestyle='--', color='#353535ff')
plt.plot(a, calculer_S(a), color='#2c56a3ff')
plt.xlim(0, a[-1])
plt.ylim(-1, 1)
plt.xlabel('$a$')
plt.ylabel('$S$', rotation = 0)
plt.savefig('diag_bif_S.svg')
plt.show()

