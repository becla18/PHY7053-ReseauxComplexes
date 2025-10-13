import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
from scipy.special import factorial


# parametre du graphe
a = 0.5

# parametres de la methode
r = 1
n = 1000

# generation des n points equidistants sur le cercle de rayon r
points = r*np.exp(1j*2*np.pi * np.array(range(n), dtype=complex) / n)

# calcul des valeurs de H_1(x)
h1_valeurs = []
for x in points:
    polynome = np.array([a**2, -2*a, 1, -x*(1-a)**2])
    racines = np.roots(polynome)
    h1_valeurs.append(racines[0])
h1_valeurs = np.array(h1_valeurs)

# valeurs de H_0(x) a partir des valeurs de H_1(x)
h0_valeurs = (1-a)*points / (1 - a * h1_valeurs)

# transformee de fourier inverse
tailles = np.array(range(n)) + 1

dist_comp = fft.ifft(h0_valeurs)

tailles = np.array(range(len(dist_comp))) + 1

# distribution exacte
dist_comp_exacte = np.concatenate([[1-a], factorial(3*tailles[1:] - 3) / (factorial(2*tailles[1:]-1)*factorial(tailles[1:] - 1)) * (1-a)**(2*tailles[1:]-1) * a**(tailles[1:]-1)])

nmax_visible = 100
plt.plot(tailles[0:nmax_visible], np.abs(dist_comp[0:nmax_visible]/dist_comp[0]), label='méthode FFT')
plt.plot(tailles[0:nmax_visible], dist_comp_exacte[0:nmax_visible]/dist_comp_exacte[0], label='distribution exacte')
plt.legend()
plt.show()
