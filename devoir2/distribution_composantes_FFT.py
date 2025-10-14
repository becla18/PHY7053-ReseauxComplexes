import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
from scipy.special import factorial, binom


# parametre du graphe
a_list = [0.1, 0.6]

# parametres de la methode
r = 1
n = 30

# generation des n points equidistants sur le cercle de rayon r
points = r*np.exp(1j*2*np.pi * np.array(range(n), dtype=complex) / n)


plt.figure(figsize=(6, 5))

for a in a_list:
    # calcul des valeurs de H_1(x)
    h1_valeurs = []
    for x in points:
        polynome = np.array([a**2, -2*a, 1, -x*(1-a)**2])
        racines = np.roots(polynome)
        # mettre les racines nulles en dehors du cercle unitaire
        racines[np.where(racines==0)] = 10
        print(racines)
        bon_indice = np.argmin(np.absolute(racines))
        h1_valeurs.append(racines[bon_indice])
        print(racines[bon_indice])
    h1_valeurs = np.array(h1_valeurs)

    # valeurs de H_0(x) a partir des valeurs de H_1(x)
    h0_valeurs = (1-a)*points / (1 - a * h1_valeurs)

    # transformee de fourier inverse
    tailles = np.array(range(n)) + 1
    dist_comp = 1/(n*r**tailles[:-1]) * fft.fft(h0_valeurs)[1:]

    # distribution exacte
    dist_comp_exacte = np.concatenate([[1-a], 1/(3*tailles[1:] - 2)*binom(3*tailles[1:] - 2, 2*tailles[1:] - 1) * (1-a)**(2*tailles[1:]-1) * a**(tailles[1:]-1)])

    print(np.sum(dist_comp_exacte))

    plt.plot(tailles[:10], dist_comp[:10], '+', label=f'FFT a={a}')
    plt.plot(tailles[:10], dist_comp_exacte[:10], 'x', label=f'Exacte a={a}')

plt.legend()
plt.ylabel('$\\pi_s$')
plt.xlabel('$s$')
plt.savefig('comparaison_fft.pdf')
plt.show()
