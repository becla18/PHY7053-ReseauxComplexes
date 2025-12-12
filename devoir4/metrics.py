import numpy as np
from scipy.special import xlogy


# define the various metrics and related functions

# assortativity
def assortativity_degrees(B):
    # compute the degree sequences and total number of links
    M = np.sum(B)
    degrees1 = np.sum(B, axis=1)
    degrees2 = np.sum(B, axis=0)

    # compute the means over the links
    mu1 = 1/M * np.sum(degrees1 ** 2)
    mu2 = 1/M * np.sum(degrees2 ** 2)

    # compute the assortativity coefficient
    num = degrees1[:, np.newaxis].T @ B @ degrees2[:, np.newaxis] - M * mu1*mu2
    denom = np.sqrt(np.sum(degrees1**3) - M*mu1**2) * np.sqrt(np.sum(degrees2**3) - M*mu2**2) 

    return (num / denom)[0, 0]

# spectral radius
def spectral_radius(M):
    return np.max(np.abs(np.linalg.eigvals(M)))

# NODF nestedness index
def nodf(B):
    # compute the degree sequences
    degrees1 = np.sum(B, axis=1)
    degrees2 = np.sum(B, axis=0)

    # sort the columns and rows according for decreasing degrees
    B = B[np.argsort(-degrees1), :]
    B = B[:, np.argsort(-degrees2)]

    # compute necessary parameters and constants
    N1, N2 = B.shape
    K = (N1 * (N1 - 1) + N2 * (N2 - 1)) / 200

    # compute the nodf
    nodf = 0
    # compute the first term
    for j in range(N1):
        for i in range(j):
            nodf += (1 - np.heaviside(degrees1[j] - degrees1[i], 1)) * (B[i] @ B[j].T) / degrees1[j]
    # compute the second term
    for l in range(N2):
        for k in range(l):
            nodf += (1 - np.heaviside(degrees2[l] - degrees2[k], 1)) * (B[:, k].T @ B[:, l]) / degrees2[l]
    # add normalization constant
    nodf = 1/K * nodf

    return nodf

# stable NODF nestedness index
def stable_nodf(B):
    # compute the degree sequences
    degrees1 = np.sum(B, axis=1)
    degrees2 = np.sum(B, axis=0)

    # sort the columns and rows according for decreasing degrees
    B = B[np.argsort(-degrees1), :]
    B = B[:, np.argsort(-degrees2)]

    # compute necessary parameters and constants
    N1, N2 = B.shape
    K = (N1 * (N1 - 1) + N2 * (N2 - 1)) / 200

    # compute the nodf
    nodf = 0
    # compute the first term
    for j in range(N1):
        for i in range(j):
            nodf += (B[i] @ B[j].T) / degrees1[j]
    # compute the second term
    for l in range(N2):
        for k in range(l):
            nodf += (B[:, k].T @ B[:, l]) / degrees2[l]
    # add normalization constant
    nodf = 1/K * nodf

    return nodf
 
def motifs5(B):
    Z = B @ B.T
    Y = B @ (np.ones(B.shape) - B).T
    return np.trace(Z @ Y.T)

def motifs6(B):
    Z = B @ B.T
    Y = B @ (np.ones(B.shape) - B).T
    d = np.sum(B, axis=1)
    return 1/4 * np.trace(Z @ (Z - np.ones(Z.shape)).T) - 1/4 * np.dot(d, (d - np.ones(d.shape)))

def niche_overlap(B):
    # compute number of nodes and degree sequences
    N1, N2 = B.shape
    d1 = np.sum(B, axis=1)
    d2 = np.sum(B, axis=0)

    B_over_d = B / np.outer(np.ones(B.shape[0]), d2)
    
    Ro = 0
    for alpha in range(N2):
        for delta in np.arange(alpha, N2, 1):
            al = B_over_d[:, alpha] 
            de = B_over_d[:, delta] 
            Ro += 1 / (2*np.log(2)) * np.sum(xlogy(al + de, al + de) - xlogy(al, al) - xlogy(de, de))
    
    return 2 / (N2 * (N2 - 1)) * Ro