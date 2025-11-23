import numpy as np


# This script implements the Curveball algorithm to generate random bipartite graphs

def curveball_iter(B):
    # choose two rows randomly
    row1_index, row2_index = np.random.randint(0, B.shape[0]), np.random.randint(0, B.shape[0])
    row1, row2 = B[row1_index], B[row2_index]

    # find the nonzero elements in one row but not the other
    in1 = np.nonzero(row1)
    in2 = np.nonzero(row2)
    in1_not2 = np.setdiff1d(in1, in2)
    in2_not1 = np.setdiff1d(in2, in1)

    # check if a swap is possible
    min_length = np.min([np.size(in1_not2), np.size(in2_not1)]) 
    if min_length == 0:
        return B

    # choose a random number and combination of swaps
    if min_length == 1:
        n_swaps = 1
    else:
        n_swaps = np.random.randint(1, min_length+1)
    swaps_1 = np.random.choice(in1_not2, n_swaps, replace=False)
    swaps_2 = np.random.choice(in2_not1, n_swaps, replace=False)
    swaps = list(zip(swaps_1, swaps_2))

    # do the swaps
    for swap in swaps:
        # apply changes to row 1
        B[row1_index, swap[0]] = 0
        B[row1_index, swap[1]] = 1
        # apply changes to row 2
        B[row2_index, swap[0]] = 1
        B[row2_index, swap[1]] = 0
    
    return B