from GrammarofTime.SSTS.backend import gotstools as gt
import numpy as np
from tools.string_processing_tools import *
from tools.plot_tools import *
import time

from dtw import *

def distStr(sig1, sig2, con = "derivative"):
    if(con=="derivative"):
        a_s = gt.connotation([sig1], "D1 0.05")
        a_s = a_s[-1]
        b_s = gt.connotation([sig2], "D1 0.05")
        b_s = b_s[-1]

    else:
        a_s = gt.connotation([sig1], "A 0.5 D1 0.025")
        a_s = a_s[-1]
        b_s = gt.connotation([sig2], "A 0.5 D1 0.025")
        b_s = b_s[-1]

    a_rle = WindowRLEString(a_s, len(sig1)//10)
    b_rle = WindowRLEString(b_s, len(sig2)//10)

    # print("a_rle:")
    # print(a_rle[-1])
    # print(len(b_rle))

    # result_join1, words1, counts1 = runLengthEncoding(a_s)
    # result_join2, words2, counts2 = runLengthEncoding(b_s)

    # print(words1)
    # print(words2)
    # print(len(words1))
    # print(len(words2))
    # max_lcs = max(len(words1), len(words2))
    # print(max_lcs)
    #get cost matrix

    print(a_rle)
    print(b_rle)

    similarity = StringDistance(a_rle, b_rle)

    # dist_ab = 1 - similarity/len(a_rle)

    acc_dist = compute_accumulated_cost_matrix(similarity)

    #get distance score
    d = acc_dist[-1, -1]

    P = compute_optimal_warping_path(acc_dist)

    plotAlignement2(sig1, sig2, P)

    ImagenandXYplots(acc_dist, sig1, sig2, P)

    return acc_dist, d


def compute_accumulated_cost_matrix(C):
    """Compute the accumulated cost matrix given the cost matrix
    https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3S2_DTWbasic.html
    Notebook: C3/C3S2_DTWbasic.ipynb

    Args:
        C: cost matrix

    Returns:
        D: Accumulated cost matrix
    """
    N = C.shape[0]
    M = C.shape[1]
    D = np.zeros((N, M))
    D[0, 0] = C[0, 0]
    for n in range(1, N):
        D[n, 0] = D[n-1, 0] + C[n, 0]
    for m in range(1, M):
        D[0, m] = D[0, m-1] + C[0, m]
    for n in range(1, N):
        for m in range(1, M):
            D[n, m] = C[n, m] + min(D[n-1, m], D[n, m-1], D[n-1, m-1])
    return D

def compute_optimal_warping_path_for_LCS(D):
    """Compute the warping path given an accumulated cost matrix

    Notebook: C3/C3S2_DTWbasic.ipynb

    Args:
        D: Accumulated cost matrix

    Returns
        P: Warping path (list of index pairs)
    """
    N = D.shape[0]
    M = D.shape[1]
    n = N - 1
    m = M - 1
    P = [(n, m)]
    while n > 0 or m > 0:
        if n == 0:
            cell = (0, m - 1)
        elif m == 0:
            cell = (n - 1, 0)
        else:
            val = max(D[n-1, m-1], D[n-1, m], D[n, m-1])
            if val == D[n-1, m-1]:
                cell = (n-1, m-1)
            elif val == D[n-1, m]:
                cell = (n-1, m)
            else:
                cell = (n, m-1)
        P.append(cell)
        (n, m) = cell
    P.reverse()
    return np.array(P)


def compute_optimal_warping_path(D):
    """Compute the warping path given an accumulated cost matrix

    Notebook: C3/C3S2_DTWbasic.ipynb

    Args:
        D: Accumulated cost matrix

    Returns
        P: Warping path (list of index pairs)
    """
    N = D.shape[0]
    M = D.shape[1]
    n = N - 1
    m = M - 1
    P = [(n, m)]
    while n > 0 or m > 0:
        if n == 0:
            cell = (0, m - 1)
        elif m == 0:
            cell = (n - 1, 0)
        else:
            val = min(D[n-1, m-1], D[n-1, m], D[n, m-1])
            if val == D[n-1, m-1]:
                cell = (n-1, m-1)
            elif val == D[n-1, m]:
                cell = (n-1, m)
            else:
                cell = (n, m-1)
        P.append(cell)
        (n, m) = cell
    P.reverse()
    return np.array(P)


# x1 = np.linspace(0, 10, 100)
# x2 = np.linspace(0, 10, 25)
# sig1 = np.sin(x1)
# sig2 = np.sin(x2)
#
# distStr(sig1, sig2, con="else")
#
# alignment = dtw(sig1, sig2, keep_internals=True)
# # ## Display the warping curve, i.e. the alignment curve
# # alignment.plot(type="alignment")
#
# ## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
# dtw(sig1, sig2, keep_internals=True,
#     step_pattern=rabinerJuangStepPattern(6, "c"))\
#     .plot(type="twoway",offset=-2)
#
# plt.show()
