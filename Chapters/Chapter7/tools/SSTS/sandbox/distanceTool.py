from GrammarofTime.SSTS.backend import gotstools as gt
from GrammarofTime.SSTS.sandbox.alignmentTool import compute_accumulated_cost_matrix, compute_optimal_warping_path, compute_optimal_warping_path_for_LCS
import numpy as np
from tools.string_processing_tools import *

from tools.plot_tools import *
from dtw import *

def linearDist(sig1,sig2, con="derivative"):
    """
    Linear distance based on string with the same size. In this case, use Hamming distance, because it is quick and does
    not search for changes regarding deletions or additions. Just substitutions.
    :param sig1:
    :param sig2:
    :param con:
    :return:
    """
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


    # a_rle = WindowRLEString(a_s, len(sig1)//10)
    # b_rle = WindowRLEString(b_s, len(sig2)//10)
    #
    # dist_list = np.zeros(len(a_rle))
    # for i, (a, b) in enumerate(zip(a_rle, b_rle)):
    #     dist_list[i] = textdistance.levenshtein(a, b) / (len(a) + len(b))

    # return np.sum(dist_list)
    return textdistance.hamming(a_s, b_s) / (len(a_s) + len(b_s))

def slidingDist(sig, query, con="derivative", dist="hamming"):
    if (con == "derivative"):
        a_s = gt.connotation([sig], "D1 0.01")
        a_s = a_s[-1]
        b_s = gt.connotation([query], "D1 0.01")
        b_s = b_s[-1]

    elif (con == "amp"):
        sub_amp_max = max([max(sig), max(query)])

        thr1 = str((max(sig)/sub_amp_max))
        thr2 = str((max(query)/sub_amp_max))

        a_s = gt.connotation([sig], "A "+thr1)
        a_s = a_s[-1]
        b_s = gt.connotation([query], "A "+thr2)
        b_s = b_s[-1]
        print(a_s)
        print(b_s)
    elif (con == "ampQ"):
        a_s = gt.connotation([sig], "AQ 5")
        a_s = a_s[-1]
        b_s = gt.connotation([query], "AQ 5")
        b_s = b_s[-1]

    elif (con == "ampdiff"):
        a_s = gt.connotation([sig], "AD 0.75")
        a_s = a_s[-1]
        b_s = gt.connotation([query], "AD 0.75")
        b_s = b_s[-1]
    else:
        a_s = gt.connotation([sig], "A 0.5 D1 0.025")
        a_s = a_s[-1]
        b_s = gt.connotation([query], "A 0.5 D1 0.025")
        b_s = b_s[-1]

    dist_l = WindowTextDist(a_s, b_s, dist)

    return dist_l

def WindowTextDist(str_1, query, dist):
    output = np.zeros(len(str_1))
    window_len = len(query)
    WinRange = int(window_len / 2)

    # print(inputString[-1:len(inputString) - WinRange:-1])

    if (isinstance(str_1, np.ndarray)):
        stringS = np.r_[str_1[WinRange:0:-1], str_1, str_1[-1:len(str_1) - WinRange:-1]]
    elif (isinstance(str_1, list)):
        stringS = str_1[WinRange:0:-1] + str_1 + str_1[-1:len(str_1) - WinRange:-1]

    # query_rle = lstStr2Str(runLengthEncoding(query)[1])

    for i in range(int(WinRange), len(stringS) - int(WinRange) + 1):
        substring = stringS[i - WinRange:WinRange + i]
        # substring_rle = lstStr2Str(runLengthEncoding(substring)[1])
        if(dist=="hamming"):
            # dist_i = textdistance.hamming(substring_rle, query_rle) / (len(substring_rle) + len(query_rle))
            dist_i = textdistance.hamming(substring, query) / (len(substring) + len(query))
        elif(dist=="lcs"):
            # dist_i = 2*dynamic_lcs(substring_rle, query_rle) / (len(substring_rle) + len(query_rle))
            dist_i = 1 - 2*dynamic_lcs(substring, query) / (len(substring) + len(query))
        else:
            dist_i = 0

        output[i - int(WinRange)] = dist_i

    return output

def ldist(s1,s2):
    dist_list = np.zeros(len(s1))
    for i, (a, b) in enumerate(zip(s1, s2)):
        dist_list[i] = textdistance.levenshtein(a, b) / (len(a) + len(b))

    return np.sum(dist_list)

def lcs_test(sig,query, con="derivative"):
    if (con == "derivative"):
        a_s = gt.connotation([sig], "D1 0.01")
        a_s = a_s[-1]
        b_s = gt.connotation([query], "D1 0.01")
        b_s = b_s[-1]

    elif (con == "amp"):
        a_s = gt.connotation([sig], "A 0.5")
        a_s = a_s[-1]
        b_s = gt.connotation([query], "A 0.5")
        b_s = b_s[-1]

    elif (con == "ampQ"):
        a_s = gt.connotation([sig], "AQ 5")
        a_s = a_s[-1]
        b_s = gt.connotation([query], "AQ 5")
        b_s = b_s[-1]

    elif (con == "ampdiff"):
        a_s = gt.connotation([sig], "AD 0.75")
        a_s = a_s[-1]
        b_s = gt.connotation([query], "AD 0.75")
        b_s = b_s[-1]
    else:
        a_s = gt.connotation([sig], "AQ 6 D1 0.025")
        a_s = a_s[-1]
        b_s = gt.connotation([query], "AQ 6 D1 0.025")
        b_s = b_s[-1]
    #
    # a = lstStr2Str(a_s, space="")
    # b = lstStr2Str(b_s, space="")

    L = dynamic_lcs_accumulated(a_s, b_s)

    # # get distance score
    d = L.transpose()[-1, -1]

    P = compute_optimal_warping_path_for_LCS(L)

    plotAlignement2(sig, query, P)

    ImagenandXYplots(L, sig, query, P)

    return L, L[-1,-1]





#Example in how to use lcs for data alignment---------------------------------------------------------------------------
x1 = np.linspace(0, 10, 500)
x2 = np.linspace(0, 10, 500)


#Exemplo1
# sig1 = np.sin(x1)
# sig2 = np.sin(x2)


#Exemplo2
# sig1 = np.sin(x1)
# sig2 = np.cos(x2)

#Exemplo3
sig1 = np.sin(x1)+np.sin(2*x1) + np.sin(3*x1)
sig2 = np.sin(x2)+np.sin(2*x2) + np.sin(2.5*x2)
#
# dist_mat, d = lcs_test(sig1, sig2, con="AQ")
dist_mat2, d2 = lcs_test(sig1, sig2, con="derivative")
# print("LCS")
# print(d/(len(sig1)+len(sig2)))
# print(d2/(len(sig1)+len(sig2)))
#
alignment = dtw(np.diff(sig1), np.diff(sig2), keep_internals=True,
    step_pattern=rabinerJuangStepPattern(6, "c"))
#
## Display the warping curve, i.e. the alignment curve
alignment.plot(type="twoway", offset=2)
dd = alignment.distance
print(dd/(len(sig1)+len(sig2)))
plt.show()

#-----------------------------------------------------------------------------------------------------------------------

#Example in how to use a distance measure along the signal
# x1 = np.linspace(0, 25, 1000)
# x2 = np.linspace(0, 5, 200)
# sig1 = np.sin(x1)
# sig2 = np.sin(x2)
# sig1 = 1*np.sin(x1)
# sig2 = np.cos(x2)
# sig1 = np.sin(x1)+np.sin(2*x1) + np.sin(3*x1)
# # sig2 = np.sin(x2)+np.sin(2*x2) + np.sin(3*x2)
# sig2 = np.sin(x2)+np.sin(2*x2) + np.sin(2.5*x2)
#
# fig, axs = plt.subplots(3,1, sharex=True)
# axs[0].plot(sig1)
# axs[0].set_title("Signal")
# axs[1].plot(sig2)
# axs[1].set_title("Query")

# dham_d = slidingDist(sig1, sig2, con="derivative", dist="hamming")
# dham_aq = slidingDist(sig1, sig2, con="ampQ", dist="hamming")
# dlcs_d = slidingDist(sig1, sig2, con="derivative", dist="lcs")
# dlcs_aq = slidingDist(sig1, sig2, con="ampQ", dist="lcs")

# axs[2].plot(dham_d, label="ham_dif")
# axs[2].plot(dham_aq, label="ham_amp")
# axs[2].plot(dlcs_d, label="lcs_dif")
# axs[2].plot(dlcs_aq, label="lcs_amp")
# axs[2].set_title("Distance")
# plt.legend()
#
# plt.show()
