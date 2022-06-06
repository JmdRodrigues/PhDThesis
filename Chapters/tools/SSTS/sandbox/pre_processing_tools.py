from GrammarofTime.SSTS.sandbox.validated_connotations import get_dif_segments
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def compute_linear_segment(slope, size, y):
    t = np.linspace(0, size, size)
    s_i = slope*t + y

    return s_i

def linear_approx(s, win):
    """
    linear approximates by window
    :param s:
    :param win:
    :return:
    """


    inputSignal = s - np.mean(s)

    WinRange = int(win / 2)

    output = np.zeros(len(s))

    num_segments = len(s)//WinRange

    for i in range(1, num_segments+1):
        if i == 1:
            segment = s[:i*WinRange]
            slope_i = (segment[-1] - segment[0]) / len(segment)
            output[:i*WinRange] = compute_linear_segment(slope_i, len(segment), segment[0])

        elif (i == num_segments-1):
            segment = s[i * WinRange:]
            slope_i = (segment[-1] - segment[0]) / len(segment)
            output[i * WinRange:] = compute_linear_segment(slope_i, len(segment), segment[0])

        else:
            segment = s[(i-1)*WinRange:i*WinRange]
            slope_i = (segment[-1] - segment[0]) / len(segment)
            output[(i-1)*WinRange:i*WinRange] = compute_linear_segment(slope_i, len(segment), segment[0])

    return output

def linearization(s, thr=0.2):
    """
    performs a linearization of signals based on the inflection points of time series
    :param s: signal
    :return: signal linearized
    """
    matches = get_dif_segments(s, thr)
    l_s = np.zeros(len(s))
    p_l = []
    for i, match in enumerate(matches):
        if i == 0:
            segment = s[:match[0]]
            slope_i = (segment[-1] - segment[0]) / len(segment)
            l_s[:match[0]] = compute_linear_segment(slope_i, len(segment), segment[0])

        elif (i == len(matches) - 1):
            segment = s[match[1]:]
            slope_i = (segment[-1] - segment[0]) / len(segment)
            l_s[match[1]:] = compute_linear_segment(slope_i, len(segment), segment[0])

        segment = s[match[0]:match[1]]
        slope_i = (segment[-1] - segment[0]) / len(segment)
        l_s[match[0]:match[1]] = compute_linear_segment(slope_i, len(segment), segment[0])

    return l_s



#
# t = np.linspace(0, 20, 200000)
# s = np.sin(2*np.pi*t)
# s2 = np.sin(2*np.pi*4*t)
#
# win = 10
# num_segments = (len(t)//win)+1
# l_s = linearization(s+s2)
#
# plt.plot(t, s+s2)
# plt.plot(t, l_s)
# plt.show()