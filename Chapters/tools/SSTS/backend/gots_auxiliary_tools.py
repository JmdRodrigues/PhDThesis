
import numpy as np
import matplotlib.pyplot as plt

from PeakFinder import detect_peaks

def second_derivative(s):
    """
    computes the second derivative of signal s
    :param s:
    :return:
    """
    ds = np.diff(s)
    ds = np.append(ds, ds[-1])
    dds = np.diff(ds)
    dds = np.append(dds, dds[-1])

    return dds

def quantilestates2(sample, quantile_vals, char_list):
    if(np.size(quantile_vals) == 1):
        sample_quantiles = [quantile_vals.tolist()]
    else:
        sample_quantiles = quantile_vals.tolist()

    sample_quantiles.append(sample)
    return char_list[list(np.sort(sample_quantiles)).index(sample)]

def quantilstatesArray2(signal, quantile_vals, char_list, conc=True):
    if(conc):
        return "".join([quantilestates2(sample, quantile_vals, char_list) for sample in signal])
    else:
        out_str = [quantilestates2(sample, quantile_vals, char_list) for sample in signal]

        return out_str


def max_AmpChange(s, m):
    """

    :param s: signal
    :param m: list of list of matches
    :return: value of maximum amplitude difference
    """

    max_ampdif = 0


    for matches in m:
        for match in matches:
            if(max_ampdif<abs(s[match[0]-1] - s[match[1]-1])):
                 max_ampdif = abs(s[match[0]-1] - s[match[1]-1])


    return max_ampdif


def get_dif_segments(s, thr):

    ds = np.abs(second_derivative(s))

    ds_peaks = detect_peaks(ds, mph=thr*max(ds))
    ds_match = [(a, b) for a, b in zip(ds_peaks[:-1], ds_peaks[1:])]
    ds_match.append((ds_match[-1][-1], len(s)))
    ds_match.insert(0, (0, ds_match[0][0]))

    return ds_match

def get_slope_segments(s, thr=0.2):
    matches = get_dif_segments(s, thr)
    p_l = []
    for i, match in enumerate(matches):
        segment = s[match[0]:match[1]]
        slope_i = (segment[-1]-segment[0])/len(segment)
        p_l.append([slope_i]*len(segment))

    # plt.plot([match[0] for match in matches], s[[match[0] for match in matches]], 'o')
    # plt.plot(np.abs(500*np.concatenate(p_l)))
    # plt.plot(s)
    # plt.show()

    return np.concatenate(p_l)

def abs_distance_2_linear(segment):
    x = np.linspace(0, len(segment), len(segment))
    z = np.polyfit(x, segment, 1)
    p = np.poly1d(z)
    linear_apprx = p(x)

    return np.sum(np.abs(linear_apprx-segment))

def distance_2_linear(segment):
    x = np.linspace(0, len(segment), len(segment))
    z = np.polyfit(x, segment, 1)
    p = np.poly1d(z)
    linear_apprx = p(x)

    return np.sum(linear_apprx-segment)


def get_difference_between_linear_segment(s, thr=0.2):
    """
    Distances to a linear approximation of segments that are separated based on the
    peaks of the absolute second derivative.
    :param s: signal
    :return: distances per segment, with the same size as the original signal
    """
    matches = get_dif_segments(s, thr)
    distances_ = []
    for i, match in enumerate(matches):
        segment = s[match[0]:match[1]]
        # calculate distances to fitted data
        distances_.append([distance_2_linear(segment)]*len(segment))

    plt.plot([match[0] for match in matches], s[[match[0] for match in matches]], 'o')
    plt.plot(np.abs(np.concatenate(distances_)))
    plt.plot(s)
    plt.show()

    return np.concatenate(distances_)

def get_concavity_per_segment(s, thr=0.2):
    """
    Computes the concavity of the signal by means of the second derivative.
    When the second derivative is negative, the segment is concave, and when positive, the segment is convex
    :param s:
    :return:
    """
    matches = get_dif_segments(s, thr)
    dds = second_derivative(s)

    concavity = []

    for i, match in enumerate(matches):
        ddsegment = dds[match[0]:match[1]]
        concavity.append([np.sign(np.sum(np.sign(ddsegment)))]*len(ddsegment))

    plt.plot([match[0] for match in matches], s[[match[0] for match in matches]], 'o')
    plt.plot((np.concatenate(concavity)))
    plt.plot(s)
    plt.show()

    return np.concatenate(concavity)

def get_concavity_absolute(s):
    dss = second_derivative(s)

    return np.sign(dss)