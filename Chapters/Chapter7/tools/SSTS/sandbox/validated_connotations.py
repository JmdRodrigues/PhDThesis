from GrammarofTime.SSTS.sandbox.connotation_tools import *
from GrammarofTime.SSTS.backend import gotstools as gt
from pyts.approximation import paa

def AmplitudeTrans(s, n, char_list, levels_quant=[0.25, 0.5, 0.75], thr = 0.0, method="distribution"):
    """
    Divide the amplitude of a signal in n levels quantized as letters from A to Z
    :param s: signal - as an array
    :param n: number of levels
    :param char_list: list with the chars in which the level is converted
    :param method: "distribution" - uses the histogram to take the bins
                   "quantiles"    - uses the quantile technique
                   "custom"       - uses 1 level and threshold value
    :return: string of levels
    TODO: levels should have a dimension 1 less than the length of char-list
    """

    if(method =="distribution"):
        hist, bins = np.histogram(s, bins=n)
        levels = bins
    elif(method == "quantiles"):
        levels = np.quantile(s, levels_quant)
        print(levels)
    elif(method == "custom"):
        levels = thr*(max(abs(s)))

    return quantilstatesArray2(s, levels, char_list, conc=False)

def D1Trans(s, t, signs=["n", "z", "p"]):
    """
    Translate the signal based on the derivative (p, n and z). Optionally, the amplitude of the
    derivative can also be used.
    :param s: signal - as an array
    :param n: number of levels for the amplitude:
                0 - no levels, only derivative
                =>1 - number of levels + amplitude of derivative
    :return: string of derivative(+ Amplitude)
    """

    ds1 = np.diff(s)
    x = np.empty(len(s), dtype=str)
    thr = (np.max(ds1) - np.min(ds1)) * t
    x[np.where(ds1 <= -thr)[0]] = signs[0]
    x[np.where(np.all([ds1 <= thr, ds1 >= -thr], axis=0))[0]] = signs[1]
    x[np.where(thr <= ds1)[0]] = signs[2]
    x[-1] = x[-2]

    return x

def AmpChange(s, thr, method="absolute"):
    """
    Defines areas of the signal where the changes in rising or falling are bigger or not. In the case of having
    higher changes, the char will be uppercase, in case of being a small amplitude change, it will be lowcase:
        - R - high rising
        - F - high falling
        - r - low rising
        - f - low falling
    :param s: signal
    :param thr: threshold. Value between [0-1] and multiplied by the maximum of amp difference found in the signal
    :param char_list: list of chars to use
    :param method: differentiate - threshold applied differently between rising and falling
                   absolute - threshold applied to the absolute value of rising and falling
    :return:
    """

    ds_str, ds_list_str, constr_list = gt.connotation([s], "D1 0.001")
    print("done with diff connotation")

    pos_matchs = gt.symbolic_search(ds_str, ds_list_str, "p+")
    neg_matchs = gt.symbolic_search(ds_str, ds_list_str, "n+")
    z_matches = gt.symbolic_search(ds_str, ds_list_str, "z+")

    print("done with matches")
    # gt.plot_matches(s, [pos_matchs, neg_matchs], color=["green", "orange"], mode="span")

    vals = np.zeros(len(s))
    vals_str = np.empty(len(s), dtype='<U5')


    #Get maximum difference values to use the thresholding
    if(method == "absolute"):
        max_ampdiff_pos = max_AmpChange(s, [pos_matchs, neg_matchs])
        max_ampdiff_neg = max_ampdiff_pos
    elif(method == "differentiate"):
        max_ampdiff_pos = max_AmpChange(s, [pos_matchs])
        max_ampdiff_neg = max_AmpChange(s, [neg_matchs])

    #Get match sequence of amplitudes difference for rise and fall both numerical and textual
    for p_match in pos_matchs:
        vals[p_match[0]:p_match[1]-1] = abs(s[p_match[0]]-s[p_match[1]-1])
        vals_str[p_match[0]:p_match[1]] = quantilstatesArray2(vals[p_match[0]:p_match[1]], thr * max_ampdiff_pos, ["r", "R"], conc=False)

    for z_match in z_matches:
        vals[z_match[0]:z_match[1]] = 0
        vals_str[z_match[0]:z_match[1]] = "z"

    for n_match in neg_matchs:
        vals[n_match[0]:n_match[1]] = abs(s[n_match[0]]-s[n_match[1]-1])
        vals_str[n_match[0]:n_match[1]] = quantilstatesArray2(vals[n_match[0]:n_match[1]], thr * max_ampdiff_neg, ["f", "F"], conc=False)

    # print(vals_str)
    return vals_str

def D1Speed(s, thr):
    """
    Design a string based on quick/slow rises/falls on the signal.
    Q or q --> for quick or slow derivative if higher or lower trend
    :param s: signal
    :return: string
    """

    ds = np.diff(s)
    ds = np.append(ds, ds[-1])
    x = AmplitudeTrans(abs(ds), 2, ["q", "Q"], thr = thr, method="custom")

    return x


# def D1Speed_q(s, thr, mode="absolute"):
#     """
#     Design a string based on quick/slow rises/falls and High/low rises/falls on the signal.
#     Falling: f/F --> f for low fall and F for high fall
#     Rising: r/R --> r for low rise and R for high rise
#     Q or S --> for quick or slow derivative if higher or lower trend
#     :param s: signal
#     :return: string
#     """
#     ds = get_slope_segments(s)
#     str_ds = np.empty(len(ds), dtype=str)
#
#     if(mode=="absolute"):
#
#         slope = np.abs(ds)
#         str_ds = AmplitudeTrans(slope, n=2, levels_quant=[0.5, 0.75], char_list=["_", "q", "Q"], method="quantiles")
#
#     elif(mode=="separated"):
#
#         pos_slope = np.where(ds > 0)[0]
#         neg_slope = np.where(ds < 0)[0]
#
#         str_ds_pos = AmplitudeTrans(ds[pos_slope], n=2, levels_quant=[0.5], char_list=["p", "P"], method="quantiles")
#         str_ds_neg = AmplitudeTrans(abs(ds[neg_slope]), n=2, levels_quant=[0.5], char_list=["n", "N"],
#                                     method="quantiles")
#
#         str_ds[pos_slope] = str_ds_pos
#         str_ds[neg_slope] = str_ds_neg
#
#     elif(mode=="threshold"):
#         slope = np.abs(ds)
#         str_ds = AmplitudeTrans(slope, n=2, thr=thr, char_list=["q", "Q"], method="custom")
#
#     return str_ds

def D1_concavity(s):
    dds = get_concavity_per_segment(s)

    str_ds = np.where(dds==1, "Convex", "Concave")

    return str_ds

def D1_linearity(s, thr):
    dds = get_difference_between_linear_segment(s)

    str_ds = AmplitudeTrans(dds, n=2, char_list=["L", "H"], thr=thr, method="custom")

    return str_ds