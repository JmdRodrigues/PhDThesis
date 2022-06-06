# -*- coding: utf-8 -*-
import math
import sys
import json
import regex as re
import numpy as np

import tsfel

from pyts.approximation import PiecewiseAggregateApproximation

from scipy import signal, stats
from scipy.signal import filtfilt
from pylab import figure, plot, vlines
# from dtw import dtw
# from scipy.stats import ttest_ind

import matplotlib.pyplot as plt

from collections import Counter

from definitions import CONFIG_PATH
from string import ascii_lowercase

from novainstrumentation import smooth, lowpass
from SSTS.backend.gots_auxiliary_tools import *

import functools
from processing_tools import chunk_data


with open(
        CONFIG_PATH + r'/SSTS/backend/gots_dictionary.json') as data_file:
    gots_func_dict = json.load(data_file)


# Filtering methods
# def smooth(input_signal, window_len=10, window='hanning'):
#     """
#     @brief: Smooth the data using a window with requested size.
#     This method is based on the convolution of a scaled window with the signal.
#     The signal is prepared by introducing reflected copies of the signal
#     (with the window size) in both ends so that transient parts are minimized
#     in the beginning and end part of the output signal.
#     @param: input_signal: array-like
#                 the input signal
#             window_len: int
#                 the dimension of the smoothing window. the default is 10.
#             window: string.
#                 the type of window from 'flat', 'hanning', 'hamming',
#                 'bartlett', 'blackman'. flat window will produce a moving
#                 average smoothing. the default is 'hanning'.
#     @return: signal_filt: array-like
#                 the smoothed signal.
#     @example:
#                 time = linspace(-2,2,0.1)
#                 input_signal = sin(t)+randn(len(t))*0.1
#                 signal_filt = smooth(x)
#     @see also:  numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
#                 numpy.convolve, scipy.signal.lfilter
#     @todo: the window parameter could be the window itself if an array instead
#     of a string
#     @bug: if window_len is equal to the size of the signal the returning
#     signal is smaller.
#     """
#
#     if input_signal.ndim != 1:
#         raise ValueError("smooth only accepts 1 dimension arrays.")
#
#     if input_signal.size < window_len:
#         raise ValueError("Input vector needs to be bigger than window size.")
#
#     if window_len < 3:
#         return input_signal
#
#     if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
#         raise ValueError("""Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'""")
#
#     sig = np.r_[2 * input_signal[0] - input_signal[window_len:0:-1],
#                 input_signal,
#                 2 * input_signal[-1] - input_signal[-2:-window_len - 2:-1]]
#
#     if window == 'flat':  # moving average
#         win = np.ones(window_len, 'd')
#     else:
#         win = eval('np.' + window + '(window_len)')
#
#     sig_conv = np.convolve(win / win.sum(), sig, mode='same')
#
#     return sig_conv[window_len: -window_len]


def RemLowPass(input_signal, window_len):
    """
    @brief: for a given signal input_signal, it removes the low frequency fluctuations.
    @params:
    input_signal: signal
    window_len: window to the signal to be removed
    """
    a = input_signal - smooth(input_signal, window_len=window_len)
    return a

def PAA(s, win):
    paa = PiecewiseAggregateApproximation(window_size=win)
    X_paa = paa.transform([s])
    return X_paa[0]

def lowpass(s, f, order=2, fs=1000.0, use_filtfilt=True):
    """
    @brief: for a given signal s rejects (attenuates) the frequencies higher
    then the cuttof frequency f and passes the frequencies lower than that
    value by applying a Butterworth digital filter
    @params:
    s: array-like
    signal
    f: int
    the cutoff frequency
    order: int
    Butterworth filter order
    fs: float
    sampling frequency
    @return:
    signal: array-like
    filtered signal
    """
    b, a = signal.butter(order, f / (fs / 2))

    if use_filtfilt:
        return filtfilt(b, a, s)

    return signal.lfilter(b, a, s)


def highpass(s, f, order=2, fs=1000.0, use_filtfilt=True):
    """
    @brief: for a given signal s rejects (attenuates) the frequencies lower
    then the cuttof frequency f and passes the frequencies higher than that
    value by applying a Butterworth digital filter
    @params:
    s: array-like
    signal
    f: int
    the cutoff frequency
    order: int
    Butterworth filter order
    fs: float
    sampling frequency
    @return:
    signal: array-like
    filtered signal
    """

    b, a = signal.butter(order, f * 2 / (fs / 2), btype='highpass')
    if use_filtfilt:
        return filtfilt(b, a, s)

    return signal.lfilter(b, a, s)


def bandpass(s, f1, f2, order=2, fs=1000.0, use_filtfilt=True):
    """
    @brief: for a given signal s passes the frequencies within a certain range
    (between f1 and f2) and rejects (attenuates) the frequencies outside that
    range by applying a Butterworth digital filter
    @params:
    s: array-like
    signal
    f1: int
    the lower cutoff frequency
    f2: int
    the upper cutoff frequency
    order: int
    Butterworth filter order
    fs: float
    sampling frequency
    @return:
    signal: array-like
    filtered signal
    """
    b, a = signal.butter(order, [f1 * 2 / fs, f2 * 2 / fs], btype='bandpass')

    if use_filtfilt:
        return filtfilt(b, a, s)

    return signal.lfilter(b, a, s)

# Statistical methods
def stat_white(x):
    return (x - np.mean(x)) / np.std(x)

def normalization(signal, newMin, newMax, xmin=None, xmax=None):
    if xmin is None:
        xmin = np.min(signal)

    if xmax is None:
        xmax = np.max(signal)

    return (signal - xmin) * (newMax - newMin) / (xmax - xmin) + newMin

# Connotation Methods-------------
def AmpC(s, t, p='>'):
    thr = ((np.max(s) - np.min(s)) * t) + np.min(s)
    if (p == '<'):
        s1 = (s <= (thr)) * 1
    elif (p == '>'):
        s1 = (s >= (thr)) * 1

    return s1

def AmpQuantiles(s, nbr_quantiles):
    """

    :param s: signal in the numerical domain
    :param nbr_quantiles: number of divisions to distribute the amplitude valuues
    :return: string representation of the signal in a sequence of chars from the ascii char list
    """
    quant_levels = np.linspace(0, 1, nbr_quantiles+1)[1:-1]

    quant_s = np.quantile(s, quant_levels)
    quant_s = np.insert(quant_s, 0, np.min(s))
    quant_s = np.append(quant_s, np.max(s)+1)
    amp_f = np.empty(len(s), dtype=str)
    for i in range(1, len(quant_s)):
        amp_f[np.where(np.logical_and(s<quant_s[i], s>=quant_s[i-1]))] = list(ascii_lowercase)[i-1]

    return amp_f

def DiffD(s, t):
    """
    :param s: signal
    :param t: threshold
    :return: str
    """

    str_s = DiffC(s, 0.025)
    rle_str_s, words, counts = runLengthEncoding(list(str_s))

    max_count = max(counts)
    words = np.array(words)
    x = []

    for word, cnt in zip(words, counts):
        if cnt > t*max_count:
            x+=[word.upper()]*cnt
        else:
            x+=[word]*cnt
    return np.array(x, dtype=str)


def DiffC(s, t, signs=['n', 'z', 'p']):
    # Quantization of the derivative.
    # TODO: Implement a better way of selecting chars
    ds1 = np.diff(s)
    x = np.empty(len(s), dtype=str)
    thr = (np.max(ds1) - np.min(ds1)) * t
    x[np.where(ds1 <= -thr)[0]] = signs[0]
    x[np.where(np.all([ds1 <= thr, ds1 >= -thr], axis=0))[0]] = signs[1]
    x[np.where(thr <= ds1)[0]] = signs[2]
    x[-1] = x[-2]

    return x


def Diff2C(s, t, symbols=['C', 'l', 'D']):
    # Quantization of the derivative.
    # TODO: Implement a better threshold methodology.
    dds1 = np.diff(np.diff(s))
    x = np.empty(len(s), dtype=str)
    thr = (np.max(dds1) - np.min(dds1)) * t
    x[np.where(dds1 < -thr)[0]] = "D"
    x[np.where(np.all([dds1 <= thr, dds1 >= -thr], axis=0))[0]] = "Z"
    x[np.where(thr < dds1)[0]] = "U"
    x[-1] = x[-2]

    return x

def Concavity_segments2(s, thr2):

    ds = np.diff(s)

    ds_str, ds_list_str, constr_list = connotation([ds], "D1 " + str(thr2))

    # print("done with diff connotation")

    pos_matchs = symbolic_search(ds_str, ds_list_str, "p+")
    neg_matchs = symbolic_search(ds_str, ds_list_str, "n+")
    z_matches = symbolic_search(ds_str, ds_list_str, "z+")

    x = np.empty(len(s), dtype=str)

    for pmatch in pos_matchs:
        x[pmatch[0]:pmatch[1]] = "U"
    for nmatch in neg_matchs:
        x[nmatch[0]:nmatch[1]] = "D"
    for zmatch in z_matches:
        x[zmatch[0]:zmatch[1]] = "L"

    return x

def Concavity_segments(s, thr1, thr2):
    ds_str, ds_list_str, constr_list = connotation([s], "D1 "+str(thr2))

    # print("done with diff connotation")

    pos_matchs = symbolic_search(ds_str, ds_list_str, "p+")
    neg_matchs = symbolic_search(ds_str, ds_list_str, "n+")
    z_matches = symbolic_search(ds_str, ds_list_str, "z+")

    vals = np.zeros(len(s))
    vals_str = np.empty(len(s), dtype='<U5')

    for match in np.concatenate([pos_matchs, neg_matchs]):
        #detect overall concavity
        segment = s[match[0]:match[1]]
        slope = (segment[-1]-segment[0])/len(segment)
        x = np.linspace(0, len(segment), len(segment))
        linear_apprx = slope*x + segment[0]
        vals[match[0]:match[1]] = linear_apprx - segment


    vals_str[np.where(vals <= 0)[0]] = "U"
    vals_str[np.where(0 < vals)[0]] = "D"

    for match in z_matches:
        vals_str[match[0]:match[1]] = "Z"

    return vals_str



def RiseAmp(Signal, t):
    # detect all valleys
    val = detect_peaks(Signal, valley=True)
    # final pks array
    pks = []
    thr = ((np.max(Signal) - np.min(Signal)) * t) + np.mean(Signal)
    # array of amplitude of rising with size of signal
    risingH = np.zeros(len(Signal))
    Rise = np.array([])

    for i in range(0, len(val) - 1):
        # piece of signal between two successive valleys
        wind = Signal[val[i]:val[i + 1]]
        # find peak between two minimums
        pk = detect_peaks(wind, mph=0.1 * max(Signal))
        # print(pk)
        # if peak is found:
        if (len(pk) > 0):
            # append peak position
            pks.append(val[i] + pk)
            # calculate rising amplitude
            risingH[val[i]:val[i + 1]] = [wind[pk] - Signal[val[i]] for a in range(val[i], val[i + 1])]
            Rise = np.append(Rise, (wind[pk] - Signal[val[i]]) > thr)

    risingH = np.array(risingH > thr).astype(int)
    Rise = Rise.astype(int)

    return risingH

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

    ds_str, ds_list_str, constr_list = connotation([s], "D1 0.05")
    # print("done with diff connotation")

    pos_matchs = symbolic_search(ds_str, ds_list_str, "p+")
    neg_matchs = symbolic_search(ds_str, ds_list_str, "n+")
    z_matches = symbolic_search(ds_str, ds_list_str, "z+")

    # print("done with matches")
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

def AmpChangebySegment(s, thr):
    matches = get_dif_segments(s, thr=0.1)

    amp_array = []

    for i, match in enumerate(matches):
        segment = s[match[0]:match[1]]
        val = [abs(segment[-1] - segment[0])]
        amp_array.append(val*len(segment))


    amp_array = np.concatenate(amp_array)
    plt.plot(s)
    plt.plot(amp_array)
    plt.hlines(min(amp_array)+(thr*(max(amp_array)-min(amp_array))), 0, len(s))
    plt.show()
    amp_array = np.where(np.array(amp_array) > min(amp_array)+(thr*(max(amp_array)-min(amp_array))), "H", "L")

    return amp_array


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


def D1Speed_q(s, thr, mode="threshold"):
    """
    Design a string based on quick/slow rises/falls and High/low rises/falls on the signal.
    Falling: f/F --> f for low fall and F for high fall
    Rising: r/R --> r for low rise and R for high rise
    Q or S --> for quick or slow derivative if higher or lower trend
    :param s: signal
    :return: string
    """
    ds = get_slope_segments(s)
    str_ds = np.empty(len(ds), dtype=str)

    if(mode=="absolute"):

        slope = np.abs(ds)
        str_ds = AmplitudeTrans(slope, n=2, levels_quant=[0.5, 0.75], char_list=["_", "q", "Q"], method="quantiles")

    elif(mode=="separated"):

        pos_slope = np.where(ds > 0)[0]
        neg_slope = np.where(ds < 0)[0]

        str_ds_pos = AmplitudeTrans(ds[pos_slope], n=2, levels_quant=[0.5], char_list=["p", "P"], method="quantiles")
        str_ds_neg = AmplitudeTrans(abs(ds[neg_slope]), n=2, levels_quant=[0.5], char_list=["n", "N"],
                                    method="quantiles")

        str_ds[pos_slope] = str_ds_pos
        str_ds[neg_slope] = str_ds_neg

    elif(mode=="threshold"):
        slope = np.abs(ds)
        plt.plot(slope)
        str_ds = AmplitudeTrans(slope, n=2, thr=thr, char_list=["q", "Q"], method="custom")

    return str_ds

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
        print(thr)
        print(levels)

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

def AmpChangeAbsolute(s, thr1, thr2):
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

    ds_str, ds_list_str, constr_list = connotation([s], "D1 "+str(thr2))
    # print("done with diff connotation")

    pos_matchs = symbolic_search(ds_str, ds_list_str, "p+")
    neg_matchs = symbolic_search(ds_str, ds_list_str, "n+")
    z_matches = symbolic_search(ds_str, ds_list_str, "z+")

    # print("done with matches")
    # gt.plot_matches(s, [pos_matchs, neg_matchs], color=["green", "orange"], mode="span")

    vals = np.zeros(len(s))
    vals_str = np.empty(len(s), dtype='<U5')

    max_ampdiff_pos = max_AmpChange(s, [pos_matchs, neg_matchs])
    max_ampdiff_neg = max_ampdiff_pos

    #Get match sequence of amplitudes difference for rise and fall both numerical and textual
    for p_match in pos_matchs:
        vals[p_match[0]:p_match[1]] = abs(s[p_match[0]]-s[p_match[1]-1])
        vals_str[p_match[0]:p_match[1]] = quantilstatesArray2(vals[p_match[0]:p_match[1]], thr1 * max_ampdiff_pos, ["r", "R"], conc=False)

    for z_match in z_matches:
        vals[z_match[0]:z_match[1]] = 0
        vals_str[z_match[0]:z_match[1]] = "z"

    for n_match in neg_matchs:
        vals[n_match[0]:n_match[1]] = abs(s[n_match[0]]-s[n_match[1]-1])
        vals_str[n_match[0]:n_match[1]] = quantilstatesArray2(vals[n_match[0]:n_match[1]], thr1 * max_ampdiff_neg, ["f", "F"], conc=False)

    # print(vals_str)
    return vals_str

def AmpChangeDif(s, thr1, thr2):
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

    ds_str, ds_list_str, constr_list = connotation([s], "D1 "+str(thr2))
    # print("done with diff connotation")

    pos_matchs = symbolic_search(ds_str, ds_list_str, "p+")
    neg_matchs = symbolic_search(ds_str, ds_list_str, "n+")
    z_matches = symbolic_search(ds_str, ds_list_str, "z+")

    # print("done with matches")
    # gt.plot_matches(s, [pos_matchs, neg_matchs], color=["green", "orange"], mode="span")

    vals = np.zeros(len(s))
    vals_str = np.empty(len(s), dtype='<U5')


    #Get maximum difference values to use the thresholding
    max_ampdiff_pos = max_AmpChange(s, [pos_matchs])
    max_ampdiff_neg = max_AmpChange(s, [neg_matchs])

    #Get match sequence of amplitudes difference for rise and fall both numerical and textual
    for p_match in pos_matchs:
        vals[p_match[0]:p_match[1]] = abs(s[p_match[0]]-s[p_match[1]-1])
        vals_str[p_match[0]:p_match[1]] = quantilstatesArray2(vals[p_match[0]:p_match[1]], thr1 * max_ampdiff_pos, ["r", "R"], conc=False)

    for z_match in z_matches:
        vals[z_match[0]:z_match[1]] = 0
        vals_str[z_match[0]:z_match[1]] = "z"

    for n_match in neg_matchs:
        vals[n_match[0]:n_match[1]] = abs(s[n_match[0]]-s[n_match[1]-1])
        vals_str[n_match[0]:n_match[1]] = quantilstatesArray2(vals[n_match[0]:n_match[1]], thr1 * max_ampdiff_neg, ["f", "F"], conc=False)


    return vals_str

def Speed(s, thr):
    # sep = np.mean(abs(np.diff(s)))
    # sep = np.median(abs(np.diff(s)))
    pos_dif = np.where(np.diff(s)>0)
    neg_dif = np.where(np.diff(s)<=0)
    if(len(pos_dif[0])>0):
        sep_pos = np.quantile(np.diff(s)[pos_dif[0]], thr)
    else:
        sep_pos = 0.66*np.abs(np.diff(s))
    if(len(neg_dif[0])>0):
        sep_neg = np.quantile(abs(np.diff(s)[np.where(np.diff(s)<=0)[0]]), thr)
    else:
        sep_neg = 0.66 * np.abs(np.diff(s))

    amp_f = np.empty(len(s), dtype='<U5')
    amp_f[np.where(np.logical_and(np.diff(s)>0, np.diff(s)>sep_pos))[0]] = "R"
    amp_f[np.where(np.logical_and(np.diff(s)>0, np.diff(s)<=sep_pos))[0]] = "r"
    amp_f[np.where(np.logical_and(np.diff(s) <= 0, abs(np.diff(s)) > sep_neg))[0]] = "F"
    amp_f[np.where(np.logical_and(np.diff(s) <= 0, abs(np.diff(s)) <= sep_neg))[0]] = "f"

    return amp_f

def FeatAUC(s, win_size):
    cfg = tsfel.get_features_by_domain("temporal")
    cfg["temporal"] = {"Area under the curve":cfg["temporal"]["Area under the curve"]}
    X = tsfel.time_series_features_extractor(cfg, s, window_size=win_size, overlap=0.9, verbose=0)

    auc_s = AmpQuantiles(X[X.keys()[0]].to_numpy(), 3)

    return auc_s

def FeatSlope(s, win_size):
    s_windows = chunk_data(s, win_size, overlap_size=math.ceil(0.9*win_size))
    vals = []
    for s_wi in s_windows:
        t = np.linspace(0, len(s_wi) - 1, len(s_wi))
        f_t = np.polyfit(t, s_wi, 1)[0]
        vals.append(f_t)

    feat_s = AmpQuantiles(np.abs(vals), 3)

    return feat_s

def Diff2Linearity(s, win_size):
    s_windows = chunk_data(s, win_size, overlap_size=math.ceil(0.9*win_size))
    vals = []
    for s_wi in s_windows:
        t = np.linspace(0, len(s_wi) - 1, len(s_wi))
        f_t = np.polyfit(t, s_wi, 1)
        p_fit = np.poly1d(f_t)(t)
        vals.append(np.sum(np.abs(s_wi-p_fit)))

    feat_s = AmpQuantiles(vals, 3)

    return feat_s

def FeatSAD(s, win_size):
    feat = "Sum absolute diff"
    cfg = tsfel.get_features_by_domain("temporal")
    cfg["temporal"] = {feat:cfg["temporal"][feat]}
    X = tsfel.time_series_features_extractor(cfg, s, window_size=win_size, overlap=0.9, verbose=0)

    auc_s = AmpQuantiles(X[X.keys()[0]].to_numpy(), 3)

    return auc_s

def FeatSkew(s, win_size):
    feat = "Skewness"
    cfg = tsfel.get_features_by_domain("statistical")
    cfg["statistical"] = {feat:cfg["statistical"][feat]}
    X = tsfel.time_series_features_extractor(cfg, s, window_size=win_size, overlap=0.9, verbose=0)
    feat_s = AmpQuantiles(X[X.keys()[0]].to_numpy(), 3)

    return feat_s

def FeatVar(s, win_size):
    feat = "Variance"
    cfg = tsfel.get_features_by_domain("statistical")
    cfg["statistical"] = {feat:cfg["statistical"][feat]}
    X = tsfel.time_series_features_extractor(cfg, s, window_size=win_size, overlap=0.9, verbose=0)
    feat_s = AmpQuantiles(X[X.keys()[0]].to_numpy(), 3)

    return feat_s

def FeatSTD(s, win_size):
    feat = "Standard deviation"
    cfg = tsfel.get_features_by_domain("statistical")
    cfg["statistical"] = {feat:cfg["statistical"][feat]}
    X = tsfel.time_series_features_extractor(cfg, s, window_size=win_size, overlap=0.9, verbose=0)
    feat_s = AmpQuantiles(X[X.keys()[0]].to_numpy(), 3)

    return feat_s
#
# def FeatETP(s, win_size):
#     feat = "Entropy"
#     cfg = tsfel.get_features_by_domain("temporal")
#     cfg["temporal"] = {feat: cfg["temporal"][feat]}
#     X = tsfel.time_series_features_extractor(cfg, s, window_size=win_size, overlap=0.9, verbose=0)
#     feat_s = AmpQuantiles(X[X.keys()[0]].to_numpy(), 4)
#
#     return feat_s

def SlopeAbsolute(s, thr):
    ds_str, ds_list_str, constr_list = connotation([s], "D1 0.001")
    # print("done with diff connotation")

    pos_matchs = symbolic_search(ds_str, ds_list_str, "p+")
    neg_matchs = symbolic_search(ds_str, ds_list_str, "n+")
    z_matches = symbolic_search(ds_str, ds_list_str, "z+")

    # print("done with matches")

    vals = np.zeros(len(s))
    vals_str = np.empty(len(s), dtype='<U5')

    #calculate slopes
    for match in np.concatenate([pos_matchs, neg_matchs]):
        segment = s[match[0]:match[1]]
        slope_i = (segment[-1] - segment[0]) / len(segment)
        vals[match[0]:match[1]] = [slope_i] * len(segment)

    max_slope_pos = max(vals)

    for match in np.concatenate([pos_matchs, neg_matchs]):
        vals_str[match[0]:match[1]] = quantilstatesArray2(vals[match[0]:match[1]], thr * max_slope_pos,
                                                              ["q", "Q"], conc=False)
    for z_match in z_matches:
        vals_str[z_match[0]:z_match[1]] = "z"

    return vals_str

def SlopeDif(s, thr):
    ds_str, ds_list_str, constr_list = connotation([s], "D1 0.001")
    # print("done with diff connotation")

    pos_matchs = symbolic_search(ds_str, ds_list_str, "p+")
    neg_matchs = symbolic_search(ds_str, ds_list_str, "n+")
    z_matches = symbolic_search(ds_str, ds_list_str, "z+")

    # print("done with matches")

    vals = np.zeros(len(s))
    vals_pos = []
    vals_neg = []
    vals_str = np.empty(len(s), dtype='<U5')

    #calculate slopes
    for p_match in pos_matchs:
        segment = s[p_match[0]:p_match[1]]
        slope_i = (segment[-1] - segment[0]) / len(segment)
        vals[p_match[0]:p_match[1]] = [slope_i] * len(segment)
        vals_pos.append(slope_i)

    for n_match in neg_matchs:
        segment = s[n_match[0]:n_match[1]]
        slope_i = (segment[-1] - segment[0]) / len(segment)
        vals[n_match[0]:n_match[1]] = [slope_i] * len(segment)
        vals_neg.append(slope_i)

    max_slope_pos = max(vals_pos)
    max_slope_neg = max(vals_neg)

    # Get match sequence of amplitudes difference for rise and fall both numerical and textual
    for p_match in pos_matchs:
        vals_str[p_match[0]:p_match[1]] = quantilstatesArray2(vals[p_match[0]:p_match[1]], thr * max_slope_pos,
                                                              ["d", "D"], conc=False)
    for z_match in z_matches:
        vals_str[z_match[0]:z_match[1]] = "z"

    for n_match in neg_matchs:
        vals_str[n_match[0]:n_match[1]] = quantilstatesArray2(vals[n_match[0]:n_match[1]], thr * max_slope_neg,
                                                              ["s", "S"], conc=False)
    return vals_str

def Concavity(s, thr):
    ds = np.diff(s, 1)
    dds = np.diff(ds, 1)

    x = np.empty(len(s), dtype=str)
    thr = (np.max(dds) - np.min(dds)) * thr
    x[np.where(dds <= -thr)[0]] = "D"
    x[np.where(np.all([dds <= thr, dds >= -thr], axis=0))[0]] = "L"
    x[np.where(thr <= dds)[0]] = "C"
    x[-1] = x[-2]

    return x

def D1_concavity(s):
    dds = get_concavity_per_segment(s)

    str_ds = np.where(dds==1, "Convex", "Concave")

    return str_ds

def D1_linearity(s, thr):
    dds = get_difference_between_linear_segment(s)

    str_ds = AmplitudeTrans(dds, n=2, char_list=["L", "H"], thr=thr, method="custom")

    return str_ds


# Auxiliary methods
def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb"""

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02 * x.size, x.size * 1.02 - 1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))


def merge_chars2list(string_matrix):
    """
        Function performs the merge of the strings generated with each method. The function assumes
        that each string is organized in the StringMatrix argument as a column.
        The function returns the merged string.
    """
    Str = []
    for i in range(len(string_matrix[0])):
        Str.append("".join(str(item[i]) for item in string_matrix))

    return Str


def merge_chars(string_matrix):
    """
    Function performs the merge of the strings generated with each method. The function assumes
    that each string is organized in the StringMatrix argument as a column.
    The function returns the merged string.
    """
    col = np.size(string_matrix, axis=0)
    lines = np.size(string_matrix, axis=1)
    # print(string_matrix)
    Str = ""
    for l in range(0, lines):
        for c in range(0, col):
            Str += str(string_matrix[c][l])

    return Str


def vmagnitude(v):
    """
    Returns the magnitude of a tridimensional vector signal.
    :param v: (ndarray-like)

    :return: The magnitude of the signal.
    """
    return np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2)


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def prep_str(cfgstr, ls):
    """This function prepares the """
    if r' | ' in cfgstr:
        pstr = cfgstr.split(r' | ')
    else:
        pstr = [cfgstr] * ls
    return pstr


def plot_matches(s, m, mode="scatter", ax=plt.subplot(111)):
    """

    :param s:
    :param m: list of list of matches (list(match)). For different match patterns, include as list of list of matches: list(match1, match2, match3,...)
    :param color:
    :param mode: Mode of plotting. If:
                                    - only give the position of the match at the beginning and end (mode==scatter/vline)
                                    - only give span of where the match was occurring (mode==span)
    :return: plot object
    """

    # figure(figsize=(16, 5))
    ax.plot(s, alpha=0.6, color="dodgerblue", linewidth=3)


    if (mode=="scatter"):
        [ax.plot(i[0], s[i[0]], 'o', color="blue") for i in m]
    elif(mode=="span"):
        [ax.axvspan(m_i[0], m_i[1]-1, alpha=0.1, color="dodgerblue") for m_i in m]
    elif (mode == "vline"):
        [ax.vlines(i[0], np.min(s), np.max(s), lw=3) for i in m]

def pre_processing(s, processing_methods):
    s = np.asarray(s)
    if s.ndim == 1:
        s = np.array([s])
    ns2 = []
    pp_str = prep_str(processing_methods, len(s))

    operands = ""

    for i in range(len(s)):
        pp_func_stack = pp_str[i].split(" ")
        ns_temp = [s[0]]
        # ns2.append(s[i])
        for j, val in enumerate(pp_func_stack):
            if not isfloat(val):
                if val is "":
                    # print("Space")
                    continue
                elif val not in list(gots_func_dict["pre_processing"].keys()):
                    sys.exit('Unknown pre-processing symbol.')
                else:
                    operator = val
                    # print(gots_func_dict["pre_processing"][operator])
                    for subval in pp_func_stack[j + 1:]:
                        if not isfloat(subval):
                            break
                        else:
                            operands += subval + ','

                    if operands is "":
                        ns_temp.append(eval(gots_func_dict["pre_processing"][operator] + '(ns_temp[' + str(i) + '])'))

                    else:
                        ns_temp.append(eval(
                            gots_func_dict["pre_processing"][operator] + '(ns_temp[' + str(i) + '],' + operands[:-1] + ')'))
                    operands = ""


            else:
                continue
        if(len(s)==1):
            return ns_temp[-1]
        else:
            return ns2.append(ns_temp[-1])

def connotation(s, connotation):
    sc_str = prep_str(connotation, len(s))
    operands = ""
    merged_sc_str = []
    for i in range(len(s)):
        sc_func_stack = sc_str[i].split(" ")
        for j, val in enumerate(sc_func_stack):
            if not isfloat(val):
                if val is "":
                    sys.exit('At least a connotation method must be supplied.')
                elif val[0] == "[" and val[-1] == "]":
                    continue
                elif val not in list(gots_func_dict["connotation"].keys()):
                    sys.exit('Unknown connotation symbol.')
                else:
                    operator = val
                    for subval in sc_func_stack[j + 1:]:
                        if not isfloat(subval) and subval[0] != "[" and subval[-1] != "]":
                            break
                        else:
                            operands += subval + ','

                    if operands is "":
                        _constr = eval(gots_func_dict["connotation"][operator] + '(s[' + str(i) + '])')

                    else:
                        _constr = eval(
                            gots_func_dict["connotation"][operator] + '(s[' + str(i) + '],' + operands[:-1] + ')')

                    operands = ""
                    merged_sc_str += [_constr]
            else:
                continue
    # print(merged_sc_str)
    constr= merge_chars(merged_sc_str)
    constr_list = merge_chars2list(merged_sc_str)

    return constr, merged_sc_str, constr_list


def regit_map(reg, size_mg):
    return (int(reg.span()[0] / size_mg),
                         int(reg.span()[1] / size_mg))

def optimized_regit_map(reg, args):
    counts, size_mg = args
    count_0 = np.sum(counts[:int(reg.span()[0]/size_mg)])
    count_1 = np.sum(counts[:int(reg.span()[1]/size_mg)])


    return (count_0, count_1)

def optimized_symbolic_search(constr, merged_sc_str, search):
    #perform RLE to connotated string
    counts, rle_sig = optimized_runLengthEncoding(constr)
    regit = re.finditer(search, rle_sig)
    matches = list(map(functools.partial(optimized_regit_map, args=(counts, np.shape(merged_sc_str)[0])), regit))

    return matches

def symbolic_search(constr, merged_sc_str, search):
    # matches = []

    regit = re.finditer(search, constr)
    matches = list(map(functools.partial(regit_map, size_mg=np.shape(merged_sc_str)[0]), regit))

    # print(matches)
    # it = np.nditer([regit], flags=["refs_OK"], op_flags=["readwrite"])
    # print(it)
    # with it:
    #     [matches.append((int(i.span()[0] / np.shape(merged_sc_str)[0]),
    #                      int(i.span()[1] / np.shape(merged_sc_str)[0]))) for i in it]
    # print(list(regit))
    # print("Done with finditer")
    # [matches.append((int(i.span()[0] / np.shape(merged_sc_str)[0]),
    #                  int(i.span()[1] / np.shape(merged_sc_str)[0]))) for i in regit]

    return matches

def recursive_symbolic_search(sig_str, nbr_connotation, pattern, n=0, match0 = 0):
    """
    TODO: ability to search for more than one pattern for each layer by adding a simple comma
    Performs symbolic search on time series in a recursive way, based on the
    tree given by the user. The string is parsed based on an arrow with the indication of the parent pattern:
    pattern 0 -0> pattern 1 -1> pattern 1.1 -0> pattern 2 -1> pattern 2.1
    :return:
    """
    #create main storage of matches in dict to divide into layers
    recursive_matches = {}

    #split the pattern string based on the arrows and levels indicated by the arrows
    p = re.split(r" -"+str(n)+"> ", pattern)

    #search for the matches of the first element (parent node)
    regit = re.finditer(p[0], sig_str)
    #get matches as a list
    matches = list(map(functools.partial(regit_map, size_mg=nbr_connotation), regit))
    #real matches are the previous matches summed with the first index of the previous match, so that we can
    #keep up with the original x_position of the matches
    real_matches = [(match[0]+match0, match[1]+match0) for match in matches]

    #increment layer of search
    n_i = n+1

    #if the size of the patterns splitted by the arrows is higher than one, it means there is the need to
    #loop inside the string
    if len(p)>1:
        #add the current layer to the dict container
        recursive_matches[n] = real_matches
        recursive_matches[n_i] = []
        #loop to search an inner match for each current match
        for match in matches:
            #loop to search for inner patterns
            for p_i in p[1:]:
                #recursive search when inner patterns are present
                finished_recursive_matches = recursive_symbolic_search(sig_str[match[0]:match[1]], nbr_connotation, p_i, n_i, match[0]+match0)
                #you have to add all the way through the recursivity from the current layer to the last inner layer found
                #if you add just the next one, you will loose deeper layers
                for n_key in finished_recursive_matches.keys():
                    #if the dict container has not yet this layer activated..initialize an array to store it
                    if(n_key not in recursive_matches.keys()):
                        recursive_matches[n_key] = []
                    #increase your matches layer
                    recursive_matches[n_key] += finished_recursive_matches[n_key]
        return recursive_matches
    #if there are no inner layers and no need for recursive steps
    #just store the matches in the container
    else:
        recursive_matches[n] = real_matches
        return recursive_matches


# Main methods
def ssts(s, cfg, report='clean'):
    """
    Performs a query on a given time series based upon on a syntactic approach.
    :param s: array-like
        The input time series.
    :param cfg: dictionary
        A configuration dictionary structure that defines the query parameters.
    :param report: string
        A string containing the report type of returned values. Available types
        include: ``clean``, which returns the positive matches only and ``full``
        which returns the pre-processed signal, the connotation string and the
        positive matches.
    :return:
        ns: (array-like)
        The signal segment that corresponds to the query result.
    """
    # Handles exception to multisignal approach.

    ns = np.copy(s)

    # Layer 1: Pre-processing
    # print(cfg["pre_processing"])
    ns = pre_processing(ns, cfg["pre_processing"])

    #Layer 2: Connotation
    constr, merged_sc_str, constr_list = connotation([ns], cfg["connotation"])

    #Layer 3: Search
    matches = symbolic_search(constr, merged_sc_str, cfg["expression"])

    # removes unnecessary nesting in case ndim is 1.
    if ns.shape[0] == 1:
        ns = ns[0]

    if report is 'clean':
        return matches
    elif report is 'full':
        return ns, constr, matches

def CharFreqTest(s_string1, s_string2, n_connotations, win):
    freq = {}
    if (n_connotations == "derivative"):
        s_string_m1 = s_string1[:win // 2][::-1] + s_string1 + s_string1[-win // 2:][::-1]
        s_string_m2 = s_string2[:win // 2][::-1] + s_string2 + s_string2[-win // 2:][::-1]

        for char in set(s_string_m1):
            freq[char] = []

        for i in range(0, len(s_string_m1)-win):
            string_i1 = s_string_m1[i:i+win]
            string_i2 = s_string_m2[i:i+win]

            freq_i, names_i = string_corr4char_count(string_i1, string_i2, ["p", "z", "n"])

            for ii in freq_i:
                freq[ii].append(freq_i[ii])
        return freq


def CharFreq(s_string, n_connotations, win):
    freq = {}
    time = {}

    if (n_connotations == 1):
        #mirror string
        s_string_m = s_string[:win//2][::-1] + s_string + s_string[-win//2:][::-1]
        for char in set(s_string_m):
            freq[char] = []
            time[char] = []
        for i in range(0, len(s_string_m)-win):
            string_i = s_string_m[i:i+win]
            cnt_data = Counter(string_i)
            for ii in set(s_string):
                if (ii not in list(cnt_data)):
                    freq[ii].append(0)
                else:
                    freq[ii].append(cnt_data[ii] / len(string_i))
        return freq

def string_corr4char_count(s1, s2, seq):
    """
    Gives a distribution of the different characters of string 1 that appear related with a character in string 2
    :param s1: string 1
    :param s2: string 2
    :return: distribution of frequency of the set of characters for both strings
    """
    str_list1 = list(s1)
    str_list2 = list(s2)
    freq = {}
    for char_1 in seq:
        freq[char_1] = []

        for char_2 in seq:
            freq_i = [1 for a_i, b_i in zip(str_list1, str_list2) if(a_i==char_1 and b_i==char_2)]
            freq[char_1].append(len(freq_i))


    return freq, list(set(s2))

def string_corr(s1, s2, ):
    """
    Gives a distribution of the different characters of string 1 that appear related with a character in string 2
    :param s1: string 1
    :param s2: string 2
    :return: distribution of frequency of the set of characters for both strings
    """
    str_list1 = list(s1)
    str_list2 = list(s2)
    freq = {}
    for char_1 in set(s1):
        freq[char_1] = []

        for char_2 in list(set(s2)):
            freq_i = [1 for a_i, b_i in zip(str_list1, str_list2) if(a_i==char_1 and b_i==char_2)]
            freq[char_1].append(len(freq_i))


    return freq, list(set(s2))


def string_corr_trans(s1, s2):
    """
    Gives a distribution of the different characters of string 1 that appear related with a character in string 2
    :param s1: string 1
    :param s2: string 2
    :return: distribution of frequency of the set of characters for both strings
    """
    str_list1 = list(s1)
    str_list2 = [s2[s2_i:s2_i+2] for s2_i in range(0, len(s2)-1)]
    set_list2 = [i for i in list(set(str_list2)) if(i[0]!=i[1])]

    freq = {}
    for char_1 in set(s1):
        freq[char_1] = []
        for char_2 in set_list2:
            freq_i = [1 for a_i, b_i in zip(str_list1, str_list2) if(a_i==char_1 and b_i==char_2)]
            freq[char_1].append(len(freq_i))


    return freq, set_list2

def Output1(s_string, matches, n_connotations, temp_array="False"):
    freq = {}
    time = {}
    if (n_connotations == 1):
        for char in set(s_string):
            freq[char] = []
            time[char] = []
        for i in matches:
            a = s_string[i[0]:i[1]]
            cnt_data = Counter(a)

            for ii in set(s_string):

                if (ii not in list(cnt_data)):
                    freq[ii].append(0)
                else:
                    freq[ii].append(cnt_data[ii] / len(a))

                if (temp_array is "False"):
                    time[ii].append((i[1] - i[0]) / 2)
                else:
                    time[ii].append((temp_array[i[0]] + (temp_array[i[1]] - temp_array[i[0]]) / 2))

    return time, freq

def levenshteinDist(s1, s2):
    d = np.zeros((len(s1), len(s2)))

    for i in range(0, len(s1)):
        d[i, 0] = i
    for j in range(0, len(s2)):
        d[0, j] = j

    for j in range(0, len(s2)):
        for i in range(0, len(s1)):
            if (s1[i] == s2[j]):
                cost = 0
            else:
                cost = 1
            d[i, j] = min([d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost])

    return d[-1, -1]


def string_matrix(string, matches, method="levenshtein"):
    d = np.zeros((len(matches), len(matches)))

    if (method == "levenshtein"):
        for i, match1 in enumerate(matches):
            s1 = string[match1[0]:match1[1]]
            for j, match2 in enumerate(matches):
                s2 = string[match2[0]:match2[1]]
                d[i, j] = levenshteinDist(s1, s2)

    return d

def convolve1d(s1, s2):
    l_s1 = np.size(s1)
    l_s2 = np.size(s2)

    C = np.zeros(l_s1 + l_s2 - 1)

    for m in range(l_s1):
        for n in range(l_s2):
            C[m + n] = C[m + n] + (s1[m] * s2[n])
    return C / (l_s1 + l_s2 - 1)


def cross_corr(s1, s2):
    C = convolve1d(np.conj(s1), s2)

    return C


def kind_of_Similarity(a, b):
    if (len(a) > len(b)):
        major = a
        minor = b
    elif (len(b) > len(a)):
        major = b
        minor = a
    else:
        major = a
        minor = b
    major = major[::-1]
    ab_sub = []
    for i in range(len(b) + len(a)):
        win = np.zeros(len(minor))
        if (i <= len(minor)):
            win[0:i] = major[0:i][::-1]
        elif (i > len(minor) and i <= len(major)):
            win = major[i - len(minor):i][::-1]
        else:
            win[i - len(major):] = major[i - len(minor):][::-1]

        ab_sub.append(np.sum(abs(np.subtract(minor, win))))

    return sum(ab_sub) / (len(major) + len(minor))


# def numerical_matrix(data_array, matches, temp_array, method):
#     d = np.zeros((len(matches), len(matches)))
#
#     if (method == "convolve"):
#         for i, match1 in enumerate(matches):
#             s1 = data_array[match1[0]:match1[1]]
#             for j, match2 in enumerate(matches):
#                 s2 = data_array[match2[0]:match2[1]]
#                 # plt.figure()
#                 # ax1 = plt.subplot(3,1,1)
#                 # ax2 = plt.subplot(3,1,2)
#                 # ax3 = plt.subplot(3,1,3)
#                 s1_1 = s1 - np.mean(s1)
#                 s2_2 = s2 - np.mean(s2)
#                 # ax1.plot(s1_1)
#                 # ax2.plot(s2_2)
#                 # ax3.plot(cross_corr(s1_1, s2_2))
#                 # plt.show()
#                 # x = cross_corr(s1_1, s2_2)
#                 # stat, pvalue = ttest_ind(s1_1, s2_2)
#                 val = kind_of_Similarity(s1_1, s2_2)
#                 print(val)
#                 d[i, j] = val
#         return d
#
#     elif (method == "slope"):
#         slope_m = []
#         slope_t = []
#         for i, match1 in enumerate(matches):
#             slice_data = data_array[match1[0]:match1[1]]
#
#             if (len(slice_data) == 1):
#                 slope_m.append(0)
#             else:
#                 slope_m.append(np.mean(np.diff(slice_data)))
#             slope_t.append((temp_array[match1[0]] + (temp_array[match1[1]] - temp_array[match1[0]]) / 2))
#         return slope_t, slope_m
#
#     elif( method =="dtw"):
#         for i, match1 in enumerate(matches):
#             s1 = data_array[match1[0]:match1[1]]
#             s1_1 = s1 - np.mean(s1)
#             for j, match2 in enumerate(matches):
#                 s2 = data_array[match2[0]:match2[1]]
#                 s2_2 = s2 - np.mean(s2)
#                 val = dtw_distance(s1_1, s2_2)
#
#                 d[i, j] = val
#         return d
#
#     elif (method == "time_width"):
#         width = []
#         width_t = []
#         for i, match1 in enumerate(matches):
#             width.append(to_integer(temp_array[match1[1]] - temp_array[match1[0]]))
#             width_t.append((temp_array[match1[0]] + (temp_array[match1[1]] - temp_array[match1[0]]) / 2))
#         return width_t, width


def to_integer(dt_time):
    return dt_time.days

# def dtw_distance(a,b):
#     euclidean_norm = lambda a, b: np.abs(a - b)
#
#     d, cost_matrix, acc_cost_matrix, path = dtw(a, b, dist=euclidean_norm)
#
#     print(d)
#
#     return d

def probability_Char(data):
    keys = list(data.keys())
    total_freq = 0
    for key in keys:
        total_freq += data[key]
    pb = {}
    for i in range(0, len(data[keys[0]])):
        for key in keys:
            pb[key] = data[key][i]/data


def optimized_runLengthEncoding(list_str):
    count = 1
    previous = ""
    counts = []
    words = []

    for word in list_str:
        if word != previous:
            if previous:
                counts.append(count)
                words.append(previous)
            count = 1
            previous = word
        else:
            count += 1
    else:
        counts.append(count)
        words.append(previous)

    words = "".join(word for word in words)

    return counts, words

def runLengthEncoding(list_str):
    # Generate ordered dictionary of all lower
    # case alphabets, its output will be
    # dict = {'w':0, 'a':0, 'd':0, 'e':0, 'x':0}

    count = 1
    previous = ""
    counts = []
    words = []

    for word in list_str:
        if word != previous:
            if previous:
                counts.append(count)
                words.append(previous)
            count = 1
            previous = word
        else:
            count += 1
    else:
        counts.append(count)
        words.append(previous)

    result_join = " ".join(f"{str(count)}{word}" for word, count in zip(words, counts))

    return result_join, words, counts



"""
More Connotations:
-> segment amplitude change if higher than previous (H) if lower than previous (L) - based on a threshold

"""