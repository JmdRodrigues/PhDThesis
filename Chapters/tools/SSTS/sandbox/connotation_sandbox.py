import numpy as np
import matplotlib.pyplot as plt
from tools.string_processing_tools import runLengthEncoding, lstStr2Str, Ngrams, NgramsInt, BagofWords, LegnthConnotation, tf_idf

from GrammarofTime.SSTS.backend import gotstools as gt
from GrammarofTime.SSTS.sandbox.connotation_tools import *

from scipy.stats import binom


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

def checkDistribution(s, n):
    hist, bins = np.histogram(s, bins=n)
    # print(quantilstatesArray(s, bins[1:-1], string.ascii_uppercase, False))


def removeShortSegments(str_signal, counts):

    str_clean_output = []

    posUpper = [i for i in range(len(str_signal)) if "L" in str_signal[i]]

    for i, count in enumerate(counts):

        if("l" in str_signal[i]):
            if("Q" in str_signal[i]):
                val_temp = "L"+str_signal[i][1:]
            else:
                val_temp = str_signal[posUpper[np.argmin(abs(np.array(posUpper) - i))]]
            for ii in range(count):
                str_clean_output.append(val_temp)
        else:
            for ii in range(count):
                str_clean_output.append(str_signal[i])

    return str_clean_output

def AmplitudeTrans(s, n, char_list, thr = 0.0, method="distribution"):
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
        levels = np.quantile(s, [0.25, 0.5, 0.75])

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

def max_AmpChange(s, m):
    """

    :param s: signal
    :param m: list of list of matches
    :return: value of maximum amplitude difference
    """

    max_ampdif = 0

    if len(m)>1:
        for matches in m:
            for match in matches:
                if(max_ampdif<abs(s[match[0]] - s[match[1] - 1])):
                     max_ampdif = abs(s[match[0]] - s[match[1] - 1])

    return max_ampdif


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

    ds_str, ds_list_str, constr_list = gt.connotation([s], "D1 0.01")
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
    Design a string based on quick/slow rises/falls and High/low rises/falls on the signal.
    Falling: f/F --> f for low fall and F for high fall
    Rising: r/R --> r for low rise and R for high rise
    Q or S --> for quick or slow derivative if higher or lower trend
    :param s: signal
    :return: string
    """

    ds = np.diff(s)
    ds = np.append(ds, ds[-1])
    x = AmplitudeTrans(abs(ds), 2, ["q", "Q"], thr = thr, method="custom")

    return x

def D1Speed_q(s, thr, mode="absolute"):
    """
    Design a string based on quick/slow rises/falls and High/low rises/falls on the signal.
    Falling: f/F --> f for low fall and F for high fall
    Rising: r/R --> r for low rise and R for high rise
    Q or S --> for quick or slow derivative if higher or lower trend
    :param s: signal
    :return: string
    """
    ds = get_slope_segments(s)
    if(mode=="absolute"):
        ds_str = AmplitudeTrans(abs(ds), 1, ["p", "P"])
    elif(mode=="separate"):
        a = 1


    return ds_str

def D2Trans(s, signs=["mM", "Mm"]):
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
    # ds2 = np.diff(ds1)
    x = np.chararray(len(s), itemsize=5)
    # thr = (np.max(ds2) - np.min(ds2)) * t

    x[np.where(ds1 <= 0)[0]] = str(signs[0])
    x[np.where(ds1 > 0)[0]] = str(signs[1])
    x[-2] = x[-3]
    x[-1] = x[-2]

    return x

def LogLikelihoodNgramsMultiple(Bow_unique, Bow_Ngrams, Bow_ngrams, N):
    """
    Log Likelihood of having the last word of the N-gram considering the previous sequence,
    e.g., for the sequence [W1, W2, W3, W4], what is the probability of having W4 when there
    is a sequence of W1, W2 and W3.
    :param Bow_unique: Individual word frequency
    :param Bow_Ngrams: N-gram word frequency
    :param Bow_ngrams: N-gram - 1 word frequency
    :param N: total number of N-grams
    :return: probabilities
    """
    Prob = {}
    for N_gram in Bow_Ngrams:
        individual_words = N_gram.split(" ")
        count_w1 = Bow_ngrams[" ".join(word for word in individual_words[:-1])]
        count_w2 = Bow_unique[individual_words[-1]]
        count_w12 = Bow_Ngrams[N_gram]

        p = count_w2 / N
        p1 = count_w12 / count_w1
        p2 = (count_w2 - count_w12) / (N - count_w1)

        # Calculate individual binomial probabilities
        pbinom1 = binom.cdf(count_w12, count_w1, p)
        pbinom2 = binom.cdf(count_w2 - count_w12, N - count_w1, p)
        pbinom3 = binom.cdf(count_w12, count_w1, p1)
        pbinom4 = binom.cdf(count_w2 - count_w12, N - count_w1, p)

        # Log likelihood
        LL_i = np.log((pbinom1 * pbinom2) / (pbinom3 * pbinom4))
        Prob[N_gram] = {"LL": LL_i, "p1": p1, "p2": p2, "p": p}

    return Prob


def logLikelihoodNgrams(BOW_unique, BOW_grams, N):
    """
    Calculates the log likelihood probabilities for each ngrams.
    :param BOW_unique: dictionnary with each word and the corresponding counts
    :param BOW_grams: dictionnary with each n-gram and the corresponding counts
    :param N: value of the total number of n-grams
    :return: dictionnary with probabilities for each n-gram
    """

    Prob = {}

    for n_gram in BOW_grams:
        count_w1 = BOW_unique[n_gram.split(" ")[0]]
        print(count_w1)
        count_w2 = BOW_unique[n_gram.split(" ")[1]]
        count_w12 = BOW_grams[n_gram]

        p = count_w2/N
        p1 = count_w12/count_w1
        p2 = (count_w2 - count_w12)/(N - count_w1)

        #Calculate individual binomial probabilities
        pbinom1 = binom.cdf(count_w12, count_w1, p)
        pbinom2 = binom.cdf(count_w2 - count_w12, N-count_w1, p)
        pbinom3 = binom.cdf(count_w12, count_w1, p1)
        pbinom4 = binom.cdf(count_w2 - count_w12, N - count_w1, p)

        # Log likelihood
        LL_i = np.log((pbinom1*pbinom2)/(pbinom3*pbinom4))
        Prob[n_gram] = {"LL":LL_i, "p1":p1, "p2":p2, "p":p}

    return Prob

def addArrayofStrings(array_strings):
    return ["".join(list(group_str)) for group_str in np.array(array_strings).T]

def SignConnotation(s, char_list=["U","D"]):
    """
    Translates the signal into positive and negative parts of the signal. If the signal is positive (char 1) if the signal is
    negative (char 2).
    :param s:signal
    :param char_list:list of chars for the sign connotation. Only lengths of 2 will be accepted
    :return: the list of str with the char values for each sample of the signal
    """
    thr = np.mean(s)
    s_str = [char_list[0] if s_i>=thr else char_list[1] for s_i in s]

    return s_str




# def plot_ProbNgramMultiple(signal_clean_str_unique, n):
#
#     #bag of words for each word
#     bow_unique = BagofWords(signal_clean_str_unique[1])
#
#     #Ngrams list with N words
#     ch1_ngramsN = Ngrams(signal_clean_str_unique[1], n)
#
#     #Ngrams list with N-1 words
#     ch1_ngramsn = Ngrams(signal_clean_str_unique[1], n-1)
#
#     #bag of words for the list of N words
#     bow_ngramsN = BagofWords(ch1_ngramsN[1])
#
#     #bag of words for the list of N-1words
#     bow_ngramsn = BagofWords(ch1_ngramsn[1])
#
#     #Probabilities calculation
#     Probs_ch1_nN = LogLikelihoodNgramsMultiple(bow_unique, bow_ngramsN, bow_ngramsn, len(ch1_ngramsN[0]))
#
#
#     ch1_ngram_total = Ngrams(signal_clean_str_unique[1], n)
#     count_ngram = NgramsInt(np.cumsum(signal_clean_str_unique[2]), n)
#
#     for ngram in Probs_ch1_nN:
#         print("Ngram:")
#         print(ngram)
#         # regit = re.finditer(ngram, ch1_clean)
#         matches = np.where(np.array(ch1_ngram_total[1]) == ngram)[0]
#         print("Matches")
#         print(matches)
#         for match in matches:
#             plt.axvspan(count_ngram[match][0], count_ngram[match][-1], 0, 1, alpha=0.25)
#             plt.plot(ch1)
#             plt.show()
