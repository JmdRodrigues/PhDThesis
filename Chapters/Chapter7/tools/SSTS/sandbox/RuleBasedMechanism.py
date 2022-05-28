"""
In this code section, I am defining a set of rule based mechanisms
for syntactic search on time series.

The purpose is to define a dictionnary in which a structure for the
description of time series is arranged. Just as in SpaCy it is possible
to search on text based on tags and info about the structure in the sentence,
it should also be possible to make a search based on a characterization of
the signal with a set of characters.

I can add mechanisms of search based on key tags from the sequence designed.

1) Design standard mechanism for time series translation based on amplitude, first and second derivative.
2) Attribute special tags into a board structure.
    Possible Tag titles: "
                          A) Maximum or Minimum Peak
                          B) Positive or Negative
                          C) Grouped or Instantaneous"

Example of usage:
search(string, {parse:"regex string", B:"Positive", A:"Maximum"})


IDEAS for Grammar of Time:

###########        Grammar Creation       ##################
1 - Design the encoder and decoder
    a) Encoder:
        1 - Start with the amplitude, first and second derivative
2 - Design a mechanism for data compression
3 - Visualization of a data compression mechanism


###########            Search             ##################
1 - Rule Based search


###########           Synthesis           ##################
1 - Generate signals with the dictionnary created.


###########        Machine Learning             ##################

1 - Design a model to understand a written question and transform it into a query or a signal
2 - Transform a signal into a query (Search based on query)
3 - Build a dictionnary model of each type of signal so that the frequency of sequences is evaluated for each type of signal
    and used to train HMM or RNN.
4 - Generate signals based on the

###########        Visualization              *******************
Test visualizations in plotly and the other library I had found!

"""

import numpy as np
import matplotlib.pyplot as plt
import string
from scipy import signal
from tools.plot_tools import strsignal2color, plot_textcolorized, Csubplot
from tools.string_processing_tools import runLengthEncoding, lstStr2Str, Ngrams, NgramsInt, BagofWords, LegnthConnotation, tf_idf
from tools.PeakFinder import detect_peaks
from GrammarofTime.SSTS.backend import gotstools as gt
from tools.load_tools import loadH5
from scipy.stats import binom
import regex as re

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


def PeakSummary(s, thr, method="absolute"):
    seq_str = AmpChange(s, thr, method="absolute")
    conc_str = ("").join([i for i in seq_str])

    #peaks of type 1 - high rise, and low fall
    peak1 = gt.symbolic_search(conc_str, np.array([seq_str]), "R+z+f+")
    peak2 = gt.symbolic_search(conc_str,  [seq_str], "r+z+F+")
    peak3 = gt.symbolic_search(conc_str, [seq_str], "R+z+F+")
    peak4 = gt.symbolic_search(conc_str, [seq_str], "r+z+f+")
    # rz = gt.symbolic_search(conc_str, [seq_str], "r+z+")


    pks_str = []
    for match1_i in peak1:
        seq_str[match1_i[0]:match1_i[1]-1] = "Rf"
    for match2_i in peak2:
        seq_str[match2_i[0]:match2_i[1]-1] = "rF"
    for match3_i in peak3:
        seq_str[match3_i[0]:match3_i[1]-1] = "RF"
    for match4_i in peak4:
        seq_str[match4_i[0]:match4_i[1]-1] = "rf"
    # for match5_i in rz:
    #     seq_str[match5_i[0]:match5_i[1]] = "rz"

    return seq_str


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
    ds_str, ds_list_str, mergeg_str = gt.connotation([s], "D1 0.01")
    print(ds_str)
    print(ds_list_str)
    pos_matchs = gt.symbolic_search(ds_str, ds_list_str, "p+")
    neg_matchs = gt.symbolic_search(ds_str, ds_list_str, "n+")
    z_matches = gt.symbolic_search(ds_str, ds_list_str, "z+")



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
        vals_str[p_match[0]:p_match[1]-1] = quantilstatesArray2(vals[p_match[0]:p_match[1]-1], thr * max_ampdiff_pos, ["r", "R"], conc=False)

    for z_match in z_matches:
        vals[z_match[0]:z_match[1]] = 0
        vals_str[z_match[0]:z_match[1]] = "z"

    for n_match in neg_matchs:
        vals[n_match[0]:n_match[1]] = abs(s[n_match[0]]-s[n_match[1]-1])
        vals_str[n_match[0]:n_match[1]] = quantilstatesArray2(vals[n_match[0]:n_match[1]], thr * max_ampdiff_neg, ["f", "F"], conc=False)

    return vals_str

def D1Speed(s, thr):
    """
    Design a string based on quick/slow rises/falls and High/low rises/falls on the signal.
    Falling: f/F --> f for low fall and F for high fall
    Rising: r/R --> r for low rise and R for high rise
    Q or q --> for quick or slow derivative if higher or lower trend
    :param s: signal
    :return: string
    """

    ds = np.diff(s)
    ds = np.append(ds, ds[-1])
    x = AmplitudeTrans(abs(ds), 2, ["q", "Q"], thr = thr, method="custom")

    return x

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

def plot_ProbNgramMultiple(signal_clean_str_unique, n):

    #bag of words for each word
    bow_unique = BagofWords(signal_clean_str_unique[1])

    #Ngrams list with N words
    ch1_ngramsN = Ngrams(signal_clean_str_unique[1], n)

    #Ngrams list with N-1 words
    ch1_ngramsn = Ngrams(signal_clean_str_unique[1], n-1)

    #bag of words for the list of N words
    bow_ngramsN = BagofWords(ch1_ngramsN[1])

    #bag of words for the list of N-1words
    bow_ngramsn = BagofWords(ch1_ngramsn[1])

    #Probabilities calculation
    Probs_ch1_nN = LogLikelihoodNgramsMultiple(bow_unique, bow_ngramsN, bow_ngramsn, len(ch1_ngramsN[0]))


    ch1_ngram_total = Ngrams(signal_clean_str_unique[1], n)
    count_ngram = NgramsInt(np.cumsum(signal_clean_str_unique[2]), n)

    for ngram in Probs_ch1_nN:
        print("Ngram:")
        print(ngram)
        # regit = re.finditer(ngram, ch1_clean)
        matches = np.where(np.array(ch1_ngram_total[1]) == ngram)[0]
        print("Matches")
        print(matches)
        for match in matches:
            plt.axvspan(count_ngram[match][0], count_ngram[match][-1], 0, 1, alpha=0.25)
            # plt.plot(ch1)
            plt.show()

# a = np.random.rand(1000)
# checkDistribution(a, 4)
# da = D1Trans(a, 0.05)
# dda = D2Trans(a)

#Exercise 1 - Define Time Series Structure
"""
1 - Original Time Series
2 - Char Conversion based on connotation or attributes (if more than 1 -> have them separately and total)
3 - Word conversion as higher analysis level:
    - With first level: derivative
    - Find second level derivative: quick, slow and high and low rises and falls
    - Find third level: polarity (is it up or down - second derivative)
4 - Include lemmatization model -> what is the lemma of a time series segment?
5 - Include Summarization of time series ---
"""

# #Exercise 2 - Try structure in Simple Triangular Signal
# """
# /\/\/\/\/\/\
# """
# freq = 5
# fs = 10
# x = np.linspace(0, 1, 1000)
# wave = signal.sawtooth(2*np.pi*freq*x, 0.5)
# # waveD1 = D1Trans(wave, 0.05)
# # waveD2 = D2Trans(wave)
# #
# # ampdiff_str = AmpChange(wave, 0.75, "absolute")
# # speed_str = D1Speed(wave)
#
# #Exercise 3 - Designing structures with different triangular shapes
#
# wave1 = signal.sawtooth(2*np.pi*freq*x, 0.5)
# wave2 = 0.75*signal.sawtooth(np.pi*freq*x, 0.5)
# wave3 = wave1+wave2
#
# # amp_level = AmplitudeTrans(wave3, 2, string.ascii_uppercase, method= "quantiles")
#
# ampdiff_str = AmpChange(wave3, 0.75, "absolute")
#
# speed_str = D1Speed(wave3, 0.75)
#
# # wave3_str = addArrayofStrings([amp_level, ampdiff_str, speed_str])
# wave3_str = addArrayofStrings([ampdiff_str, speed_str])
# # wave3_str = addArrayofStrings([amp_level])
#
# wave3_conc_str_tpl = runLengthEncoding(wave3_str)
#
#
# ax1 = plt.subplot(1,1,1)
# # plot_textcolorized(wave3, wave3_conc_str_tpl[2], ax1)
# plot_textcolorized(wave3, wave3_str, ax1)
# plt.show()
#
# ax1 = plt.subplot(1,1,1)
# plot_textcolorized(wave3, wave3_str, ax1)
# plt.show()
#
# #Exercise 4 - Test this new approach on other signals
# from tools.processing_tools import *
# example_path = r"D:\PhD\Code\PhDProject\Hui_SuperProject\Data_Examples\\"
#
# signalHui1 = loadH5(example_path+"arthrokinemat_2018_06_02_23_51_55.h5")
#
# #pre process
# ch1 = smooth(mean_norm(signalHui1[:,5]), 2000)
# ch1 = ch1[1000:]
#
# amp_level = AmplitudeTrans(ch1, 2, string.ascii_uppercase, method= "quantiles")
#
# ampdiff_str = AmpChange(ch1, 0.75, "absolute")
#
# speed_str = D1Speed(ch1, 0.75)
#
# sign_str = SignConnotation(ch1)
#
# wave3_str = addArrayofStrings([sign_str, ampdiff_str, speed_str])
#
# ch1_conc_str, ch1_word_lst, ch1_count_lst = runLengthEncoding(wave3_str)
#
# ch1_connotated_lst_unique, ch1_connotated_lst = LegnthConnotation(ch1_word_lst, ch1_count_lst)
#
#
# import argparse
#
# # def pysequitur_map(str_seq_unique, win_len):
# #     print(str_seq_unique[0:20])
# #     total_str = np.concatenate([str_seq_unique[0:-win_len//2:-1], str_seq_unique, str_seq_unique[-win_len//2:-1:-1]]).tolist()
# #     print(len(str_seq_unique))
# #     print(len(total_str))
# #
# #     for i in range(win_len//2, len(total_str)-win_len//2):
# #         temp_str = total_str[i-win_len//2:i+win_len//2]
# #         print(temp_str)
# #         s_temp = psq.Sequencer2(temp_str)
# #         print(s_temp.get())
# #         print(s_temp.resolve())
#         # psq.print_grammar(s_temp)
#
#
#
# #
# # suffixtree= Tree({"A":ch1_word_lst[:20]})
# # suffixtree.prepare_lca()
# #
# #
# # from graphviz import Source
#
# # dot_object = suffixtree.to_dot()
# # s = Source(dot_object)
# # s.render("test.gv", view=True)
#
# # pysequitur_map(ch1_word_lst, 20)
#
#
# ch1_clean_str = removeShortSegments(ch1_connotated_lst_unique.split(" "), ch1_count_lst)
# ch1_clean_str_unique = runLengthEncoding(ch1_clean_str)
#
#
# # print(ch1_connotated_lst)
# #search for the transitions and add probability in these cases
# ax1 = plt.subplot(1,1,1)
# # plot_textcolorized(wave3, wave3_conc_str_tpl[2], ax1)
# plot_textcolorized(ch1, ch1_connotated_lst, ax1)
# plt.show()
# # print(ch1_clean_str_unique[1])
# #convert to one string spaced
# # ch1_clean = lstStr2Str(ch1_clean_str)
#
# #check n_grams
# #design bag of words for the sequence of n-grams
# bow_ch1 = BagofWords(ch1_clean_str_unique[1])
# ch1_ngrams2 = Ngrams(ch1_clean_str_unique[1], 2)
# bow_ngrams2 = BagofWords(ch1_ngrams2[1])
# ch1_ngrams3 = Ngrams(ch1_clean_str_unique[1], 3)
# bow_ngrams3 = BagofWords(ch1_ngrams3[1])
# ch1_ngrams4 = Ngrams(ch1_clean_str_unique[1], 4)
# bow_ngrams4 = BagofWords(ch1_ngrams4[1])
#
#
# # Probs_ch1_n2 = logLikelihoodNgrams(bow_ch1, bow_ngrams2, len(ch1_ngrams2[0]))
#
# # ch1_clean = lstStr2Str(ch1_clean_str)
#
#
#
#
# # ch1_ngram2_total = Ngrams(ch1_clean_str, 2)
# #
# # for ngram in Probs_ch1_n2:
# #     print("Ngram:")
# #     print(ngram)
# #
# #     # regit = re.finditer(ngram, ch1_clean)
# #     matches = np.where(np.array(ch1_ngram2_total[1]) == ngram)[0]
# #     print("Matches")
# #     print(matches)
# #     for match in matches:
# #         plt.vlines(match, 0, Probs_ch1_n2[ngram]["LL"])
#
#
# #TODO: verificar probabilidades para casos de Ngrams3, 4, 5 e assim sucessivamente
# #Possivelmente terei de implementar uma metrica mais abrangente para fazer o Ngrams,
# #uma vez que dificilmente encontrarei exatamente as mesmas sequencias.
#
# # Probs_ch1_n4 = LogLikelihoodNgramsMultiple(bow_ch1, bow_ngrams4, bow_ngrams3, len(ch1_ngrams4[0]))
# #
# # ch1_ngram4_total = Ngrams(ch1_clean_str_unique[1], 4)
# # count_ngram4 = NgramsInt(np.cumsum(ch1_clean_str_unique[2]), 4)
# #
# #
# # plot_ProbNgramMultiple(ch1_clean_str_unique, 10)
#
# # for ngram in Probs_ch1_n4:
# #     print("Ngram:")
# #     print(ngram)
# #     # regit = re.finditer(ngram, ch1_clean)
# #     matches = np.where(np.array(ch1_ngram4_total[1]) == ngram)[0]
# #     print("Matches")
# #     print(matches)
# #     for match in matches:
# #         plt.axvspan(count_ngram4[match][0], count_ngram4[match][-1], 0, 1, alpha=0.1)
# #
# # plt.show()
#
#
#
# """
# Another Idea: Report the difference between signals
# """
# #
# # Csubplot(3, 1, [[wave3], [abs(np.diff(wave3))], [np.diff(np.diff(wave3))]])
# # plt.show()
#
# #Exercise 5 - Try separating the signal into several documents and apply tf-idf
# divisions=20
# divider = np.linspace(1, len(ch1), divisions).astype(int)
# # print(divider)
# try:
#     tt = np.reshape(ch1, (divisions,-1))
# except Exception as e:
#     print(e)
#
# documents_arr = []
# documents_dict = {}
# documents_list = []
# for i in range(1, int(divisions)):
#     sig = ch1[(divider[i-1]):divider[i]]
#     documents_arr.append(np.array(sig))
#     documents_dict["doc" + str(i)] = ch1_clean_str[(divider[i-1]):divider[i]]
#     documents_list.append(lstStr2Str(ch1_clean_str[(divider[i-1]):divider[i]]))
#
#
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.cluster import KMeans
# # from sklearn.decomposition import PCA
# # from sklearn.preprocessing import normalize
#
# # print(documents_list)
#
# # vectorizer = TfidfVectorizer()
# # X = vectorizer.fit_transform(documents_list)
# # tf_idf_norm = normalize(X)
# # tf_idf_arr = tf_idf_norm.toarray()
# # # print(vectorizer.get_feature_names())
# #
# # tf_idf_dict = tf_idf(documents_dict, set(ch1_clean_str))
# #
# # sklearnpca = PCA(n_components=2)
# # Y_sklearn = sklearnpca.fit_transform(tf_idf_arr)
# # kmeans = KMeans(n_clusters=3).fit(tf_idf_arr)
# # fitted = kmeans.fit(Y_sklearn)
# # predicted = kmeans.predict(Y_sklearn)
# #
# # plt.figure()
# # plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], c=predicted, s=50, cmap='viridis')
# # centers = kmeans.cluster_centers_
# # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=300, alpha=0.6)
# # plt.show()
# # for word in set(ch1_clean_str):
# #     for doc in documents_dict:
# #         print("Doc "+doc)
# #         print("Word " + word)
# #         print(tf_idf_dict[0][doc][word])
