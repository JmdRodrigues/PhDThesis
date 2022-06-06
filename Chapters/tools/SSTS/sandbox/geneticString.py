from tools.processing_tools import *
from tools.load_tools import *
from tools.string_processing_tools import Seq_StringDistance, mostFreqSeq, leastFreqSeq, BagofWords, tf_idf, runLengthEncoding, Ngrams, Decode_tfDict
import novainstrumentation as ni
from GrammarofTime.SSTS.backend.gotstools import *
from PDF_generator.reportGen import Report
from sklearn.neighbors import KernelDensity
from definitions import CONFIG_PATH
from tools.plot_tools import plot_config, font_config, subplot_pars, plot_textcolorized, strsignal2color
from pandas import read_json
# from textblob import TextBlob



guide = CONFIG_PATH + "/Hui_SuperProject/MovDict.json"
guide_dict = read_json(guide).to_dict()
example_path = CONFIG_PATH + "/Hui_SuperProject/Data_Examples/"
# pdf_config
doc = Report("libphys", "report_AnnotationFiles")

for key in guide_dict.keys():
    signal = loadH5(example_path + guide_dict[key]["file"])

    fs = 1000
    b = 10
    # for i in range(5, 8):
    ch1 = signal[b * fs:(len(signal) // (guide_dict[key]["divider"] // 4)) + b * fs, 7]
    ref = signal[b * fs:(len(signal) // (guide_dict[key]["divider"] // 4)) + b * fs, -1]
    # ch1 = (ch1-np.mean(ch1))/np.std(ch1)
    # ch1_pp = pre_processing(ch1, "M LP 2")
    ch1_pp = pre_processing(ch1, "M")
    ref = pre_processing(ref, "M")


    t = np.linspace(0, len(ch1_pp) / 1000, len(ch1_pp))

    hist, bins = np.histogram(ch1_pp, bins=6)
    pd_s = prob_hist(ch1_pp, hist, bins, inverted=False, log=True)

    # plot config
    plot_config()
    font0, font1, font2 = font_config()

    # for windown_len in [10, 100, 250, 500, 1000]:
    windown_len = 500

    proc_ch1 = NewWindowStat(ch1_pp, ["std", "mean", "Azcr"], 1000, windown_len)


    proc_str = []

    all_feat_str = []
    docs = {}
    docs_time = {}
    docs_conc = {}
    for nbr, feat in enumerate(proc_ch1.T):
        # print(feat)
        quantile_vals = np.quantile(feat, [0.25, 0.5, 0.75])

        str_feat = quantilstatesArray(feat, quantile_vals, conc=False)
        ngrams_str_feat_0= Ngrams(str_feat, n=4)
        print(len(str_feat))
        print(len(ngrams_str_feat_0))

        rl_str, rl_str_feat, rl_counts = runLengthEncoding(str_feat)

        ngrams_str_feat= Ngrams(rl_str_feat, n=4)

        docs_conc[nbr] = ngrams_str_feat_0
        docs[nbr] = ngrams_str_feat
        docs_time[nbr] = rl_counts[:-3]
        all_feat_str += ngrams_str_feat




    #get set of words with n_gram method
    #set of unique words for all documents
    sets = set(all_feat_str)
    tf_idfDict, tfDict = tf_idf(docs, sets)

    tf_Seq = Decode_tfDict(docs, tfDict)
    print(tf_idfDict)
    print(tfDict)
    print(tf_Seq)
    plt.plot(np.cumsum(docs_time[0]), tf_Seq[0], 'o')
    plt.plot(np.cumsum(docs_time[1]), tf_Seq[1], 'o')
    plt.plot(np.cumsum(docs_time[2]), tf_Seq[2], 'o')
    plt.show()


        # plot_textcolorized(ch1_pp, str_feat, plt.subplot(1,1,1))
        # plt.show()






        # str_feat2 = quantilstatesArray(feat, quantile_vals, conc=False)
    #
        # proc_str.append([str_feat2])
    #
    # proc_str = np.array(proc_str)
    # conc_proc_str = concat_np_strings(proc_str, 0)
    #
    # ax1 = plt.subplot(1, 1, 1)
    # ax1.plot(ref)
    # strsignal2color(ch1_pp, conc_proc_str[0], ax=ax1)
    # plt.show()

    proc_str = np.array(proc_str)
    conc_proc_str = concat_np_strings(proc_str, 0).tolist()


    dist_str_methods = ["hamming", "levenshtein", "damerau_lev", "jaccard", "tversky", "tanimoto",
     "cosine", "bwt"]

    print(BagofWords(conc_proc_str))
    seq_max = mostFreqSeq(conc_proc_str)
    seq_min = leastFreqSeq(conc_proc_str)

    print(seq_max)
    print(seq_min)

    # for seq in set(conc_proc_str):
        # for method in dist_str_methods:

    dist_1 = Seq_StringDistance(conc_proc_str, seq_max, dist_str_methods[0])
    dist_2 = Seq_StringDistance(conc_proc_str, seq_min, dist_str_methods[0])

    plt.plot(ref)
    plt.plot(dist_1)
    plt.plot(dist_2)

    plt.show()













