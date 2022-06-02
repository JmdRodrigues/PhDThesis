import sys

import numpy as np

from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB

from pyts.classification import KNeighborsClassifier

from tools.tools_ssts import BoW_cv_train_document, BoW_test_document, BoW_tfidf_train_document
from tools.style_tools import phd_thesis_cmap1

import re
import json

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
import matplotlib.gridspec as grid_spec

from novainstrumentation import smooth

import pandas as pd

def KNNEuclid(X_train, y_train, X_test, y_test):

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred_eucl = knn.predict(X_test)

    return y_pred_eucl


def pre_model_SVMClassifier(Bow_vec_train, BoW_vec_test, y_train, y_test):
    model = LinearSVC(tol=1.0e-6,max_iter=5000,verbose=0)
    # model = SVC(tol=1.0e-6,max_iter=5000,verbose=0, probability=True)
    model.fit(Bow_vec_train, y_train)
    y_pred = model.predict(BoW_vec_test)
    # y_pred_prob = model.predict_proba(BoW_vec_test)

    # labels = np.unique(y_test)
    # y_pre_pred_acc = [1 if y_tr_i in labels[np.argsort(y_pred_proba_i)[::-1][:4]] else 0 for y_tr_i, y_pred_proba_i in
    #                   zip(y_test, y_pred_prob)]

    # mean_probs = []
    # for label in np.unique(y_test):
    #     pos_i = np.where(y_test == label)[0]
    #     mat_probs_i = np.mean(y_pred_prob[pos_i], axis=0)
    #     mean_probs.append(mat_probs_i)

    return y_pred

def pre_model_naiveBclassifier(BoW_vec_train, BoW_vec_test, y_train, y_test):

    naive_bayes_classifier = MultinomialNB()
    naive_bayes_classifier.fit(BoW_vec_train, y_train)
    y_pred = naive_bayes_classifier.predict(BoW_vec_test)
    y_pred_prob = naive_bayes_classifier.predict_proba(BoW_vec_test)

    labels = np.unique(y_test)
    y_pre_pred_acc = [1 if y_tr_i in labels[np.argsort(y_pred_proba_i)[::-1][:4]] else 0 for y_tr_i, y_pred_proba_i in
                      zip(y_test, y_pred_prob)]

    mean_probs = []
    for label in np.unique(y_test):
        pos_i = np.where(y_test == label)[0]
        mat_probs_i = np.mean(y_pred_prob[pos_i], axis=0)
        mean_probs.append(mat_probs_i)

    return y_pred, mean_probs, y_pre_pred_acc

def BofShapesModel(docs_train, y_train, docs_test, y_test, config, clf, bow):

    ngram_i = config["ngram"]
    # ngram_i = [1, 2]
    min_df_i = config["mindf"]
    max_df_i = config["maxdf"]

    if(clf == "NB"):
        if(bow == "CV"):
            train_vecs_cv, c_model_cv = BoW_cv_train_document(docs_train, ngram_i, min_df_i, max_df_i)
            test_cv_vecs = BoW_test_document(c_model_cv, docs_test)
            y_pred, _, _ = pre_model_naiveBclassifier(train_vecs_cv, test_cv_vecs, y_train, y_test)
        elif(bow == "TFIDF"):
            train_vecs_tfidf, c_model_tfidf = BoW_tfidf_train_document(docs_train, ngram_i, min_df_i, max_df_i)
            test_tfidf_vecs = BoW_test_document(c_model_tfidf, docs_test)
            y_pred, _, _ = pre_model_naiveBclassifier(train_vecs_tfidf, test_tfidf_vecs, y_train, y_test)
        else:
            print("Choose a correct word vec model")
            return 0

    elif (clf == "SVM"):
        if (bow == "CV"):
            train_vecs_cv, c_model_cv = BoW_cv_train_document(docs_train, ngram_i, min_df_i, max_df_i)
            test_cv_vecs = BoW_test_document(c_model_cv, docs_test)
            y_pred = pre_model_SVMClassifier(train_vecs_cv, test_cv_vecs, y_train,
                                                                                 y_test)
        elif (bow == "TFIDF"):
            train_vecs_tfidf, c_model_tfidf = BoW_tfidf_train_document(docs_train, ngram_i, min_df_i, max_df_i)
            test_tfidf_vecs = BoW_test_document(c_model_tfidf, docs_test)
            y_pred = pre_model_SVMClassifier(train_vecs_tfidf, test_tfidf_vecs, y_train, y_test)

            return y_pred, c_model_tfidf, test_tfidf_vecs

        else:
            print("Choose a correct word vec model")
            return 0
    else:
        print("Choose a correct Classifier")
        return 0

    return y_pred

def pattern_relevance(signal, sig_vec, vec_patterns, text, matches):
    # plt.plot(signal, color="orange")
    # sig_vec = (sig_vec-min(sig_vec))/(max(sig_vec)-min(sig_vec))
    weight_vec = np.zeros(len(signal))
    words_ = []
    vals_ = []
    text = re.sub("\. || !\s", "", text)


    for pattern, vec_val in zip(vec_patterns, sig_vec):
        words_.append(pattern)
        vals_.append(vec_val)
        pattern_match = [match.span() for match in re.finditer(pattern, text)]
        # print(pattern_match)
        # print(matches)
        if(len(pattern_match)>0):
            for match_i in pattern_match:
                ind_of_matches_a = len(list(filter(str.strip, text[:match_i[0]].split(" "))))
                ind_of_matches_b = ind_of_matches_a+len(pattern.split(" "))
                matches_i = matches[ind_of_matches_a:ind_of_matches_b]
                for match in matches_i:
                    # print(match)
                    weight_vec[match[0]:match[1]] += vec_val
                    # print(match[0])
                    # print(match[1])
            #         plt.plot(range(match[0], match[1]), signal[match[0]:match[1]], alpha=vec_val, color="blue")
            #         plt.xlim([0, len(signal)])
            # plt.title(pattern)
            # plt.show()
    # weight_vec = (weight_vec-min(weight_vec))/(max(weight_vec)-min(weight_vec))
    weight_vec = 0.01 + weight_vec
    plotColoredLine(smooth(signal, 25), weight_vec, "YlGnBu")
        # plt.scatter(range(len(signal)), signal, c=weight_vec, cmap="YlGnBu")
    plt.show()
    words_ = np.array(words_)
    return words_[np.argsort(vals_)[::-1]], vals_

def plotColoredLine(s, weight, cmap):
	x = np.linspace(0, len(s), len(s))
	y = s
	points = np.array([x, y]).T.reshape(-1, 1, 2)
	segments = np.concatenate([points[:-1], points[1:]], axis=1)

	fig, ax = plt.subplots(1, 1)
	norm = plt.Normalize(min(weight), max(weight))
	# lc = LineCollection(segments, cmap=cmap, norm=colors.PowerNorm(gamma=0.05))
	# lc = LineCollection(segments, cmap=cmap, norm=colors.LogNorm(vmin=np.min(weight), vmax=np.max(weight)))
	lc = LineCollection(segments, cmap=cmap, norm=norm)
	lc.set_array(weight)
	lc.set_linewidth(2)
	line = ax.add_collection(lc)
	fig.colorbar(line, ax=ax)
	ax.set_xlim(min(x), max(x))
	ax.set_ylim(min(y)+0.1*min(y), max(y)+0.1*max(y))
	plt.show()

def sort_coo(tfidf_matrix):
    tuples = zip(tfidf_matrix.col, tfidf_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def load_best_params(key_dataset):
    f_a = open(r"C:\Users\joao0\OneDrive\Documentos\PhDProjects\UCR\BoSP\OverallUCRResults\COMB.json", )
    # f_a2 = open(r"C:\Users\joao0\OneDrive\Documentos\PhDProjects\UCR\BoSP\OverallUCRResults\CV_NB.json", )
    # f_a3 = open(r"C:\Users\joao0\OneDrive\Documentos\PhDProjects\UCR\BoSP\OverallUCRResults\CV_SVM.json", )
    # f_a4 = open(r"C:\Users\joao0\OneDrive\Documentos\PhDProjects\UCR\BoSP\OverallUCRResults\TFIDF_NB.json", )
    f_a5 = open(r"C:\Users\joao0\OneDrive\Documentos\PhDProjects\UCR\BoSP\OverallUCRResults\TFIDF_SVM.json", )

    a = json.load(f_a)
    a5 = json.load(f_a5)


    try:
        best_conf = a[key_dataset][np.argmax(a5[key_dataset])]
        return best_conf, max(a5[key_dataset])
    except:
        print("Error...no file found")

def pattern_relevance_grid(signal, sig_vec, vec_patterns, text, matches, ax_):
    weight_vec = np.zeros(len(signal))
    words_ = []
    vals_ = []
    text = re.sub("\. || !\s", "", text)

    for pattern, vec_val in zip(vec_patterns, sig_vec):
        words_.append(pattern)
        vals_.append(vec_val)
        pattern_match = [match.span() for match in re.finditer(pattern, text)]

        if (len(pattern_match) > 0):
            for match_i in pattern_match:
                ind_of_matches_a = len(list(filter(str.strip, text[:match_i[0]].split(" "))))
                ind_of_matches_b = ind_of_matches_a + len(pattern.split(" "))
                matches_i = matches[ind_of_matches_a:ind_of_matches_b]
                for match in matches_i:
                    weight_vec[match[0]:match[1]] += vec_val

    # weight_vec = (weight_vec-min(weight_vec))/(max(weight_vec)-min(weight_vec))
    weight_vec = 0.01 + weight_vec
    plotColoredLine_with_ax(signal, weight_vec, "YlGnBu", ax_)

    words_ = np.array(words_)
    return words_[np.argsort(vals_)[::-1]]


def plot_distance_mat_classes(X, y, sig_vecs, vec_names, text, matches):
    y_ = np.unique(y)
    #create matplotlib matrix subplots
    # Create four polar axes and access them through the returned array
    fig, axs = plt.subplots(len(y_), len(y_))

    for y_i in y_:
        for y_ii in y_:
            if(y_i != y_ii):
                i_1 = np.where(y_i == y)[0]
                i_2 = np.where(y_ii == y)[0]

                signal_1 = X[i_1[0]]
                signal_2 = X[i_2[0]]

                sig1 = np.sum(sig_vecs[i_1], axis=0)
                sig2 = np.sum(sig_vecs[i_2], axis=0)

                sig_dif = np.abs(sig1 - sig2)

                text1 = text[i_1[0]]
                matches1 = matches[i_1[0]]

                text2 = text[i_2[0]]
                matches2 = matches[i_2[0]]

                pattern_relevance_grid(signal_1, sig_dif, vec_names, text1, matches1, axs[y_i-1, y_ii-1])
                pattern_relevance_grid(signal_2, sig_dif, vec_names, text2, matches2, axs[y_ii-1, y_i-1])

    plt.show()

def plot_distance_oneclassesVSall(X, y, sig_vecs, vec_names, text, matches):
    y_ = np.unique(y)
    #create matplotlib matrix subplots
    # Create four polar axes and access them through the returned array


    # all_sums = [np.sum(sig_vecs[i], axis=0) for i in y_]

    results_ = []
    for n in range(0, 10):
        fig, axs = plt.subplots(1, len(y_))
        for y_i in y_:
            print(n)
            i_1 = np.where(y_i == y)[0]
            i_2 = np.where(y_i != y)[0]

            signal_1 = X[i_1[n]]

            sig1 = np.sum(sig_vecs[i_1], axis=0)
            sigall = np.sum(sig_vecs[i_2], axis=0)/(len(y_)-1)

            sig_dif = np.abs(sig1 - sigall)

            text1 = text[i_1[n]]
            matches1 = matches[i_1[n]]

            pattern_relevance_grid(signal_1, sig_dif, vec_names, text1, matches1, axs[y_i-1])
            vec_names = np.array(vec_names)
            results = vec_names[np.argsort(sig1)[::-1]]
            results_.append(results[:10])
        plt.show()
    return results_

def plotColoredLine_with_ax(s, weight, cmap, ax_):
	x = np.linspace(0, len(s), len(s))
	y = s
	points = np.array([x, y]).T.reshape(-1, 1, 2)
	segments = np.concatenate([points[:-1], points[1:]], axis=1)
	# lc = LineCollection(segments, cmap=cmap, norm=colors.PowerNorm(gamma=0.05))
	# lc = LineCollection(segments, cmap=cmap, norm=colors.LogNorm(vmin=np.min(weight), vmax=np.max(weight)))
	lc = LineCollection(segments, cmap=cmap)
	lc.set_array(weight)
	lc.set_linewidth(2)
	line = ax_.add_collection(lc)
	# fig.colorbar(line, ax=ax)
	ax_.set_xlim(min(x), max(x))
	ax_.set_ylim(min(y)+0.1*min(y), max(y)+0.1*max(y))
    
def set_colormap(docs):
    lpcmap = phd_thesis_cmap1()
    color_range = int(256 / len(docs)) - 1

    return lpcmap, color_range

def createDistributionPlots(fig, gs, count_vecs, max_per_col, sm_d):
    lpcmap, color_range = set_colormap(count_vecs)
    ax_objs = []
    for i, vecs in enumerate(count_vecs):
        if(i==0):
            ax_objs.append(fig.add_subplot(gs[i, 1]))
        else:
            ax_objs.append(fig.add_subplot(gs[i, 1], sharex=ax_objs[-1]))
        rect = ax_objs[-1].patch
        rect.set_alpha(0)
        # remove borders, axis ticks, and labels
        ax_objs[-1].set_yticklabels([])
        ax_objs[-1].set_ylabel('')
        if i == len(count_vecs) - 1:
            ax_objs[-1].set_xticks(np.arange(len(vecs.keys())))
#             ax_objs[-1].set_xticklabels(vecs.keys())
#             plt.setp(ax_objs[-1].get_xticklabels(), rotation=45, ha="right",
#                      rotation_mode="anchor")
        else:
            # ax_objs[-1].set_xticklabels([])
            plt.setp(ax_objs[-1].get_xticklabels(), visible=False)

        spines = ["top", "right", "left", "bottom"]
        for s in spines:
            ax_objs[-1].spines[s].set_visible(False)


        for vec in vecs.values:
            # # print(vec)
            # # vec = np.exp2(vec)
            ax_objs[-1].bar(np.arange(0, len(vec)), vec, alpha=0.25, align="center",
                       color=lpcmap(i*color_range))
            # axs[i].plot(ni.smooth(vec, 5), color="b")
            # ax_objs[-1].fill_between(np.arange(0, len(vec)), ni.smooth(vec ** 2, sm_d),
            #                          color=lpcmap(i * color_range), alpha=0.1)
            # ax_objs[-1].set_ylim(0, 1.2)
            ax_objs[-1].tick_params(left=False, bottom=False)
        ax_objs[-1].fill_between(np.arange(0, len(vecs.max(axis=0).values)),
                                 smooth((vecs.max(axis=0).values), sm_d), alpha=0.3, color=lpcmap(i * color_range),
                                 edgecolor=lpcmap(i * color_range))
        # gs.update(hspace=-0.5)
        plt.tight_layout()

    return ax_objs

def createSignalsPlots(fig, gs, X_train):
    lpcmap, color_range = set_colormap(X_train)
    ax_objs = []
    for i, vecs in enumerate(X_train):
        ax_objs.append(fig.add_subplot(gs[i, 0]))
        rect = ax_objs[-1].patch
        rect.set_alpha(0)
        # remove borders, axis ticks, and labels
        ax_objs[-1].set_yticklabels([])
        ax_objs[-1].set_ylabel('')
        if i == len(X_train) - 1:
            pass
        else:
            ax_objs[-1].set_xticklabels([])

        spines = ["top", "right", "left", "bottom"]
        for s in spines:
            ax_objs[-1].spines[s].set_visible(False)
        ax_objs[-1].tick_params(left=False, bottom=False)
        [ax_objs[-1].plot(vec, color=lpcmap(i * color_range), alpha=0.2) for vec in vecs]
    plt.tight_layout()

    
def plotBoWdict(docs, X_train, Bow_vec):

    count_vecs = [pd.DataFrame(Bow_vec.transform(doc).toarray(), columns=Bow_vec.get_feature_names()) for doc in docs]
    merged_df = pd.concat(count_vecs)
    max_per_col = merged_df.max(axis=0)

    #figure and grid
    fig = plt.figure(figsize=(10, 4))
    gs = (grid_spec.GridSpec(len(count_vecs), 2))

    #if normalize, select a normalization method
    # data_normalizer = colors.NoNorm()
    ax_objs1 = createDistributionPlots(fig, gs, count_vecs, max_per_col, 10)
    createSignalsPlots(fig, gs, X_train)
    # createKeywordsPlot(fig, gs, ax_objs1, count_vecs, 3)

    plt.show()

def BoWDistributionRepresentation(X_train, X_train_docs, cv_model, y_train):
    docs_per_cat = [[docs_train_ii for y_ii, docs_train_ii in zip(y_train, X_train_docs) if y_ii==y_i] for y_i in np.unique(y_train)]

    X_train_per_cat = [[smooth(X_ii, 10) for y_ii, X_ii in zip(y_train, X_train) if y_ii==y_i] for y_i in np.unique(y_train)]

    plotBoWdict(docs_per_cat, X_train_per_cat, cv_model)

def BOWDistributionSignal(sig, sig_doc, cv_model, color_i, name):
    #figure and grid
    fig = plt.figure(figsize=(10, 4))
    gs = (grid_spec.GridSpec(1, 2))

    sig_vec = pd.DataFrame(cv_model.transform(sig_doc).toarray(), columns=cv_model.get_feature_names())

    sm_d = 10

    ax_sig = fig.add_subplot(gs[0, 0])
    ax_sig.plot(sig)

    ax_objs = fig.add_subplot(gs[0, 1])
    ax_objs.bar(np.arange(0, len(sig_vec.values[0])), sig_vec.values[0], alpha=0.8, align="center", color=color_i)
    ax_objs.fill_between(np.arange(0, len(sig_vec.values[0])),
                                 smooth(sig_vec.values[0], sm_d), alpha=0.3, color=color_i,
                                 edgecolor=color_i)
    
    ax_sig.set_yticklabels([])
    ax_sig.set_ylabel('')
    ax_objs.set_yticklabels([])
    ax_objs.set_ylabel('')
    spines = ["top", "right", "left", "bottom"]
    for s in spines:
        ax_objs.spines[s].set_visible(False)
        ax_sig.spines[s].set_visible(False)
    ax_objs.tick_params(left=False, bottom=False)
    ax_sig.tick_params(left=False, bottom=False)
    plt.setp(ax_objs.get_xticklabels(), visible=False)
    plt.setp(ax_sig.get_xticklabels(), visible=False)
    rect = ax_objs.patch
    rect.set_alpha(0)
    rect = ax_sig.patch
    rect.set_alpha(0)
    
    fig.savefig(name+".svg")