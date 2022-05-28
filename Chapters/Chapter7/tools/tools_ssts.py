import numpy as np
from gsheets import Sheets

import re

import pandas as pd

from tools.SSTS.backend.gotstools import connotation, symbolic_search
import tools.SSTS.backend.gotstools as gt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def get_gsheet_access():
    # sheets = Sheets.from_files('D:\PhD\CodeProjects\Code\PhDProject\GrammarofTime\ExcelDocs\client_secrets.json',
    #                            'D:\PhD\CodeProjects\Code\PhDProject\GrammarofTime\ExcelDocs\storage.json')
    # url = 'https://docs.google.com/spreadsheets/d/1LfwSPn99YfHBvPnyKCKmILupEz44igSpe6pkdSgHAX4/edit#gid=977182459'
    #
    # s = sheets.get(url)

    # shape_patterns = s.sheets[-1].to_csv(make_filename=r"D:\PhD\CodeProjects\Code\PhDProject\GrammarofTime\ExcelDocs\search.csv")

    # search_patterns_df = pd.read_csv(r'D:\PhD\CodeProjects\Code\PhDProject\GrammarofTime\ExcelDocs\search.csv')
    search_patterns_df = pd.read_csv(r"tools\SSTS_functions.csv", delimiter=";")

    return search_patterns_df

def load_connotations_patterns(gsheet_df):
    pre_processing = gsheet_df["Pre-Processing"]
    connotation = gsheet_df["Connotation String"]
    search_string = gsheet_df["Symbolic Search"]
    description = gsheet_df["Description"]
    group = gsheet_df["Group"]
    tag = gsheet_df["Symbolic Reduction"]

    return group, pre_processing, connotation, search_string, description, tag


def connotation_and_search():
    # load patterns
    patterns_df = get_gsheet_access()
    print(patterns_df)
    # patterns_df = pd.read_csv("shapes.csv")
    group, pre_processings, connotations, search_strings, descriptions, tag = load_connotations_patterns(patterns_df)

    return group, pre_processings, connotations, search_strings, descriptions, tag


def load_config(win_len, d1_thr, pk_size, groups="all"):
    group, pre_processings, connotations, search_strings, descriptions, tag = connotation_and_search()

    GROUP = group
    PRE_PROCESSING = pre_processings
    CONNOTATIONS = connotations
    SEARCH_STRINGS = search_strings
    TAG = tag

    df_replace_pp = np.array([re.sub("win_size", str(win_len), pp_i) for pp_i in PRE_PROCESSING])
    df_replace_con = np.array([re.sub("thr", str(d1_thr), con_i) for con_i in CONNOTATIONS])
    df_replace_con = np.array([re.sub("win_size", str(win_len), con_i) for con_i in df_replace_con])
    df_replace_search = np.array([re.sub("size", str(pk_size), s_i) for s_i in SEARCH_STRINGS])

    PRE_PROCESSING[:] = df_replace_pp
    CONNOTATIONS[:] = df_replace_con
    SEARCH_STRINGS[:] = df_replace_search

    # print(PRE_PROCESSING)
    # print(CONNOTATIONS)
    # print(SEARCH_STRINGS)
    
    if(groups=="all"):
        config = {"tag": TAG, "group": GROUP, "pre_processing": PRE_PROCESSING, "connotation": CONNOTATIONS,
                  "search": SEARCH_STRINGS}
    else:
        config = {"tag": TAG[:22], "group": GROUP[:22], "pre_processing": PRE_PROCESSING[:22], "connotation": CONNOTATIONS[:22],
                  "search": SEARCH_STRINGS[:22]}

    return config


def get_Docs(X_train, X_test, config):
    docs_train, docs_train_matches = Documents_text(X_train, config)
    docs_test, docs_test_matches = Documents_text(X_test, config)

    return docs_train, docs_test, docs_train_matches, docs_test_matches


def Documents_text(X, config):
    docs, docs_matches = pre_document_selection(X, config)

    return docs, docs_matches

def pre_document_selection(X, config):
    documents = []
    documents_matches = []
    sds = 0
    for s_iterator, signal in enumerate(X):
        sorted_tags = ""
        sorted_matches = []
        for g_i in np.unique(config["group"]):
            ordered_tags = []
            match_init = []
            all_matches = []
            g_indexes = np.where(config["group"] == g_i)[0]
            pp_sig = gt.pre_processing(signal, config["pre_processing"][g_indexes[0]])
            intermediate_con = config["connotation"][g_indexes]
            intermediate_search = config["search"][g_indexes]
            intermediate_tags = config["tag"][g_indexes]
            for con_i in np.unique(intermediate_con):
                #if dataframe:
                c_indexes = g_indexes[0]+np.where(config["connotation"][g_indexes] == con_i)[0]
                intermediate_search_2 = intermediate_search[c_indexes]
                intermediate_tags_2 = intermediate_tags[c_indexes]
                con_r = connotation([pp_sig], con_i)
                for search_i, tag_i in zip(intermediate_search_2, intermediate_tags_2):
                    matches = gt.symbolic_search(con_r[0], con_r[1], search_i)
                    if(len(matches)>0):
                        [ordered_tags.append(tag_i) for match in matches]
                        [match_init.append(match[0]) for match in matches]
                        [all_matches.append(match) for match in matches]

                # if (sds % 100 == 0):
                #     plot_textcolorized(pp_sig, con_r[-1], plt.subplot(111))
                #     plt.show()
            sorted_matches += [x for _, x in sorted(zip(match_init, all_matches))]
            sorted_tags += ". " + " ".join([x for _,x in sorted(zip(match_init,ordered_tags))])

        # sds+=1
        # if(sds%1000==0):
        #     plt.plot(smooth(signal, 1))
        #     print(sorted_tags)
        #     plt.show()
        # sds+=1
        documents.append(sorted_tags)
        documents_matches.append(sorted_matches)

    return documents, documents_matches

def BoW_cv_train_document(Docs_train, ngrams, min_df=1, max_df=1.0):
    c_vectorize = CountVectorizer(lowercase=False, ngram_range=ngrams, token_pattern=r"(?u)\b\w+\b", min_df=min_df, max_df=max_df)
    X_vec = c_vectorize.fit_transform(Docs_train)

    return X_vec, c_vectorize

def BoW_tfidf_train_document(Docs_train, ngrams, min_df=1, max_df=1.0):
    c_vectorize = TfidfVectorizer(lowercase=False, ngram_range=ngrams, token_pattern=r"(?u)\b\w+\b", min_df=min_df, max_df=max_df)
    X_vec = c_vectorize.fit_transform(Docs_train)

    return X_vec, c_vectorize

def BoW_test_document(c_model, Docs_test):
    # X_vec = c_model.transform(Docs_test.tolist())
    X_vec = c_model.transform(Docs_test)

    return X_vec