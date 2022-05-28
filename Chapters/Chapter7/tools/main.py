from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score

from pyts import datasets
from tools_ssts import load_config, get_Docs
from tools import BofShapesModel, KNNEuclid, pattern_relevance

sns.set()

dataset_lst = datasets.ucr_dataset_list()

pk_size = 10
thr = 0.05
win_len = 25
config = load_config(win_len=win_len, d1_thr=thr, pk_size=pk_size)

data = datasets.fetch_ucr_dataset("GunPoint", use_cache=True, data_home=None, return_X_y=False)
X_train = data["data_train"]
y_train = data["target_train"]
X_test = data["data_test"]
y_test = data["target_test"]

docs_train, docs_test, docs_train_matches, docs_test_matches = get_Docs(X_train, X_test, config)
y_pred_nb_cv = BofShapesModel(docs_train, y_train, docs_test, y_test, config, "NB", "CV")
y_pred_nb_tfidf = BofShapesModel(docs_train, y_train, docs_test, y_test, config, "NB", "TFIDF")
y_pred_svm_cv = BofShapesModel(docs_train, y_train, docs_test, y_test, config, "SVM", "CV")
y_pred_svm_tfidf, model_tfidf_svm, test_vecs = BofShapesModel(docs_train, y_train, docs_test, y_test, config, "SVM", "TFIDF")
y_pred_eucl = KNNEuclid(X_train, y_train, X_test, y_test)

cf = confusion_matrix(y_test, y_pred_svm_tfidf)

sns.heatmap(cf, annot=True, cmap="YlGnBu")
plt.show()

vec_pattern_names = model_tfidf_svm.get_feature_names()
test_vecs_arr = test_vecs.toarray()

for y_i in np.unique(y_test):
# for i in np.linspace(0, len(y_test)-1, 10).astype(int):
    i = np.where(y_test==y_i)[0][0]
    sig = test_vecs_arr[i]
    text1 = docs_test[i]
    matches1 = docs_test_matches[i]
    pattern_relevance(X_test[i], sig, vec_pattern_names, text1, matches1)
