import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

import numpy as np
from pyts import datasets
from BoSP.tools_ssts import load_config, get_Docs
from BoSP.tools import BofShapesModel, KNNEuclid, pattern_relevance, load_best_params, plot_distance_mat_classes, plot_distance_oneclassesVSall

dataset_lst = datasets.ucr_dataset_list()

dataset = "LargeKitchenAppliances"


p, p_best = load_best_params(dataset)

print(p)
print(p_best)

pk_size = p["pk_s"]
thr = p["thr"]
win_len = p["win"]

#load configuration
config = load_config(win_len=win_len, d1_thr=thr, pk_size=pk_size)

#load data
data = datasets.fetch_ucr_dataset(dataset, use_cache=True, data_home=None, return_X_y=False)
X_train = data["data_train"]
y_train = data["target_train"]
X_test = data["data_test"]
y_test = data["target_test"]

docs_train, docs_test, docs_train_matches, docs_test_matches = get_Docs(X_train, X_test, config)

# y_pred_nb_cv = BofShapesModel(docs_train, y_train, docs_test, y_test, config, "NB", "CV")
# y_pred_nb_tfidf = BofShapesModel(docs_train, y_train, docs_test, y_test, config, "NB", "TFIDF")
# y_pred_svm_cv = BofShapesModel(docs_train, y_train, docs_test, y_test, config, "SVM", "CV")
y_pred_svm_tfidf, model_tfidf_svm, test_vecs = BofShapesModel(docs_train, y_train, docs_test, y_test, p, "SVM", "TFIDF")

vec_pattern_names = model_tfidf_svm.get_feature_names()
test_vecs_arr = test_vecs.toarray()

cf = confusion_matrix(y_test, y_pred_svm_tfidf)
print(accuracy_score(y_test, y_pred_svm_tfidf))
sns.heatmap(cf, annot=True, cmap="YlGnBu")
plt.show()

# for y_i in np.unique(y_test):
# # for i in np.linspace(0, len(y_test)-1, 10).astype(int):
#     i = np.where(y_test==y_i)[0]
#
#     # sig = np.sum(test_vecs_arr[i], axis=0)
#     sig = test_vecs_arr[i[0]]
#     text1 = docs_test[i[0]]
#     matches1 = docs_test_matches[i[0]]
#     results, vals = pattern_relevance(X_test[i[0]], sig, vec_pattern_names, text1, matches1)
#     print(results)


#try to plot every difference:
# plot_distance_mat_classes(X_test, y_test, test_vecs_arr, vec_pattern_names, docs_test, docs_test_matches)
results = plot_distance_oneclassesVSall(X_test, y_test, test_vecs_arr, vec_pattern_names, docs_test, docs_test_matches)
print(results)
#try differences between classes:
y1 = 2
i1 = np.where(y_test==y1)[0]
sig1 = np.sum(test_vecs_arr[i1], axis=0)
y2 = 8
i2 = np.where(y_test==y2)[0]
sig2 = np.sum(test_vecs_arr[i2], axis=0)

sig_dif = np.abs(sig1-sig2)
text1 = docs_test[i1[0]]
matches1 = docs_test_matches[i1[0]]
results, vals = pattern_relevance(X_test[i1[0]], sig_dif, vec_pattern_names, text1, matches1)
print(results)

text2 = docs_test[i2[0]]
matches2 = docs_test_matches[i2[0]]
results2, vals2 = pattern_relevance(X_test[i2[0]], sig_dif, vec_pattern_names, text2, matches2)
print(results2)

