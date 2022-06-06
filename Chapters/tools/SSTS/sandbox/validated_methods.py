from GrammarofTime.SSTS.backend import gotstools as gt
from GrammarofTime.SSTS.sandbox.plt_drawing import get_ts_drawing, get_ts_drawing0

import matplotlib.pyplot as plt

def gt_split(s, connotation, search, pos="b"):
   """
   split time series based on a search mechanism
   :param s:
   :param connotation:
   :param search:
   :param pos: position of the match being possible solutions "b"-begin, "e"-end, "m"-middle
   :return: time series s splited in a list of arrays
   """

   constr, merge_str, str_lst = gt.connotation([s], connotation)
   matches = gt.symbolic_search(constr, merge_str, search)

   segments = []
   match_i = 0
   for match in matches:
        if(pos=="b"):
            segments.append(s[match_i:match[0]])
            match_i = match[0]
        elif(pos=="e"):
            segments.append(s[match_i:match[-1]])
            match_i = match[-1]
        elif(pos=="m"):
            segments.append(s[match_i:match[0]+(match[-1]-match[0])//2])
            match_i = match[0]+(match[-1]-match[0])//2

   return segments

def gt_multi_split(s, connotation, search, pos="b", key=0):
    constr, merge_str, str_lst = gt.connotation([s[key]], connotation)
    matches = gt.symbolic_search(constr, merge_str, search)
    multi_segments = []

    for s_i in s:
        segments = []
        match_i = 0
        for match in matches:
            if (pos == "b"):
                segments.append(s_i[match_i:match[0]])
                match_i = match[0]
            elif (pos == "e"):
                segments.append(s_i[match_i:match[-1]])
                match_i = match[-1]
            elif (pos == "m"):
                segments.append(s_i[match_i:match[0] + (match[-1] - match[0]) // 2])
                match_i = match[0] + (match[-1] - match[0]) // 2
        multi_segments.append(segments)

    return multi_segments

def gt_modify(s, connotation, search, function):
    constr, merge_str, str_lst = gt.connotation([s], connotation)
    matches = gt.symbolic_search(constr, merge_str, search)


    for match in matches:
        s[match[0]:match[1]] = function(s[match[0]:match[1]])

    return s

def gt_label(s, connotation, search, label):
    constr, merge_str, str_lst = gt.connotation([s], connotation)
    matches = gt.symbolic_search(constr, merge_str, search)

    labels = {}

    for i, match in enumerate(matches):
        labels[i] = {"match0":match[0], "match1":match[1], "label":label}

    return labels

def gt_search_span_drawing(s, connotation):
    """
    TODO: find a way to get the span of a signal to search for it...
    """

def gt_search_drawing(s, connotation):
    """
    TODO: ts_drawing is not good because it is based on the X values...therefore it should be resampled to equally spaced data...otherwise, use a click based method....
    :param s:
    :param connotation:
    :return:
    """
    x, y = get_ts_drawing()
    print(x, y)

    plt.plot(y)
    plt.plot(x, y)
    plt.show()

    constr, merged_str, merged_str_lst = gt.connotation([y], connotation)

    search_code = gt.runLengthEncoding(merged_str_lst)

    search_rgx = "+".join(search_code[1]) + "+"

    s_constr, s_merged_str, s_merged_str_lst = gt.connotation([s], connotation)
    print(search_rgx)
    matches = gt.symbolic_search(s_constr, s_merged_str, search_rgx)

    return matches


# def gt_substitute(s, connotation, search, sub):

# def gt_operate(s, connotation, search, operation):

# def gt_feature_extraction(s, connotation, search, feature):
