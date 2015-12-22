from sklearn.cross_validation import KFold
from data_preparation import *
from estimation import *
# Aggr = Sum of supports for each sample (calculated as in the task doc)


def classify(train, test):
    train300, train100 = split_the_data_into_classes(train)
    cv_res = {
     "PP": 0,
     "PN": 0,
     "NP": 0,
     "NN": 0,
     "contradictory": 0,
    }
    l = len(test)
    for elem in range(0, l):
        result = check_hypothesis(train300, train100, test.iloc[elem])
        cv_res[result] += 1
    return cv_res


def k_fold(data, n):
    index_length = len(data.index)
    kf = KFold(index_length, n_folds=n)
    return kf


def check_hypothesis(context300, context100, example):
    example_intent = dataframe_to_string(example)
    labels = {"300": 0, "100": 0}
    for i in range(0, len(context300)):
        context300_intent = dataframe_to_string(context300.iloc[i])
        intersection_intent = cross(context300_intent[0], example_intent[0])
        anti_support = check_involvement(intersection_intent, context100)
        labels["300"] += float(len(anti_support)) / len(context100)
    for i in range(0, len(context100)):
        context100_intent = dataframe_to_string(context100.iloc[i])
        intersection_intent = cross(context100_intent[0], example_intent[0])
        anti_support = check_involvement(intersection_intent, context300)
        labels["100"] += float(len(anti_support)) / len(context300)
    labels["300"] = float(labels["300"]) / len(context300)
    labels["100"] = float(labels["100"]) / len(context100)
    if labels["300"] > labels["100"]:
        if example_intent[-1] == "300":
            return "PP"
        else:
            return "NP"
    elif labels["300"] < labels["100"]:
        if example_intent[-1] == "100":
            return "NN"
        else:
            return "PN"
    else:
        return "contradictory"


def anti_supp_fca(data, n):
    total = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
    kf = k_fold(data, n)
    for test, train in kf:
        # print("%s %s" % (train, test))
        kf_train = data.iloc[train]
        kf_test = data.iloc[test]
        res = classify(kf_train, kf_test)
        stat = summary(res)
        total["accuracy"] += stat["accuracy"]
        total["precision"] += stat["precision"]
        total["recall"] += stat["recall"]
        total["f1"] += stat["f1"]
    total["accuracy"] = total["accuracy"]/n
    total["precision"] = total["precision"]/n
    total["recall"] = total["recall"]/n
    total["f1"] = total["f1"]/n
    print(total)


def check_involvement(example, current_contex):
    decisions = []
    for lo in range(0, len(current_contex)):
        new_intent = dataframe_to_string(current_contex.iloc[lo])
        if cross(new_intent[0], example) != example:
            decisions.append(True)
    return decisions
