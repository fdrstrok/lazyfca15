C = 0.7
ds_num = 10


q = open("train"+str(ds_num)+".csv", "r")
train = [a.strip().split(",") for a in q]
plus = [a for a in train if a[-1] == "positive"]
minus = [a for a in train if a[-1] == "negative"]
q.close()
w = open("test"+str(ds_num)+".csv", "r")
unknown = [a.strip().split(",") for a in w]
del unknown[0]
w.close()


cv_res = {
    "true_positive": 0,
    "false_negative": 0,
    "false_positive": 0,
    "true_negative": 0,
    "contradictory": 0,
}


attrib_names = [
    'top-left-square',
    'top-middle-square',
    'top-right-square',
    'middle-left-square',
    'middle-middle-square',
    'middle-right-square',
    'bottom-left-square',
    'bottom-middle-square',
    'bottom-right-square',
    'class'
]


def make_intent(example):
    global attrib_names
    return set([i+':'+str(k) for i, k in zip(attrib_names, example)])


def check_hypothesis(context_plus, context_minus, example):
    eintent = make_intent(example)
    eintent.discard('class:positive')
    eintent.discard('class:negative')
    labels = {"positive": 0, "negative": 0}
    global cv_res
    for e in context_plus:
        ei = make_intent(e)
        candidate_intent = ei & eintent
        if len(candidate_intent) > C*len(eintent):
            labels["positive"] += 1
    for e in context_minus:
        ei = make_intent(e)
        candidate_intent = ei & eintent
        if len(candidate_intent) > C*len(eintent):
            labels["negative"] += 1
    labels["positive"] = float(labels["positive"]) / len(context_plus)
    labels["negative"] = float(labels["negative"]) / len(context_minus)
    if labels["positive"] > labels["negative"]:
        if example[-1] == "positive":
            cv_res["true_positive"] += 1
            print "true_positive"
            return
        else:
            cv_res["false_positive"] += 1
            print "false_positive"
            return
    elif labels["positive"] < labels["negative"]:
        if example[-1] == "negative":
            cv_res["true_negative"] += 1
            print "true_negative"
            return
        else:
            cv_res["false_negative"] += 1
            print "false_negative"
            return
    else:
        cv_res["contradictory"] += 1
        print "contradictory"
        return


for elem in unknown:
    check_hypothesis(plus, minus, elem)
print cv_res
accuracy = float(cv_res["true_positive"] + cv_res["true_negative"]) / float(cv_res["true_positive"] +
        cv_res["true_negative"] + cv_res["false_negative"] + cv_res["false_positive"] + cv_res["contradictory"])
precision = float(cv_res["true_positive"]) / float(cv_res["true_positive"] + cv_res["false_positive"])
recall = float(cv_res["true_positive"]) / float(cv_res["true_positive"] + cv_res["false_negative"])
F_measure = 2 / (1/precision + 1/recall)
print "accuracy: " + str(accuracy)
print "precision: " + str(precision)
print "recall: " + str(recall)
print "F_measure: " + str(F_measure)

