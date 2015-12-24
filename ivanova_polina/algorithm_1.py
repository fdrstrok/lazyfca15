import pprint
import sys
from numpy import *
from functools import reduce

index = sys.argv[1]

q = open("train"+index+".csv", "r")
train = [a.strip().split(",") for a in q]
plus = [a for a in train if a[-1] == "positive"]
minus = [a for a in train if a[-1] == "negative"]
q.close()

w = open("test"+index+".csv", "r")
unknown = [a.strip().split(",") for a in w]
w.close()

plus = [a for a in train if a[-1] == "positive"]
minus = [a for a in train if a[-1] == "negative"]


cv_res = {
 "positive_positive": 0,
 "positive_negative": 0,
 "negative_positive": 0,
 "negative_negative": 0,
 "contradictory": 0,
}

attrib_names = [
 'top-left',
 'top-middle',
 'top-right',
 'middle-left',
 'middle-middle',
 'middle-right',
 'bottom-left',
 'bottom-middle',
 'bottom-right',
 'class'
]


def make_intent(example):
    global attrib_names
    return set([i+':'+str(k) for i, k in zip(attrib_names, example)])

def check_hypothesis(context_plus, context_minus, example, threshold, cv_res):
    eintent = make_intent(example)
    eintent.discard('class:positive')
    eintent.discard('class:negative')
    labels = {}
    #global cv_res
    for e in context_plus:
        ei = make_intent(e)
        candidate_intent = ei & eintent
        support = [make_intent(i) for i in context_plus if make_intent(i).issuperset(candidate_intent)]
        support_size = len([i for i in support if len(i)])
        closure = [ make_intent(i) for i in context_minus if make_intent(i).issuperset(candidate_intent)]
        closure_size = len([i for i in closure if len(i)])
        res = reduce(lambda x,y: x&y if x&y else x|y, support, set())
        for cs in ['positive','negative']:
            if 'class:'+cs in res:
                labels[cs] = True
                labels[cs+'_support_avrg'] = labels.get(cs+'_support_avrg',0) + support_size * 1.0 / len(context_plus)
                labels[cs+'_false_avrg'] = labels.get(cs+'_false_avrg',0) + closure_size * 1.0 / len(context_minus)
    for e in context_minus:
        ei = make_intent(e)
        support = [make_intent(i) for i in context_minus if make_intent(i).issuperset(candidate_intent)]
        support_size = len([i for i in support if len(i)])
        candidate_intent = ei & eintent
        closure = [ make_intent(i) for i in context_plus if make_intent(i).issuperset(candidate_intent)]
        closure_size = len([i for i in closure if len(i)])
        res = reduce(lambda x,y: x&y if x&y else x|y, support, set())
        for cs in ['positive','negative']:
            if 'class:'+cs in res:
                labels[cs] = True
                labels[cs+'_support_avrg'] = labels.get(cs+'_support_avrg',0) + support_size * 1.0 / len(context_minus)
                labels[cs+'_false_avrg'] = labels.get(cs+'_false_avrg',0) + closure_size * 1.0 / len(context_plus)
    labels['positive_support_avrg'] = labels.get('positive_support_avrg',0) / len(context_plus)
    labels['positive_false_avrg'] = labels.get('positive_false_avrg',0) / len(context_plus)
    labels['negative_support_avrg'] = labels.get('negative_support_avrg',0) / len(context_minus)
    labels['negative_false_avrg'] = labels.get('negative_false_avrg',0) / len(context_minus)
    labels["positive"] = labels['positive_support_avrg'] > threshold
    labels["negative"] = labels['negative_support_avrg'] > threshold
    #print(eintent)
    print("positive: ", labels['positive'], "negative: ", labels['negative'], "pos_supp: ", labels['positive_support_avrg'], 'pos_false: ',labels['positive_false_avrg'], "neg_supp: ", labels['negative_support_avrg'], 'neg_false: ',labels['negative_false_avrg'])
    if labels.get("positive",False) and labels.get("negative",False):
       cv_res["contradictory"] += 1
       return
    if example[-1] == "positive" and labels.get("positive",False):
       cv_res["positive_positive"] += 1
    if example[-1] == "negative" and labels.get("positive",False):
       cv_res["negative_positive"] += 1
    if example[-1] == "positive" and labels.get("negative",False):
       cv_res["positive_negative"] += 1
    if example[-1] == "negative" and labels.get("negative",False):
       cv_res["negative_negative"] += 1
    return cv_res


#threshold = 0.125
#sanity check:
#check_hypothesis(plus_examples, minus_examples, plus_examples[3])
#
# unknown = unknown[1:]
# i = 0
# for elem in unknown:
#     #print elem
#     print("done ", i)
#     i += 1
#     check_hypothesis(plus, minus, elem, threshold)
# #    if i == 3: break
# print(cv_res)
