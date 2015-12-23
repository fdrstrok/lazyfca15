import pprint
import sys
import numpy

cv_res = {
 "positive_positive": 0,
 "positive_negative": 0,
 "negative_positive": 0,
 "negative_negative": 0,
 "contradictory": 0,
}

attrib_names = [
'buying',
'maint',
'doors',
'persons',
'lug_boot',
'safety',
'class'
]


def make_intent(example):
    global attrib_names
    return set([i+':'+str(k) for i,k in zip(attrib_names,example)])
    
def check_hypothesis(context_plus, context_minus, example, list_pos_pos, list_neg_neg, list_pos_neg, list_neg_pos):
    #print example
    eintent = make_intent(example)
    #print eintent
    eintent.discard('class:positive')
    eintent.discard('class:negative')
    labels = {}
    global cv_res
    for e in context_plus:
        ei = make_intent(e)
        candidate_intent = ei & eintent
        closure = [ make_intent(i) for i in context_minus if make_intent(i).issuperset(candidate_intent)]
        closure_size = len([i for i in closure if len(i)])
        labels['neg_total_weight'] = labels.get('neg_total_weight',0) +closure_size * 1.0  / len(context_plus) / len(context_minus)
    for e in context_minus:
        ei = make_intent(e)
        candidate_intent = ei & eintent
        closure = [ make_intent(i) for i in context_plus if make_intent(i).issuperset(candidate_intent)]
        closure_size = len([i for i in closure if len(i)])
        labels['pos_total_weight'] = labels.get('pos_total_weight',0) +closure_size * 1.0  / len(context_minus) / len(context_plus)
    print numpy.mean(list_pos_pos)
    if abs(labels['pos_total_weight']-numpy.mean(list_pos_pos))<abs(labels['pos_total_weight']-numpy.mean(list_neg_pos))and abs(labels['neg_total_weight']-numpy.mean(list_pos_neg))<abs(labels['neg_total_weight']-numpy.mean(list_neg_neg)):
        labels['positive']=True
        #ans = raw_input('positive?:'+str(example))
        if example[-1]=='negative':
            minus.append(example)
    if abs(labels['pos_total_weight']-numpy.mean(list_pos_pos))>=abs(labels['pos_total_weight']-numpy.mean(list_neg_pos))and abs(labels['neg_total_weight']-numpy.mean(list_pos_neg))>=abs(labels['neg_total_weight']-numpy.mean(list_neg_neg)):
        labels['negative']=True
        #ans = raw_input('negative?:'+str(example))
        if example[-1]=='positive':
            plus.append(example)
    
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
    
    pos = labels['pos_total_weight']
    neg = labels['neg_total_weight']
    
    if example[-1]=='positive':
        list_pos_pos.append(pos)
        list_pos_neg.append(neg)
    else:
        list_neg_pos.append(pos)
        list_neg_neg.append(neg)

max_index = 10
for index in range(0, max_index):
    index = str(index)
    q = open("car.data_train_" + index + ".txt", "r")
    train = [a.strip().split(",") for a in q]
    plus = [a for a in train if a[-1] == "positive"]
    minus = [a for a in train if a[-1] == "negative"]
    q.close()
    w = open("car.data_validation_" + index + ".txt", "r")
    unknown = [a.strip().split(",") for a in w]
    w.close()
    i = 0
    list_pos_pos = [0.313]
    list_neg_neg = [0.276]
    list_pos_neg = [0.158]
    list_neg_pos = [0.192]
    
    for elem in unknown:
        i += 1
        check_hypothesis(plus, minus, elem, list_pos_pos, list_neg_neg, list_pos_neg, list_neg_pos)
        
        #if i == 100: break
    print 1.0*(cv_res["positive_positive"]+cv_res["negative_negative"])/len(unknown)
