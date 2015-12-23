import pprint
import sys


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
    
def check_hypothesis(context_plus, context_minus, example):
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
  
    #if abs(labels['pos_total_weight']-0.313)<abs(labels['pos_total_weight']-0.192)and abs(labels['neg_total_weight']-0.158)<abs(labels['neg_total_weight']-0.276):
    #    labels['positive']=True
    #if abs(labels['pos_total_weight']-0.313)>=abs(labels['pos_total_weight']-0.192)and abs(labels['neg_total_weight']-0.158)>=abs(labels['neg_total_weight']-0.276):
    #    labels['negative']=True
    if labels['pos_total_weight']>0.27:
        labels['positive']=True
    if labels['neg_total_weight']>0.23:
        labels['negative']=True
    
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
    list_pos_pos = []
    list_neg_neg = []
    list_pos_neg = []
    list_neg_pos = []
    for elem in unknown:
        #print "done"
        i += 1
        check_hypothesis(plus, minus, elem)
        #if i == 100: break
    print 1.0*(cv_res["positive_positive"]+cv_res["negative_negative"])/len(unknown)
