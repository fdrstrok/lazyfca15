import pprint
import sys

index = sys.argv[1]

q=open("train"+index+".csv","r")
train = [ a.strip().split(",") for a in q]
plus = [a for a in train if a[-1]=="positive"]
minus = [a for a in train if a[-1]=="negative"]

#print t
q.close()
w=open("test"+index+".csv","r")
unknown = [a.strip().split(",") for a in w]
w.close()


cv_res = {
 "positive_positive": 0,
 "positive_negative": 0,
 "negative_positive": 0,
 "negative_negative": 0,
 "contradictory": 0,
}

#attrib_names = [ 'class','a1','a2','a3','a4','a5','a6' ]
attrib_names = [
'top-left-square',
'top-middle-square',
' top-right-square',
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
    return set([i+':'+str(k) for i,k in zip(attrib_names,example)])
    
def check_hypothesis(context_plus, context_minus, example):
  #  print example
    eintent = make_intent(example)
  #  print eintent
    eintent.discard('class:positive')
    eintent.discard('class:negative')
    labels = {}
    global cv_res
    for e in context_plus:
        ei = make_intent(e)
        candidate_intent = ei & eintent
        closure = [ make_intent(i) for i in context_minus if make_intent(i).issuperset(candidate_intent)]
        closure_size = len([i for i in closure if len(i)])
    #    print closure
        #print closure_size * 1.0 / len(context_minus)
        res = reduce(lambda x,y: x&y if x&y else x|y, closure ,set())
        for cs in ['positive','negative']:
            if 'class:'+cs in res:
                labels[cs] = True
                labels[cs+'_res'] = candidate_intent
                labels[cs+'_total_weight'] = labels.get(cs+'_total_weight',0) +closure_size * 1.0 / len(context_minus) / len(context_plus)
    for e in context_minus:
        ei = make_intent(e)
        candidate_intent = ei & eintent
        closure = [ make_intent(i) for i in context_plus if make_intent(i).issuperset(candidate_intent)]
        closure_size = len([i for i in closure if len(i)])
        #print closure_size * 1.0 / len(context_plus)
        res = reduce(lambda x,y: x&y if x&y else x|y, closure, set())
        for cs in ['positive','negative']:
            if 'class:'+cs in res:
                labels[cs] = True
                labels[cs+'_res'] = candidate_intent
                labels[cs+'_total_weight'] = labels.get(cs+'_total_weight',0) +closure_size * 1.0 / len(context_plus) / len(context_minus)
    print eintent
    print labels
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

#sanity check:
#check_hypothesis(plus_examples, minus_examples, plus_examples[3])
i = 0
for elem in unknown:
    #print elem
    print "done"
    i += 1
    check_hypothesis(plus, minus, elem)
#    if i == 3: break
print cv_res
