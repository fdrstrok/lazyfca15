import sys
import random
from itertools import count, takewhile


def frange(start, stop, step):
        return takewhile(lambda x: x< stop, count(start, step))

# attrib_names = [ 'class','a1','a2','a3','a4','a5','a6' ]
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
'''
attrib_names = []
for i in range(0,137):
    attrib_names.append(str(i))
'''
def make_intent(example):
    global attrib_names
    return set([i+':'+str(k) for i, k in zip(attrib_names, example)])


cv_res = {
    "positive_positive": 0,
    "positive_negative": 0,
    "negative_positive": 0,
    "negative_negative": 0,
    "contradictory": 0,
    "accuracy":0,
    "total": 0,
}

def check_intersect(context_plus, context_minus, example,threshold, num_sub=1):
    global cv_res, cv_res2
    pos = 0
    neg = 0
    intent = make_intent(example)
    for i in xrange(num_sub):
        t = set(random.sample(example, random.randrange(len(intent))))
        for j in context_plus:
            if t.issubset(j):
                pos += len(t)
        for k in context_minus:
            if t.issubset(k):
                neg += len(t)

    def score(pos, neg):
        return pos * 1. / (neg + 1)
    #max_acc = 0
    #for threshold in frange(1.1, 3.0, 0.5):
    #threshold = 1.1
    if score(pos, neg) > threshold:
        if example[0] == 'positive':
            cv_res['positive_positive'] += 1
        else:
            cv_res['negative_positive'] += 1
    elif score(neg, pos) > threshold:
        if example[0] == 'positive':
            cv_res['positive_negative'] += 1
        else:
            cv_res['negative_negative'] += 1
    else:
        cv_res['contradictory'] += 1
    cv_res['accuracy'] = (cv_res['positive_positive'] + cv_res['negative_negative'])

def calc(threshold):
    max_index  = 10
    for index in xrange(1, int(max_index)):
        index = str(index)
        q = open("train" + index + ".csv", "r")
        train = [a.strip().split(",") for a in q]
        plus = [a for a in train if a[0] == "positive"]
        minus = [a for a in train if a[0] == "negative"]

        # print t
        q.close()
        w = open("test" + index + ".csv", "r")
        unknown = [a.strip().split(",") for a in w]
        w.close()

        for elem in unknown:
            cv_res['total'] += 1
            #   print elem
            # print "done"
            # check_hypothesis(plus, minus, elem)
            # check_intersect(plus, minus, elem, len(elem) / 2)
            check_intersect(plus, minus, elem, 1.1 , len(elem) / 2)
        print "done: %s" % index

    

    #print cv_res metrics
    cv_res['accuracy'] = (cv_res['positive_positive'] + cv_res['negative_negative'])*1. / cv_res["total"]
    #print cv_res
    for k, v in cv_res.iteritems():
        #print cv_res[k], v
        cv_res[k] = v * 1. / cv_res["total"]
    
    

        
    print cv_res
    print cv_res['accuracy'], threshold
    return cv_res  #['accuracy'] 
    

####################################### 
#print ClassifierNaiveBayes()

print 'max accuracy is: ' + str(max(map(calc, [threshold for threshold in frange(1.1, 3.0, 0.5) ] )))

libs = map(calc, [threshold for threshold in frange(1.1, 3.0, 0.5) ] )

print libs




