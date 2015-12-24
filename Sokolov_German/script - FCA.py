cd 'C:\\Users\\Олег\\Dropbox\\Учеба\\Упорядоченные множества в анализе данных\\Домашние задания\\Большое ДЗ\\lazyfca15-master\\lazyfca15-master'

from functools import reduce
from random import shuffle

q=open("contr_data.csv","r")
train = [a.strip().split(",") for a in q][1:]
plus = [a for a in train if a[-1]=="positive"]
minus = [a for a in train if a[-1]=="negative"]
q.close()

cv_res = {
 "positive_positive": 0,
 "positive_negative": 0,
 "negative_positive": 0,
 "negative_negative": 0,
 "contradictory": 0,
}

attrib_names = [
'age <=21',
'age 21-30',
'age >30',
'educ level:1',
'educ level:2',
'educ level:3',
'educ level:4',
'husband`s educ:1',
'husband`s educ:2',
'husband`s educ:3',
'husband`s educ:4',
'num_children:0',
'num_children:1-3',
'num_children:3-5',
'num_children:>5',
'islamic',
'work',
'husband`s occup:1',
'husband`s occup:2',
'husband`s occup:3',
'husband`s occup:4',
'life standard:1',
'life standard:2',
'life standard:3',
'life standard:4',
'media',
'class'
]

def make_intent(example):
    global attrib_names
    return set([i+':'+str(k) for i,k in zip(attrib_names,example)])

# mode=0 - default method
# mode=1 - proposed algorithm
# mode=2 - naive bayes classifier
def check_hypothesis(context_plus, context_minus, example,mode = 2):
    eintent = make_intent(example)
    eintent.discard('class:positive')
    eintent.discard('class:negative')
    labels = {}
    global cv_res

    if mode == 1:
        comm_plus = [make_intent(i)&eintent for i in context_plus]
        comm_minus = [make_intent(i)&eintent for i in context_minus]

        max_comm_plus = max([make_intent(i)&eintent for i in context_plus],key = len)
        max_comm_minus = max([make_intent(i)&eintent for i in context_minus],key = len)
        
        count_max_comm_plus = len ([i for i in comm_plus if len(i) == len(max_comm_plus)])
        count_max_comm_minus = len ([i for i in comm_minus if len(i) == len(max_comm_minus)])
        count_maxmin1_comm_plus = len ([i for i in comm_plus if len(i) == len(max_comm_plus)-1])
        count_maxmin1_comm_minus = len ([i for i in comm_minus if len(i) == len(max_comm_minus)-1])
        count_maxmin2_comm_plus = len ([i for i in comm_plus if len(i) == len(max_comm_plus)-2])
        count_maxmin2_comm_minus = len ([i for i in comm_minus if len(i) == len(max_comm_minus)-2])
        
        if len(max_comm_plus) > len(max_comm_minus): labels['positive'] = True
        elif len(max_comm_plus) < len(max_comm_minus): labels['negative'] = True
        elif len(max_comm_plus) == len(max_comm_minus) and count_max_comm_plus/len(context_plus) > count_max_comm_minus/len(context_minus): labels['positive'] = True
        elif len(max_comm_plus) == len(max_comm_minus) and count_max_comm_plus/len(context_plus) < count_max_comm_minus/len(context_minus): labels['negative'] = True

        elif count_max_comm_plus/len(context_plus) == count_max_comm_minus/len(context_minus) and count_maxmin1_comm_plus/len(context_plus) > count_maxmin1_comm_minus/len(context_minus): labels['positive'] = True
        elif count_max_comm_plus/len(context_plus) == count_max_comm_minus/len(context_minus) and count_maxmin1_comm_plus/len(context_plus) < count_maxmin1_comm_minus/len(context_minus): labels['negative'] = True   

        elif count_maxmin1_comm_plus/len(context_plus) == count_maxmin1_comm_minus/len(context_minus) and count_maxmin2_comm_plus/len(context_plus) > count_maxmin2_comm_minus/len(context_minus): labels['positive'] = True
        elif count_maxmin1_comm_plus/len(context_plus) == count_maxmin1_comm_minus/len(context_minus) and count_maxmin2_comm_plus/len(context_plus) < count_maxmin2_comm_minus/len(context_minus): labels['negative'] = True  
        
        else: labels['contradictory'] = True

        if example[-1] == "positive" and labels.get("positive")==True:
           cv_res["positive_positive"] += 1
        if example[-1] == "negative" and labels.get("positive")==True:
           cv_res["negative_positive"] += 1
        if example[-1] == "positive" and labels.get("negative")==True:
           cv_res["positive_negative"] += 1
        if example[-1] == "negative" and labels.get("negative")==True:
           cv_res["negative_negative"] += 1
        if labels.get("contradictory")==True:
            cv_res["contradictory"] += 1
        
    elif mode == 0:
        for e in context_plus:
            ei = make_intent(e)
            candidate_intent = ei & eintent
            closure = [ make_intent(i) for i in context_minus if make_intent(i).issuperset(candidate_intent)]
            closure_size = len([i for i in closure if len(i)])
            res = reduce(lambda x,y: x&y if x&y else x|y, closure ,set())
            for cs in ['positive','negative']:
                if 'class:'+cs in res:
                    labels[cs+'_total_weight'] = labels.get(cs+'_total_weight',0) + closure_size * 1.0/ len(context_minus) / len(context_plus)
        for e in context_minus:
            ei = make_intent(e)
            candidate_intent = ei & eintent
            closure = [ make_intent(i) for i in context_plus if make_intent(i).issuperset(candidate_intent)]
            closure_size = len([i for i in closure if len(i)])
            res = reduce(lambda x,y: x&y if x&y else x|y, closure, set())
            for cs in ['positive','negative']:
                if 'class:'+cs in res:
                    labels[cs+'_total_weight'] = labels.get(cs+'_total_weight',0) +closure_size * 1.0 / len(context_plus) / len(context_minus)
        
        if example[-1] == "positive" and (labels.get("positive_total_weight")>labels.get("negative_total_weight")):
           cv_res["positive_positive"] += 1
        if example[-1] == "negative" and (labels.get("positive_total_weight")>labels.get("negative_total_weight")):
           cv_res["negative_positive"] += 1
        if example[-1] == "positive" and (labels.get("negative_total_weight")>labels.get("positive_total_weight")):
           cv_res["positive_negative"] += 1
        if example[-1] == "negative" and (labels.get("negative_total_weight")>labels.get("positive_total_weight")):
           cv_res["negative_negative"] += 1

    elif mode == 2:
        prob_plus = len(context_plus)/(len(context_plus)+len(context_minus))
        prob_minus = 1 - prob_plus
        for i in eintent:
                feature = set([i])
                common_plus = [feature & make_intent(x) for x in context_plus] 
                common_minus = [feature & make_intent(y) for y in context_minus]
                num_common_plus = len([z for z in common_plus if len(z)>0])
                num_common_minus = len([p for p in common_minus if len(p)>0])
                prob_plus = prob_plus*num_common_plus/len(context_plus)
                prob_minus = prob_minus*num_common_minus/len(context_minus)

        if prob_plus > prob_minus: labels['positive'] = True
        elif prob_plus < prob_minus: labels['negative'] = True
        else: labels['contradictory'] = True
     
        if example[-1] == "positive" and labels.get("positive")==True:
           cv_res["positive_positive"] += 1
        if example[-1] == "negative" and labels.get("positive")==True:
           cv_res["negative_positive"] += 1
        if example[-1] == "positive" and labels.get("negative")==True:
           cv_res["positive_negative"] += 1
        if example[-1] == "negative" and labels.get("negative")==True:
           cv_res["negative_negative"] += 1
        if labels.get("contradictory")==True:
            cv_res["contradictory"] += 1
        
def metrics(results):
    metr = {}
    acc = div_by_zero(results["positive_positive"] + results["negative_negative"],sum(results.values()))*100
    prec = div_by_zero(results["positive_positive"],results["positive_positive"] + results["negative_positive"])*100
    recall = div_by_zero(results["positive_positive"],results["positive_positive"] + results["positive_negative"])*100
    spec = div_by_zero(results["negative_negative"],results["negative_negative"]+ results["negative_positive"])*100
    NPV = div_by_zero(results["negative_negative"],results["positive_negative"] + results["negative_negative"])*100
    FPR = div_by_zero(results["negative_positive"],results["negative_negative"] + results["negative_positive"])*100
    FDR = div_by_zero(results["negative_positive"],results["positive_negative"] + results["positive_positive"])*100
    metr['Accuracy'] = acc; metr['Precision'] = prec; metr['Recall'] = recall; metr['Specificity'] = spec
    metr['NPV'] = NPV; metr['FPR'] = FPR; metr['FDR'] = FDR
    return metr

def aver_metrics(D):
    aver_metr = {}
    for i in D[0]:
        for x in D:
            aver_metr[i] = (aver_metr.get(i,0)+x[i])
        aver_metr[i] = (aver_metr[i]/len(D))
    return aver_metr

def div_by_zero(x,y):
    if y!=0:
        return x/y
    else:
        return 0

def k_fold_cross_validation(X, K, randomise = False):
	if randomise: X=list(X); shuffle(X)
	for k in range(K):
		training = [x for i, x in enumerate(X) if i % K != k]
		validation = [x for i, x in enumerate(X) if i % K == k]
		yield training, validation, k+1

K = 5
res=[]
split = k_fold_cross_validation(train, K, True)
i = 0
total_res = {
 "positive_positive": 0,
 "positive_negative": 0,
 "negative_positive": 0,
 "negative_negative": 0,
 "contradictory": 0,
}
count = 0
while i < K:
    print('k = ',i+1,' wait...')
    cv_res = {
 "positive_positive": 0,
 "positive_negative": 0,
 "negative_positive": 0,
 "negative_negative": 0,
 "contradictory": 0,
}
    together = next(split)
    training = together[0]
    positive = [a for a in train if a[-1]=="positive"]
    negative = [a for a in train if a[-1]=="negative"]
    testing = together[1]
    
    for elem in testing:
        check_hypothesis(positive, negative, elem, 1)
        count+=1
        print (str(count)+"/"+str(len(train))+" wait...")
    i+=1
    res.append(metrics(cv_res))
    aver_metr = aver_metrics(res)

    total_res["positive_positive"] = total_res.get("positive_positive",0)+cv_res["positive_positive"]
    total_res["positive_negative"] = total_res.get("positive_negative",0)+cv_res["positive_negative"]
    total_res["negative_positive"] = total_res.get("negative_positive",0)+cv_res["negative_positive"]
    total_res["negative_negative"] = total_res.get("negative_negative",0)+cv_res["negative_negative"]
    
print (aver_metr)
print (total_res)
