import pprint
import sys
import functools
import pandas as pd
import sklearn
import sklearn.ensemble as sk_en
import sklearn.metrics as sk_m
import sklearn.cross_validation as sk_cv
from sklearn.metrics import confusion_matrix

#index = sys.argv[1]
global cv_res
global threshold
threshold=0
max_cv_res = {
    "positive_positive": 0,
    "positive_negative": 0,
    "negative_positive": 0,
    "negative_negative": 0,
    "contradictory": 0,
    "total": 0,
    "threshold": 0,
    "max_index":0,
    "max_sum":0,
    "max_accuracy":0,
}


q1 = open("tictactoe.csv", "r")
threshold=0.7
index = 10;
i = 0
l_arr=[]
for i in range(0,20):
    max_index=11
    threshold=threshold+0.1
    for k in range(0,10):
        exec(compile(open("test_implication_changed.py", "rb").read(), "test_implication_changed.py", 'exec'), globals(), locals())
        max_cv_res=cv_res
        max_cv_res["threshold"]=threshold
        max_cv_res["max_sum"]=cv_res["positive_positive"] + cv_res["negative_negative"]
        max_cv_res["max_accuracy"]=max_cv_res["max_sum"]/max_cv_res["total"]
        l_arr.append(max_cv_res)
max_accuracy=0
#sorted(l_arr,key=itemgetter(0))
best_accuracy_index =0
for i in range(0, 200):
    if(max_accuracy<=l_arr[i]["max_accuracy"]):
        max_accuracy=l_arr[i]["max_accuracy"]
        best_accuracy_index=i;
    print(l_arr[i])
TP=l_arr[best_accuracy_index]["positive_positive"]
TN=l_arr[best_accuracy_index]["negative_negative"]
FP=l_arr[best_accuracy_index]["negative_positive"]
FN=l_arr[best_accuracy_index]["positive_negative"]
precision=TP/(FP+TP)
recall = TP/(TP+FN)
F1 = 2*precision*recall/(precision+recall)
print('True Positive :\t', TP)
print('True Negative :\t', TN)
print('False Positive :\t', FP)
print('False Negative :\t', FN)
print('True Positive Rate(TPR) :\t', TP/(TP+FN))
print('True Negative Rate(TNR) :\t', TN/(TN+FP))
print('Negative Predictive Value(NPV) :\t', TN/(TN+FN))
print('False Positive Rate(FPR :\t', FP/(FP+TN))
print('FDR :\t', FP/(FP+TP))
print('Accuracy :\t', (TP+TN)/(TP+TN+FP+FN))
print('PPV(precision) :\t', TP/(FP+TP))
print('F1 :\t', F1)
print('Recall :\t', TP/(TP+FN))
#print('Max accuracy  :\t', max_accuracy, '\n')

l_arr=[]
for k, v in cv_res.items():
        max_cv_res[k]=0
threshold=l_arr[best_accuracy_index]["threshold"]
for i in range(0,10):
    max_index=i+1
    for k in range(0,10):
        exec(compile(open("test_implication_changed.py", "rb").read(), "test_implication_changed.py", 'exec'), globals(), locals())
        max_cv_res=cv_res
        max_cv_res["threshold"]=threshold
        max_cv_res["max_sum"]=cv_res["positive_positive"] + cv_res["negative_negative"]
        max_cv_res["max_accuracy"]=max_cv_res["max_sum"]/max_cv_res["total"]
        max_cv_res["max_index"] = max_index
        l_arr.append(max_cv_res)

max_accuracy=0
for i in range(0, 100):
    if(max_accuracy<=l_arr[i]["max_accuracy"]):
        max_accuracy=l_arr[i]["max_accuracy"]
        best_accuracy_index=i
    print(l_arr[i])
TP=l_arr[best_accuracy_index]["positive_positive"]
TN=l_arr[best_accuracy_index]["negative_negative"]
FP=l_arr[best_accuracy_index]["negative_positive"]
FN=l_arr[best_accuracy_index]["positive_negative"]
precision=TP/(FP+TP)
recall = TP/(TP+FN)
F1 = 2*precision*recall/(precision+recall)
print('True Positive :\t', TP)
print('True Negative :\t', TN)
print('False Positive :\t', FP)
print('False Negative :\t', FN)
print('True Positive Rate(TPR) :\t', TP/(TP+FN))
print('True Negative Rate(TNR) :\t', TN/(TN+FP))
print('Negative Predictive Value(NPV) :\t', TN/(TN+FN))
print('False Positive Rate(FPR :\t', FP/(FP+TN))
print('FDR :\t', FP/(FP+TP))
print('Accuracy :\t', (TP+TN)/(TP+TN+FP+FN))
print('PPV(precision) :\t', TP/(FP+TP))
print('F1 :\t', F1)
print('Recall :\t', TP/(TP+FN))
#print('Max accuracy  :\t', max_accuracy, '\n')



data = pd.read_csv('tic-tac-toe.data', header=None)
data.ix[:, data.shape[1]-1] = data.ix[:, data.shape[1]-1].map(lambda x: 1 if x == 'positive' else 0)
data = pd.get_dummies(data)
X = data.ix[:, 1:].values
y = data.ix[:, 0].values
for train_index, test_index in sk_cv.StratifiedShuffleSplit(y, n_iter=100, test_size=0.33):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
clf = sk_en.RandomForestClassifier(n_estimators=200, max_depth=15)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
#print(sk_m.recall_score(y_true, y_pred)
#print(sk_m.precision_score(y_test, y_pred))
#print(sk_m.f1_score(y_test, y_pred))
#print(sk_m.accuracy_score(y_true, y_pred))
arr=sk_m.confusion_matrix(y_test, y_pred)
TP=arr[0][0]
FN=arr[0][1]
TN=arr[1][1]
FP=arr[1][0]
precision=TP/(FP+TP)
recall = TP/(TP+FN)
F1 = 2*precision*recall/(precision+recall)

print('True Positive :\t', TP)
print('True Negative :\t', TN)
print('False Positive :\t', FP)
print('False Negative :\t', FN)
print('True Positive Rate(TPR) :\t', TP/(TP+FN))
print('True Negative Rate(TNR) :\t', TN/(TN+FP))
print('Negative Predictive Value(NPV) :\t', TN/(TN+FN))
print('False Positive Rate(FPR :\t', FP/(FP+TN))
print('FDR :\t', FP/(FP+TP))
print('Accuracy :\t', (TP+TN)/(TP+TN+FP+FN))
print('PPV(precision) :\t', TP/(FP+TP))
print('F1 :\t', F1)
print('Recall :\t', TP/(TP+FN))

stratified_kfold = sk_cv.StratifiedKFold(y, n_folds=15, shuffle=True),
clf = sklearn.ensemble.RandomForestClassifier(n_estimators=200, max_depth=15)
scores = sklearn.cross_validation.cross_val_score(clf, X, y, scoring='accuracy', cv=stratified_kfold)
print('Accuracy score=', scores.mean())

#for k, v in cv_res.items():
 #   cv_res[k] = v * 1. / cv_res["total"]


