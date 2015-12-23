from __future__ import division
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import tree

data = np.genfromtxt('2008_100_200.csv', delimiter=";", skip_header=1)
print(data.shape)

#removing nan
data = data[~np.isnan(data).any(axis=1)]
#or this way
#data = data[np.all(np.isfinite(data), axis=1)]

#remove 0 class
data = data[np.logical_not(data[:, 16] == 0)]

#check for deletions
print(data.shape)

#check if nan detected
#print(np.isnan(X).any())

#create model
clf = tree.DecisionTreeClassifier(max_features=3, max_depth=5, random_state=0)
data_train, data_test = train_test_split(data, test_size=0.5, random_state=0)
target = [x[16] for x in data_train]
train = data_train[:, 0:16]
fitted_tree = clf.fit(train, target)
test = data_test[:, 0:16]

predicted_probs = clf.predict_proba(test)
predicted_probs = predicted_probs.astype(int)
test_to_compare = np.array(data_test[:, 16])
print(test_to_compare.shape)
#saving the result of classifying to csv
compare = np.column_stack((predicted_probs, test_to_compare))
print(compare[:100, :])

np.savetxt('filepath.csv', compare, delimiter=',', header='Pred1,Pred2,Pred3,class', comments='')

TP = 0
TN = 0
FP = 0
FN = 0
#class 100 is +, 200 is -
for row in compare:
    if (row[0] == 1) & (row[2] == 100):
        TP += 1
    if (row[1] == 1) & (row[2] == 200):
        TN += 1
    if (row[1] == 1) & (row[2] == 100):
        FN += 1
    if (row[0] == 1) & (row[2] == 200):
        FP += 1

TPR_recall = TP/(TP+FN)
TNR = TN/(FP+TN)
NPV = TN/(TN+FN)
FPR = FP/(FP+TN)
FDR_precision = FP/(FP+TP)
Accuracy = (TP+TN)/(TP+TN+FP+FN)
summary = {"TP": TP, "TN": TN, "FP": FP, "FN": FN,
           "TPR_recall": TPR_recall, "TNR": TNR, "NPV": NPV, "FPR": FPR,
           "FDR_precision": FDR_precision, "Accuracy": Accuracy}
print(summary)

data = np.delete(data, [0, 3, 4, 10, 11, 14], 1)