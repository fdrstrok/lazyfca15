import sys
from numpy import *
from sklearn import cross_validation
from algorithm_2 import check_hypothesis as check_hypothesis_2, make_intent, attrib_names
from algorithm_1 import check_hypothesis as check_hypothesis_1

index = sys.argv[1]

q = open("train"+index+".csv", "r")
train = [a.strip().split(",") for a in q]
plus = [a for a in train if a[-1] == "positive"]
minus = [a for a in train if a[-1] == "negative"]
q.close()

train = train[1:]
#train = train[530:580]

def cv_f(threshold, check_function):
    global train
    nf = 5
    kf = cross_validation.KFold(len(train), n_folds=nf, shuffle=True, random_state=0)

    accuracy = 0
    i = 0
    for train_index, test_index in kf:
        if i>2:
            return accuracy/3
        print(i)
        i += 1
        X_train, X_test = list( train[i] for i in train_index ), list( train[i] for i in test_index )
        plus = [a for a in X_train if a[-1] == "positive"]
        minus = [a for a in X_train if a[-1] == "negative"]
        cv_res = { "positive_positive": 0, "positive_negative": 0, "negative_positive": 0, "negative_negative": 0, "contradictory": 0}
        print(test_index)
        for elem in X_test:
            cv_res_temp = check_function(plus, minus, elem, threshold, cv_res)
            if cv_res_temp is not None:
                cv_res = cv_res_temp
            print(cv_res)
        cvv = [s for s in cv_res.values()]
        print( sum(cvv) )
        print((cv_res["positive_positive"] + cv_res["negative_negative"]) / len(X_test))
        accuracy += (cv_res["positive_positive"] + cv_res["negative_negative"]) / len(X_test) #/ sum(cvv)
    accuracy /= nf
    return accuracy

# For algorithm_1.py
# threshold = 0.127
# accuracy = cv_f(threshold, check_hypothesis_1)
# print(accuracy)

# For algorithm_2.py
threshold = 0.6
accuracy = cv_f(threshold, check_hypothesis_2)
print(accuracy)

