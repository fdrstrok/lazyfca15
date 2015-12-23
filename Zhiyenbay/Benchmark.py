import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.cross_validation
import sklearn.ensemble
import sklearn.grid_search


columns = ['top-left-square', 'top-middle-square', 'top-right-square', 'middle-left-square', 'middle-middle-square', 'middle-right-square', 'bottom-left-square', 'bottom-middle-square', 'bottom-right-square', 'Class']
data = pd.read_csv('tic-tac-toe.data', header=None)
data.columns = columns
data.Class = data.Class.map(lambda x: 1 if x == 'positive' else 0)
data.to_csv('tic-tac-toe.csv', index=False)
data = pd.get_dummies(data)
X = data.ix[:, 1:].values
y = data.ix[:, 0].values
for train_index, test_index in sklearn.cross_validation.StratifiedShuffleSplit(y, n_iter=1, test_size=0.3):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    

class MyClassifier():
    def __init__(self, min_supp=0.9, min_similarity=0.7):
        self.min_supp = min_supp
        self.min_similarity = min_similarity
        
    def __predict_one(self, x):
        pos_intersection = (self.pos*x)
        pos_conf = (pos_intersection == x).sum(axis=1)/x.shape[0]
        neg_intersection = (self.neg*x)
        neg_conf = (neg_intersection == x).sum(axis=1)/x.shape[0]

        pos_intersection = pos_intersection[pos_conf >= min(self.min_similarity, pos_conf.max())]
        neg_intersection = neg_intersection[neg_conf >= min(self.min_similarity, neg_conf.max())]

        pos_dash = (pos_intersection.dot(pos_intersection.T) ==
                    pos_intersection.sum(axis=1).reshape(pos_intersection.shape[0], 1))
        pos_dash = pos_dash.sum(axis=1)/self.pos.shape[0]
        pos_dash = pos_dash[pos_dash >= min(self.min_supp, pos_dash.max())]

        neg_dash = (neg_intersection.dot(neg_intersection.T) ==
                    neg_intersection.sum(axis=1).reshape(neg_intersection.shape[0], 1))
        neg_dash = neg_dash.sum(axis=1)/self.neg.shape[0]
        neg_dash = neg_dash[neg_dash >= min(self.min_supp, neg_dash.max())]
        
        pos_coeff = pos_dash.mean()
        neg_coeff = neg_dash.mean()
        
        c = 1/(pos_coeff + neg_coeff)
        return [c*neg_coeff, c*pos_coeff]
        
    def fit(self, X, y):
        self.pos, self.neg = X[y == 1], X[y == 0]
        return self
    
    def predict(self, X):
        return [np.argmax(l) for l in self.predict_proba(X)]
    
    def predict_proba(self, X):
        return np.array([self.__predict_one(x) for x in X])
    
    def get_params(self, deep=True):
        return {'min_supp': self.min_supp, 'min_similarity': self.min_similarity}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


skf = sklearn.cross_validation.StratifiedKFold(y_train, n_folds=10, shuffle=True)
clf = MyClassifier()
gs = sklearn.grid_search.RandomizedSearchCV(clf, {'min_supp': np.linspace(0, 1, 20), 
                                                  'min_similarity': np.linspace(0.7, 1, 20)},
                                            scoring='accuracy', n_jobs=-1, n_iter=100, cv=skf, error_score=0)
gs.fit(X_train, y_train)
gs.best_params_, gs.best_score_


def TP(y_true, y_pred):
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    return confusion_matrix[0,0]

def TN(y_true, y_pred):
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    return confusion_matrix[1,1]

def FP(y_true, y_pred):
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    return confusion_matrix[1,0]

def FN(y_true, y_pred):
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    return confusion_matrix[0,1]

def TPR(y_true, y_pred):
    return TP(y_true, y_pred)/(TP(y_true, y_pred) + FN(y_true, y_pred))

def TNR(y_true, y_pred):
    return TN(y_true, y_pred)/(TN(y_true, y_pred) + FP(y_true, y_pred))

def NPV(y_true, y_pred):
    return TN(y_true, y_pred)/(TN(y_true, y_pred) + FN(y_true, y_pred))

def FPR(y_true, y_pred):
    return FP(y_true, y_pred)/(FP(y_true, y_pred) + TN(y_true, y_pred))

def FDR(y_true, y_pred):
    return FP(y_true, y_pred)/(FP(y_true, y_pred) + TP(y_true, y_pred))

def TNR(y_true, y_pred):
    return TN(y_true, y_pred)/(TN(y_true, y_pred) + FP(y_true, y_pred))

metrics = [TP, TN, FP, FN, TPR, TNR, NPV, FPR, FDR, TNR,
           sklearn.metrics.accuracy_score, sklearn.metrics.precision_score,
           sklearn.metrics.recall_score, sklearn.metrics.roc_auc_score, sklearn.metrics.f1_score]
metrics_names = [func.__name__ for func in metrics]

clf = gs.best_estimator_
# clf = MyClassifier(min_similarity=0.9, min_supp=0.4)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

for metric_name, metric in zip(metrics_names, metrics):
    score = metric(y_test, y_pred)
    print(metric_name, '=', score)

skf = sklearn.cross_validation.StratifiedKFold(y, n_folds=10, shuffle=True)
scores = sklearn.cross_validation.cross_val_score(clf, X, y, scoring='accuracy', n_jobs=-1, cv=skf)
print('Accuracy', scores.mean())

skf = sklearn.cross_validation.StratifiedKFold(y_train, n_folds=10, shuffle=True)
clf = sklearn.ensemble.RandomForestClassifier()
gs = sklearn.grid_search.RandomizedSearchCV(clf, {'n_estimators': np.arange(10, 500, 10), 
                                                  'max_depth': np.arange(3, 15)},
                                            scoring='accuracy', n_jobs=-1, n_iter=100, cv=skf, error_score=0)
gs.fit(X_train, y_train)
gs.best_params_, gs.best_score_
