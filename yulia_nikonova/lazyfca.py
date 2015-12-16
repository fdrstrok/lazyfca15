
# coding: utf-8

# In[144]:

import pandas as pd
import numpy as np


# In[145]:

import copy

def dummy_encode_categorical_columns(data):
    result_data = copy.deepcopy(data)
    for column in data.columns.values:
        result_data = pd.concat([result_data, pd.get_dummies(result_data[column], prefix = column, prefix_sep = ': ')], axis = 1)
        del result_data[column]
    return result_data


# In[146]:

def parse_file(name):
    df = pd.read_csv(name, sep=',')
    df = df.replace(to_replace='positive', value=1)
    df = df.replace(to_replace='negative', value=0)
    y = np.array(df['V10'])
    del df['V10']
    bin_df = dummy_encode_categorical_columns(df)
    return np.array(bin_df).astype(int), y
    


# In[147]:

df_test = pd.read_csv('test1.csv', sep=',')
df_train = pd.read_csv('train1.csv', sep=',')


# In[148]:

X_train, y_train = parse_file('train1.csv')
X_test, y_test = parse_file('test1.csv')


# In[149]:

X_train_pos = X_train[y_train == 1]
X_train_neg = X_train[y_train == 0]


# In[150]:

y_pred = []
for test_obj in X_test:
    pos = np.sum(test_obj == X_train_pos) / float(len(X_train_pos))
    neg = np.sum(test_obj == X_train_neg) / float(len(X_train_neg))
    if (pos > neg):
        y_pred.append(1)
    else:
        y_pred.append(0)


# In[151]:

y_pred = np.array(y_pred)


# In[152]:

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[153]:

TP = np.sum(y_test * y_pred)
TN = np.sum(y_test + y_pred == 0)
FP = np.sum((y_test  == 0) * (y_pred == 1))
FN = np.sum((y_test  == 1) * (y_pred == 0))
TPR = float(TP) / np.sum(y_test == 1)
TNR = float(TN) / np.sum(y_test == 0)
FPR = float(FP) / (TP + FN)
NPV = float(TN) / (TN + FN)
FDR = float(FP) / (TP + FP)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)


# In[154]:

print "True Positive: {}\nTrue Negative: {}\nFalse Positive: {}\nFalse Negative: {}\nTrue Positive Rate: {}\nTrue Negative Rate: {}\nNegative Predictive Value: {}\nFalse Positive Rate: {}\nFalse Discovery Rate: {}\nAccuracyPrecision: {}\nRecall: {}".format(TP, TN, FP, FN, TPR, TNR, FPR, NPV, FDR, acc, prec, rec)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



