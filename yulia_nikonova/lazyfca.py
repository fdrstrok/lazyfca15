
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np


# In[2]:

import copy

def dummy_encode_categorical_columns(data):
    result_data = copy.deepcopy(data)
    for column in data.columns.values:
        result_data = pd.concat([result_data, pd.get_dummies(result_data[column], prefix = column, prefix_sep = ': ')], axis = 1)
        del result_data[column]
    return result_data


# In[3]:

def parse_file(name):
    df = pd.read_csv(name, sep=',')
    df = df.replace(to_replace='positive', value=1)
    df = df.replace(to_replace='negative', value=0)
    y = np.array(df['V10'])
    del df['V10']
    bin_df = dummy_encode_categorical_columns(df)
    return np.array(bin_df).astype(int), y
    


# In[4]:

df_test = pd.read_csv('../test1.csv', sep=',')
df_train = pd.read_csv('../train1.csv', sep=',')


# In[11]:

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[13]:

def pred_data(i, part):
    X_train, y_train = parse_file('../train' + str(i) + '.csv')
    X_test, y_test = parse_file('../test' + str(i) + '.csv')
    X_train_pos = X_train[y_train == 1]
    X_train_neg = X_train[y_train == 0]
    
    y_pred = []

    for test_obj in X_test:
        pos = 0
        neg = 0
        for pos_obj in X_train_pos:
            if np.sum(test_obj == pos_obj) > int(len(pos_obj) * part):
                pos += 1
        for neg_obj in X_train_neg:
            if np.sum(test_obj == neg_obj) > int(len(neg_obj) * part):
                neg += 1

        pos = pos / float(len(X_train_pos))
        neg = neg / float(len(X_train_neg))
        if (pos > neg):
            y_pred.append(1)
        else:
            y_pred.append(0)
            
    y_pred = np.array(y_pred)
    #print y_pred
    
    '''
    TP = np.sum(y_test * y_pred)
    TN = np.sum(y_test + y_pred == 0)
    FP = np.sum((y_test  == 0) * (y_pred == 1))
    FN = np.sum((y_test  == 1) * (y_pred == 0))
    TPR = float(TP) / np.sum(y_test == 1)
    TNR = float(TN) / np.sum(y_test == 0)
    FPR = float(FP) / (TP + FN)
    NPV = float(TN) / (TN + FN)
    FDR = float(FP) / (TP + FP)
    '''
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    print "Dataset {}".format(i)
    #print "True Positive: {}\nTrue Negative: {}\nFalse Positive: {}\nFalse Negative: {}\nTrue Positive Rate: {}\nTrue Negative Rate: {}\n\
    #Negative Predictive Value: {}\nFalse Positive Rate: {}\nFalse Discovery Rate: {}\nAccuracy: {}\nPrecision: {}\nRecall: {}".format(TP, TN, FP, FN, TPR, TNR, FPR, NPV, FDR, acc, prec, rec)
    print "Accuracy: {}\nPrecision: {}\nRecall: {}".format(acc, prec, rec)
    print("===========")


# In[14]:

for i in range(0, 10):
    pred_data(i+1, 0.5)


# In[15]:

for i in range(0, 10):
    pred_data(i+1, 0.4)


# In[16]:

for i in range(0, 10):
    pred_data(i+1, 0.3)


# In[16]:




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



