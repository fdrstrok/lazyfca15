
# coding: utf-8

# # Илья Иваницкий

# In[554]:

import pandas as pd
import numpy as np
import random as rd


# In[555]:

source = 'car.data'
#source = 'SPECT.test'
k_ = 7
all_data = np.array(pd.read_csv(source,header=None))


# In[556]:

def k_fold_cross_validation(X, K, randomise = True):
    if randomise: from random import shuffle; 
    X=list(X)
    shuffle(X)
    for k in range(K):
        training = [x for i, x in enumerate(X) if i % K != k]
        validation = [x for i, x in enumerate(X) if i % K == k]
        yield training, validation, k


# In[557]:

for training, validation, k in k_fold_cross_validation(all_data, k_, True):  
    pd.DataFrame(training).to_csv("%s_train_%d.csv" % (source, k),index = False)
    pd.DataFrame(validation).to_csv("%s_test_%d.csv" % (source, k),index = False)


# In[558]:

import numpy as np
import datetime
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# In[559]:

attrib_names = list(range(all_data.shape[1]))


# In[560]:

def make_intent(example):
    global attrib_names
    return set([str(i)+':'+str(k) for i, k in zip(attrib_names, example)])


# In[561]:

def read_files(indexes_train, indexes_test):
    
    plus=[]
    minus=[]
    x_test=[]
    y_test=[]

    for index in indexes_train:
        index=str(index)
        q = open(source+"_train_" + index + ".csv", "r")
        if source == 'car.data':
            positive = "acc"
            negative = "unacc"
        else:
            positive = "1"
            negative = "0"
        train = [a.strip().split(",") for a in q]
        _plus = [make_intent(a[:-1]) for a in train if a[-1] != negative]
        _minus = [make_intent(a[:-1]) for a in train if a[-1] == negative]
        q.close()
        plus+=_plus
        minus+=_minus   
        
    for index in indexes_test:
        index=str(index)
        w = open(source+"_test_" + index + ".csv", "r")
        _unknown = [a.strip().split(",") for a in w]

        _x_test = [make_intent(a[:-1]) for a in _unknown]
        _y_test = [1 if a[-1] != negative else 0 for a in _unknown ]
        del _x_test[0]
        del _y_test[0]
        w.close()
        x_test+=_x_test
        y_test+=_y_test
        
    return plus, minus, x_test, y_test


# In[562]:

plus, minus, x_test, y_test = read_files([0],[0])


# In[563]:

from IPython.display import clear_output as clr


# In[564]:

print('a')
clr()


# In[565]:

len(plus), len(minus), len(x_test)


# # 1.
# Алгоритм основан на нормированной сумме мощности пересечения признаков неизвестного примера с примерами-(+) и примерами-(-). Неизвестный пример относится к тому набору, где эта сумма больша.

# In[566]:

import os
import math


# In[567]:

def intersection_classif(plus, minus, x_test, y_test, threshold = 0,pow_ = 1):
    y_pred=[]
    inv_len_plus  = 1./ len(plus) 
    inv_len_minus = 1./ len(minus) 
    inv_len_pos = 1./ len(plus[0])
    for i in x_test:
        unkn_set=i
        pos=0
        neg=0
        for j in plus:
            pos_set=j
            res=pos_set & unkn_set
            #pos+=math.pow(float(len(res)), pow_)*inv_len_pos
            pos+=math.pow(float(len(res)),len(res))*inv_len_pos
        pos = pos * inv_len_plus
        for j in minus:
            neg_set=j
            res=neg_set & unkn_set
            #neg+=math.pow(float(len(res)), pow_)* inv_len_pos
            neg+=math.pow(float(len(res)),len(res))* inv_len_pos
            
        neg = neg *inv_len_minus

        if (neg - pos > threshold):
            y_pred.append(0)
        else:
            if (neg - pos < -threshold):
                y_pred.append(1)
            else:
                    tresh=4 # порог вычисленный.
                    #print('in killer feature')
                    for j in plus:
                        #print('first_cycle'+str(j))
                        pos_set=j
                        res=pos_set & unkn_set
                        if len(res)!=0:
                            closure1=0.0
                            for k in plus:
                                if k.issuperset(res) and k!=j:
                                    closure1+=1
                            if closure1>tresh:
                                pos+= float(closure1) * inv_len_plus * float(len(res)) * inv_len_pos
                    pos=  pos * inv_len_plus

                    for j in minus:
                        #print('secon_cycle'+str(j))
                        neg_set=j
                        res=neg_set & unkn_set
                        if len(res)!=0:
                            closure2=0
                            for k in minus :
                                if k.issuperset(res) and k!=j:
                                    closure2+=1
                            if closure2>tresh:        
                                neg+=float(closure2) * inv_len_minus * float(len(res)) *inv_len_pos
                    neg = neg * inv_len_minus

                    if (pos < neg):
                        y_pred.append(0)
                    else:
                        y_pred.append(1)
            
            


    y_pred=np.array(y_pred)
    y_test=np.array(y_test)
    
    return print_metris(y_test, y_pred)


# 
# # 2.
# 
# 
# #### Простой алгоритм. 
# 
# Пересекаем с положительным и проверяем чтобы пересечение не вкладывалось ни в одно отрицательное. если все так, то начисляем голос в виде "относительной мощности пересечения".  
# 
# То же самое для отрицательных. 

# In[568]:

def intersection_with_contra_classif(plus, minus, x_test, y_test, threshold = 0,pow_ = 1):
    y_pred=[]
    counter=0
    inv_len_plus  = 1./ len(plus) 
    inv_len_minus = 1./ len(minus)
    inv_len_pos = 1./ len(plus[0])
    for i in x_test:
        counter+=1
        unkn_set=i
        pos=0
        neg=0

        for j in plus:
            pos_set=j
            res=pos_set & unkn_set
            closure=0
            for k in minus:
                if k.issuperset(res):
                    closure+=1
                    break
                    
            if closure == 0:
                #pos+=math.pow(float(len(res)),pow_) *inv_len_pos
                pos+=math.pow(float(len(res)),len(res)) *inv_len_pos
        pos= pos * inv_len_plus

        for j in minus:
            neg_set=j
            res=neg_set & unkn_set
            closure=0
            for k in plus:
                if k.issuperset(res):
                    closure+=1
                    break
                    
            if closure==0:
                #neg+=math.pow(float(len(res)),pow_)  *inv_len_pos
                neg+=math.pow(float(len(res)),len(res))  *inv_len_pos
        neg=neg*inv_len_minus 

#        if (counter % 10 == 0):
#            print 'done {} %'.format( round(float(counter)/len(x_test)*100, 2) )

        if (neg - pos > threshold):
            y_pred.append(0)
        else:
            if (neg - pos < -threshold):
                y_pred.append(1)
            else:
                # берем алгоритм 3. -- киллер фича.
                    tresh=4 # порог вычисленный.
                #if (pos==0 and neg==0):
                    for j in plus:
                        pos_set=j
                        res=pos_set & unkn_set
                        if len(res)!=0:
                            closure1=0.0
                            for k in plus:
                                if k.issuperset(res) and k!=j:
                                    closure1+=1
                            if closure1>tresh:
                                pos+= closure1*inv_len_plus  * len(res) *inv_len_pos
                    pos=  pos *inv_len_plus

                    for j in minus:
                        neg_set=j
                        res=neg_set & unkn_set
                        if len(res)!=0:
                            closure2=0
                            for k in minus :
                                if k.issuperset(res) and k!=j:
                                    closure2+=1
                            if closure2>tresh:        
                                neg+=closure2 *inv_len_minus * len(res) *inv_len_pos
                    neg =  neg*inv_len_minus

                    if (pos <= neg):
                        y_pred.append(0)
                    else:
                        y_pred.append(1)
            
            

    y_pred=np.array(y_pred)
    y_test=np.array(y_test)
    
    return print_metris(y_test, y_pred)


# # Test 
# 
# ### Скользящий контроль - обучается, тестируем.

# In[569]:

def print_metris(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    return acc


# ##### 1 алгоритм. 

# ###### Car.data data set

# In[570]:

acc=0.0
counter=0
begin=datetime.datetime.now()   
for i in range(k_):
        counter+=1
        plus, minus, x_test, y_test = read_files([i],[i])
        res_acc=intersection_classif(plus, minus, x_test, y_test,0.12,0)
        print (i,' train, ', (i), ' test. Accuracy _ ', round(res_acc,3))
        acc+=res_acc
end=datetime.datetime.now()
print ('finished! in time  ',end-begin)
print ('My average accuracy (with threshold=',0,') = ' ,round(acc/counter,3))


# In[571]:

acc=0.0
counter=0
begin=datetime.datetime.now()
for j in range(1,60,10):    
    for i in range(k_):
        counter+=1
        plus, minus, x_test, y_test = read_files([i],[i])
        res_acc=intersection_classif(plus, minus, x_test, y_test,0,j)
        #print (i,' train, ', (i), ' test. Accuracy _ ', round(res_acc,3))
        acc+=res_acc
    end=datetime.datetime.now()
    #print ('finished! in time  ',end-begin)
    print ('My average accuracy (with threshold=',j/1000,') = ' ,round(acc/counter,3))


# ##### 2 алгоритм

# ###### Car.data data set

# In[ ]:

counter = 0
acc = 0
for i in range(k_):
        counter+=1
        plus, minus, x_test, y_test = read_files([i],[i])
        res_acc=intersection_with_contra_classif(plus, minus, x_test, y_test,0,0)
        print (i,' train, ', (i), ' test. Accuracy _ ', round(res_acc,3))
        acc+=res_acc
end=datetime.datetime.now()
print ('finished! in time  ',end-begin)
print(str(0))
print ('My average accuracy ',float(acc)/counter)
print('________')
# 0.9918957807276164 - 0.12


# In[ ]:

for j in range(1,40,10):
    acc=0.0
    counter=0
    begin=datetime.datetime.now()
    for i in range(k_):
        counter+=1
        plus, minus, x_test, y_test = read_files([i],[i])
        res_acc=intersection_with_contra_classif(plus, minus, x_test, y_test,0,j)
    #    print (i,' train, ', (i), ' test. Accuracy _ ', round(res_acc,3))
        acc+=res_acc
    end=datetime.datetime.now()
    #print ('finished! in time  ',end-begin)
    #print(str(j))
    print ('My average accuracy with coef',j/1000,' is ',float(acc)/counter)
    #print('________')


# ### Распараллелим скользящий контроль

# In[ ]:

from threading import Thread

acc=[0.0 for i in range (k_)]
counter=0
begin=datetime.datetime.now()

def ww(i, j):
    global acc
    global counter
    counter+=1
    plus, minus, x_test, y_test = read_files([i],[i])
    print (i,' train, ', (i), ' test')
    acc[i]+=intersection_classif(plus, minus, x_test, y_test,j)

t = []
for i in range(k_):
    t.append(Thread(target=ww, args=(i,0.001,)))

for i in range(k_):
    t[i].setDaemon(True) 

for i in range(k_):
    t[i].start()

for i in range(k_):
    t[i].join()
    
end=datetime.datetime.now()
print ('finished! in time  ',end-begin)

rr= np.sum(np.array(acc))/counter
print ('My average accuracy ',rr)


# In[ ]:

for j in range(0,10):
    acc=[0.0 for i in range (k_)]
    counter=0
    begin=datetime.datetime.now()
    
    t = []
    for i in range(k_):
        t.append(Thread(target=ww, args=(i,j*1.0/100,)))

    for i in range(k_):
        t[i].setDaemon(True) 

    for i in range(k_):
        t[i].start()

    for i in range(k_):
        t[i].join()

    end=datetime.datetime.now()
    #print ('finished! in time  ',end-begin)
    #print(j*1.0/10)
    rr= np.sum(np.array(acc))/counter
    print ('My average accuracy with coef ',j*1.0/10,' is ',rr)
    #print('_____________')


# # Посмотрим популярные алгоритмы классификации.

# In[ ]:




# In[ ]:

import pandas as pd
import numpy as np
import copy
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# >  дальше мой)

# In[ ]:

a = np.array([['a','a','a'],['b',4,4],['c',0,9]])
b = np.array([['a','a','a'],['b',4,4],['c',0,9]])
print(a)
pd.Series(a[:,0]).factorize()
print(np.concatenate([a,b], axis = 1))


# In[ ]:




# In[ ]:

def fact(x):
    res = []
    uniq = set(x)
    for i in uniq:
        temp = list(map(lambda y: int(y == i), x))
        res.append(temp)
    return np.array(res).T


# In[ ]:

def change(x):
    if x == 'unacc':
        return 0
    else:
        return 1


# In[ ]:

res1=0.0
res2=0.0
res3=0.0

for i in range (k_):
    str1=source+'_train_'+str(i)+'.csv'
    str2=source+'_test_'+str(i)+'.csv'
    
    train_ = np.array(pd.read_csv(str1,delimiter=','))
    test_ = np.array(pd.read_csv(str2,delimiter=','))
        
    train =  train_[:,:-1]
    res = np.zeros([train.shape[0],0])
    for i in range(train.shape[1]):
        j = fact(train[:,i])
        res = np.concatenate([res,j],axis = 1)
    train = res
    label_tr = list(map(lambda x: change(x),train_[:,-1]))
    
    test =  test_[:,:-1]
    res = np.zeros([test.shape[0],0])
    for i in range(test.shape[1]):
        j = fact(test[:,i])
        res = np.concatenate([res,j],axis = 1)
    test = res
    label_te = list(map(lambda x: change(x),test_[:,-1]))
    
    clf1 = SVC(C=25,gamma=0.16)
    clf1.fit(train,label_tr)
    y_pred1 = clf1.predict(test)
    acc1 = accuracy_score(label_te, y_pred1)
    
    clf2 = RandomForestClassifier(n_estimators=100, random_state=3, min_samples_leaf=1) 
    clf2.fit(train,label_tr)
    y_pred2 = clf2.predict(test)
    acc2 = accuracy_score(label_te, y_pred2)
    

    clf3 =  KNeighborsClassifier(n_neighbors=18, p=1, weights='distance') 
    clf3.fit(train,label_tr)
    y_pred3 = clf3.predict(test)
    acc3 = accuracy_score(label_te, y_pred3)
    
    
    res1+=acc1
    res2+=acc2
    res3+=acc3
    
    print ('test ', i , 'train', i)
    print ('   SVM ',"Accuracy: {}".format(acc1))
    print ("   RF Accuracy: {}".format(acc2))
    print ("   KNN Accuracy: {}".format(acc3))
print 
print ('svm avg acc ',res1/k_)
print ('rf avg acc ',res2/k_)
print ('knn avg acc ',res3/k_)

