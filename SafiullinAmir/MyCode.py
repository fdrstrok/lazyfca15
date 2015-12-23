
# coding: utf-8

# # Сафиуллин Амир.

# In[42]:

import pandas as pd


# In[63]:

a=pd.DataFrame()
a['Алгоритмы'] = pd.Series(['1','2','3.1','3.2 (Tresh = 10)','4 (tresh = 0)',' ','SVC','RandForest', 'kNN'])
a['Avg_Accuracy'] = pd.Series([0.659, 0.99, 0.933, 0.979, 0.989, 0 ,0.998, 0.986, 0.982])


# >> сравнительная таблица показывающая среднее значение accuracy на массиве tic tac для разработанных алгоритмов и трех известных (SVM, randome forest, k-Nearest Neighbor )
# 
# >> (описание алгоритмов представлено ниже)

# In[64]:

a


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[26]:

import numpy as np
import datetime
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[17]:

def print_metris_all(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    roc_auc=roc_auc_score(y_test, y_pred)
    print "Accuracy: {}\nROC AUC: {}".format(acc,roc_auc)
    
    TP = np.sum(y_test * y_pred)
    TN = np.sum(y_test + y_pred == 0)
    FP = np.sum((y_test  == 0) * (y_pred == 1))
    FN = np.sum((y_test  == 1) * (y_pred == 0))
    try:
        TPR = float(TP) / (TP + FN)
        TNR = float(TN) / (TN + FP)
        FPR = float(FP) / (FP + TN)
        NPV = float(TN) / (TN + FN)
        FDR = float(FP) / (TP + FP)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        print '''
True Positive: {}
True Negative: {} 
False Positive: {}
False Negative: {}
True Positive Rate: {}
True Negative Rate: {}
Negative Predictive Value: {}
False Positive Rate: {}
False Discovery Rate: {}
Precision: {}
Recall: {}'''.format(TP, TN, FP, FN, TPR, TNR, NPV, FPR, FDR, prec, rec)
    except:
        print 'sorry'


# In[27]:

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
]

#attrib_names = [ str(i) for i in range(0,37) ]


# In[28]:

def make_intent(example):
    global attrib_names
    return set([i+':'+str(k) for i, k in zip(attrib_names, example)])


# In[29]:

def read_files(indexes_train, indexes_test):
    
    plus=[]
    minus=[]
    x_test=[]
    y_test=[]

    for index in indexes_train:
        index=str(index)
#        q = open("./kr/kr-vs-kp.data_train_" + index + ".txt", "r")
        q = open("../train" + index + ".csv", "r")
        positive = "positive"
        negative = "negative"
#        positive = "won"
#        negative = "nowin"
        train = [a.strip().split(",") for a in q]
        _plus = [make_intent(a[:-1]) for a in train if a[-1] == positive]
        _minus = [make_intent(a[:-1]) for a in train if a[-1] == negative]
        q.close()
        plus+=_plus
        minus+=_minus   
        
    for index in indexes_test:
        index=str(index)
#        w = open("./kr/kr-vs-kp.data_validation_" + index + ".txt", "r")
        w = open("../test" + index + ".csv", "r")
        _unknown = [a.strip().split(",") for a in w]

        _x_test = [make_intent(a[:-1]) for a in _unknown]
        _y_test = [1 if a[-1] == positive else 0 for a in _unknown ]
        del _x_test[0]
        del _y_test[0]
        w.close()
        x_test+=_x_test
        y_test+=_y_test
        
    return plus, minus, x_test, y_test


# In[17]:

plus, minus, x_test, y_test = read_files([1],[1])


# In[18]:

len(plus), len(minus), len(x_test)


# In[54]:

# Пересечения
count = 0
for i in plus:
    for j in x_test:
        if i == j: 
            count+=1
print count


# In[ ]:




# # 1.
# Алгоритм основан на нормированной сумме мощности пересечения признаков неизвестного примера с примерами-(+) и примерами-(-). Неизвестный пример относится к тому набору, где эта сумма больша.

# In[30]:

def vlob(plus, minus, x_test, y_test):
    y_pred=[]
    for i in x_test:
        unkn_set=i
        pos=0
        neg=0

        for j in plus:
            pos_set=j
            res=pos_set & unkn_set
            pos+=float(len(res)) / len(pos_set)
        pos=float(pos) / len(plus) 

        for j in minus:
            neg_set=j
            res=neg_set & unkn_set
            neg+=float(len(res)) / len(neg_set)
        neg=float(neg) / len(minus) 

        if (pos < neg):
            y_pred.append(0)
        else:
            y_pred.append(1)

    y_pred=np.array(y_pred)
    y_test=np.array(y_test)
    
    return print_metris(y_test, y_pred)


# In[ ]:




# # 2.
# 
# 
# #### Простой алгоритм. 
# 
# Пересекаем с положительным и проверяем чтобы пересечение не вкладывалось ни в одно отрицательное. если все так, то начисляем голос в виде "относительной мощности пересечения".  
# 
# То же самое для отрицательных. 
# 
# Где сумма накопленных "голосов" больше - туда и классифицируем, в случае равенства смотрим по поддержке как в алгоритме 3.2 (с порогом 10.)

# In[43]:

def is_in_intent(plus, minus, x_test, y_test):
    y_pred=[]
    counter=0
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
                    
            if closure==0:
                pos+=float(len(res)) / len(pos_set)
        pos=float(pos) / len(plus)   

        for j in minus:
            neg_set=j
            res=neg_set & unkn_set
            closure=0
            for k in plus:
                if k.issuperset(res):
                    closure+=1
                    break
                    
            if closure==0:
                neg+=float(len(res)) / len(neg_set)
        neg=float(neg) / len(minus) 

#        if (counter % 10 == 0):
#            print 'done {} %'.format( round(float(counter)/len(x_test)*100, 2) )

        if (pos < neg):
            y_pred.append(0)
        else:
            if (neg < pos):
                y_pred.append(1)
            else:
                # берем алгоритм 3. -- киллер фича.
                tresh=10 # порог вычисленный.
                if (pos==0 and neg==0):
                    for j in plus:
                        pos_set=j
                        res=pos_set & unkn_set
                        if len(res)!=0:
                            closure1=0.0
                            for k in plus:
                                if k.issuperset(res) and k!=j:
                                    closure1+=1
                            if closure1>tresh:
                                pos+= float(closure1) / len(plus) * float(len(res)) / len(pos_set)
                    pos=  float(pos) / len(plus) 

                    for j in minus:
                        neg_set=j
                        res=neg_set & unkn_set
                        if len(res)!=0:
                            closure2=0
                            for k in minus :
                                if k.issuperset(res) and k!=j:
                                    closure2+=1
                            if closure2>tresh:        
                                neg+=float(closure2) / len(minus) * float(len(res)) / len(neg_set) 
                    neg =  float(neg) / len(minus) 

                    if (pos < neg):
                        y_pred.append(0)
                    else:
                        y_pred.append(1)
            
            

    y_pred=np.array(y_pred)
    y_test=np.array(y_test)
    
    return print_metris(y_test, y_pred)


# In[ ]:




# In[ ]:




# # 3.1
# 
# ### Задаем веса рассматривая поддержку
# 
# для примера начисляем голос пропорционально "относительной мощности пересечения" пересечения (признаков неизвестного примера и (+)примера) с остальными (+)-примерами, если это пересечение вкладывается в них  
# 
# для минуса то же самое.
# 
# Где больше - туда и относим

# In[40]:

def is_in_int1(plus, minus, x_test, y_test):
    y_pred=[]
    counter=0
    for i in x_test:
        counter+=1
        unkn_set=i
        pos=0
        neg=0

        for j in plus:
            pos_set=j
            res=pos_set & unkn_set
            if len(res)!=0:
                for k in plus:
                    if k.issuperset(res) and k!=j:
                        #pos+= 1.0 * len(res) / len(plus) #float(len(k&res)) / len(res) * float(len(res)) / len (pos_set) / len(plus)
                        pos+= float(len(k&res)) / len(res) * float(len(res)) / len (pos_set) / len(plus)
                        
        pos=  float(pos) / len(plus) 

        for j in minus:
            neg_set=j
            res=neg_set & unkn_set
            if len(res)!=0:
                for k in minus :
                    if k.issuperset(res) and k!=j:
                        #neg+= 1.0 * len(res) / len(minus) #float(len(k&res)) / len(res) * float(len(res)) / len(neg_set) / len(minus)
                        neg+= float(len(k&res)) / len(res) * float(len(res)) / len(neg_set) / len(minus)

        neg =  float(neg) / len(minus) 

        if (pos < neg):
            y_pred.append(0)
        else:
            if (pos==0 and neg==0):
                print 'i dont know'
                y_pred.append(0.5)
            else:
                y_pred.append(1)

    y_pred=np.array(y_pred)
    y_test=np.array(y_test)
    end=datetime.datetime.now()

    
    return print_metris(y_test, y_pred) 


# In[ ]:




# # 3.2
# 
# 
# ### Переход к простым весам и порог для поддержки
# 
# #### лучший tresh = 10 максимизирующий среднее accuracy для tic-tac data set
# 
# для плюс контекста начисляем голос пропорционально поддержке, если она больше порога.
# 
# для минуса то же самое.
# 
# Где больше - туда и относим

# In[33]:

def is_in_int2(plus, minus, x_test, y_test, tresh):
    begin=datetime.datetime.now()
    y_pred=[]
    counter=0
    for i in x_test:
        counter+=1
        unkn_set=i
        
        pos=0
        neg=0

        for j in plus:
            pos_set=j
            res=pos_set & unkn_set
            if len(res)!=0:
                closure1=0.0
                for k in plus:
                    if k.issuperset(res) and k!=j:
                        closure1+=1
                if closure1>tresh:
                    pos+= float(closure1) / len(plus) * float(len(res)) / len(pos_set)
                        
        pos=  float(pos) / len(plus) 

        for j in minus:
            neg_set=j
            res=neg_set & unkn_set
            if len(res)!=0:
                closure2=0
                for k in minus :
                    if k.issuperset(res) and k!=j:
                        closure2+=1
                if closure2>tresh:        
                    neg+=float(closure2) / len(minus) * float(len(res)) / len(neg_set) 
                
        neg =  float(neg) / len(minus) 
        
           
        if (pos < neg):
            y_pred.append(0)
        else:
            if (pos==0 and neg==0):
                print 'i dont know' # for roc auc - 0,5
                y_pred.append(0.5)
            else:
                y_pred.append(1)

    y_pred=np.array(y_pred)
    y_test=np.array(y_test)
    end=datetime.datetime.now()
  
    return print_metris(y_test, y_pred) 


# In[ ]:




# # 4.
# 
# ### Объединяем 2 и 3.1 алгоритмы
# 
# ##### задаем штраф если пересечение с плюс примером вкладывается в отрицательный пример.
# (видимо, логично, что порог на штраф тут 0, ведь поддержка то никак не должна влиять, либо влиять как то отрицательно)
# 
# ### Лучший thresh = 0 

# In[85]:

def costs_and_penalty(plus, minus, x_test, y_test, tresh):
    y_pred=[]
    counter=0
    for i in x_test:
        counter+=1
        unkn_set=i
        pos=0
        neg=0

        for j in plus:
            pos_set=j
            res=pos_set & unkn_set
            if len(res)!=0:
                closure=0
                for k in minus:
                    if k.issuperset(res):
                        closure+=float(len(k&res)) / len(res)

                if closure != 0:        
                    closure1=0
                    for k in plus:
                        if k.issuperset(res):
                            closure1+=float(len(k&res)) / len(res)

                if (closure == 0):
                    pos+=float(len(res)) / len(pos_set) / len(plus)
                else:
                    if (closure1 > tresh):
                        pos-= float(closure) / len(minus) * float(closure1) / len(plus) * (float(len(res)) / len(pos_set))  / len(plus)
                        
        for j in minus:
            neg_set=(j)
            res=neg_set&unkn_set
            if len(res)!=0:
                closure=0
                for k in plus:
                    if k.issuperset(res):
                        closure+=float(len(k&res))/len(res)

                if closure != 0:        
                    closure1=0
                    for k in minus:
                        if k.issuperset(res):
                            closure1+=float(len(k&res))/len(res)

                if (closure == 0):
                    neg+=float(len(res))/len(neg_set) / len(minus)
                else:
                    if (closure1 > tresh):
                         neg-= float( closure ) / len( plus ) * float(closure1) / len(minus) * (float(len(res)) / len(neg_set))  / len(minus)

#        if (counter % 10 == 0):
#            print 'done ', round(float(counter)/len(x_test)*100, 2) , ' %  in ',datetime.datetime.now()-begin

        if (pos < neg):
            y_pred.append(0)
        else:
            if (pos == 0 and neg == 0):
                print 'i dont know'
                y_pred.append(0.5)
            else:
                y_pred.append(1)

    y_pred=np.array(y_pred)
    y_test=np.array(y_test)

    return print_metris(y_test, y_pred)


# In[ ]:




# In[ ]:




# In[ ]:




# # Test 
# 
# ### Скользящий контроль - обучается, тестируем.

# In[31]:

def print_metris(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    return acc


# ##### 1 алгоритм. 

# ###### Kr vs kp data set

# In[47]:

acc=0.0
counter=0
begin=datetime.datetime.now()
for i in range(0,10):
    counter+=1
    plus, minus, x_test, y_test = read_files([i],[i])
    res_acc=vlob(plus, minus, x_test, y_test)
    print i,' train, ', (i), ' test. Accuracy _ ', round(res_acc,3)
    acc+=res_acc
end=datetime.datetime.now()
print 'finished! in time  ',end-begin
print 'My average accuracy ',round(acc/counter,3)


# ###### tic tac data set

# In[121]:

acc=0.0
counter=0
begin=datetime.datetime.now()
for i in range(1,11):
    counter+=1
    plus, minus, x_test, y_test = read_files([i],[i])
    res_acc=vlob(plus, minus, x_test, y_test)
    print i,' train, ', (i), ' test. Accuracy _ ', round(res_acc,3)
    acc+=res_acc
end=datetime.datetime.now()
print 'finished! in time  ',end-begin
print 'My average accuracy ',round(acc/counter,3)


# In[ ]:




# ##### 2 алгоритм

# ###### Kr vs kp data set

# In[46]:

acc=0.0
counter=0
begin=datetime.datetime.now()
for i in range(0,10):
    counter+=1
    plus, minus, x_test, y_test = read_files([i],[i])
    res_acc=is_in_intent(plus, minus, x_test, y_test)
    print i,' train, ', (i), ' test. Accuracy _ ', round(res_acc,3)
    acc+=res_acc
end=datetime.datetime.now()
print 'finished! in time  ',end-begin
print 'My average accuracy ',float(acc)/counter


# ###### tic tac data set

# In[44]:

acc=0.0
counter=0
begin=datetime.datetime.now()
for i in range(1,11):
    counter+=1
    plus, minus, x_test, y_test = read_files([i],[i])
    res_acc=is_in_intent(plus, minus, x_test, y_test)
    print i,' train, ', (i), ' test. Accuracy _ ', round(res_acc,3)
    acc+=res_acc
end=datetime.datetime.now()
print 'finished! in time  ',end-begin
print 'My average accuracy ',float(acc)/counter


# In[ ]:




# ##### 3.1 алгоритм

# ###### tic tac data set

# In[41]:

acc=0.0
counter=0
begin=datetime.datetime.now()
for i in range(1,10):
    counter+=1
    plus, minus, x_test, y_test = read_files([i],[i])
    res_acc=is_in_int1(plus, minus, x_test, y_test)
    print i,' train, ', (i), ' test. Accuracy _ ', round(res_acc,3)
    acc+=res_acc
end=datetime.datetime.now()
print 'finished! in time  ',end-begin
print 'My average accuracy ',round(acc/counter,3)


# ##### 3.2 алгоритм с порогом  = 10

# ###### tic tac data set

# In[68]:

acc=0.0
counter=0
begin=datetime.datetime.now()
for i in range(1,10):
    counter+=1
    plus, minus, x_test, y_test = read_files([i],[i])
    res_acc=is_in_int2(plus, minus, x_test, y_test, 10)
    print i,' train, ', (i), ' test. Accuracy _ ', round(res_acc,3)
    acc+=res_acc
end=datetime.datetime.now()
print 'finished! in time  ',end-begin
print 'My average accuracy ',acc/counter


# ##### 4. алгоритм с порогом 0

# ###### tic tac data set

# In[62]:

acc=0.0
counter=0
begin=datetime.datetime.now()
for i in range(1,10):
    counter+=1
    plus, minus, x_test, y_test = read_files([i],[i])
    res_acc=costs_and_penalty(plus, minus, x_test, y_test,0)
    print i,' train, ', (i), ' test. Accuracy _ ', round(res_acc,3)
    acc+=res_acc
end=datetime.datetime.now()
print 'finished! in time  ',end-begin
print 'My average accuracy ', acc/counter


# In[ ]:




# In[ ]:




# ### Подбор порогов для алгоритмов 3.2 и 4.

# In[63]:

import numpy as np


# ###### tic tac data set

# In[64]:

acc=[0.0 for i in range(0,10)]
counter=0
begin=datetime.datetime.now()
for i in range(1,10):
    counter+=1
    plus, minus, x_test, y_test = read_files([i],[i])
    print i,' train, ', (i), ' test'
    for tresh in range(0,10):
        res_acc=is_in_int2(plus, minus, x_test, y_test, tresh)
        print 'tr ',tresh,'  and acc ', res_acc
        acc[tresh]+=res_acc
end=datetime.datetime.now()
print 'finished! in time  ',end-begin
print 'My average accuracy ',np.array(acc)/counter


# > treshold = 10 
# 
# > больше 12 - вышло что уменьшается точность

# ###### tic tac data set

# In[86]:

acc=[0.0 for i in range(0,10)]
counter=0
begin=datetime.datetime.now()
for i in range(1,10):
    counter+=1
    plus, minus, x_test, y_test = read_files([i],[i])
    print i,' train, ', (i), ' test'
    for tresh in range(0,10):
        res_acc=costs_and_penalty(plus, minus, x_test, y_test, tresh)
        print 'tr ',tresh,'  and acc ', res_acc
        acc[tresh]+=res_acc
end=datetime.datetime.now()
print 'finished! in time  ',end-begin
rr= np.array(acc)/counter
print 'My average accuracy ',rr


# > тут понятно что порог - это лишнее

# In[129]:

treshold = 0 #  возьмем меньшее, то есть если поддержка больше нуля.


# ### Распараллелим скользящий контроль

# In[128]:

from threading import Thread

acc=[0.0 for i in range (1,11)]
counter=0
begin=datetime.datetime.now()

def ww(i):
    global acc
    global counter
    counter+=1
    plus, minus, x_test, y_test = read_files([i],[i])
    print i,' train, ', (i), ' test'
    acc[i-1]+=is_in_intent(plus, minus, x_test, y_test)
        

t1 = Thread(target=ww, args=(1,))
t2 = Thread(target=ww, args=(2,))
t3 = Thread(target=ww, args=(3,))
t4 = Thread(target=ww, args=(4,))
t5 = Thread(target=ww, args=(5,))
t6 = Thread(target=ww, args=(6,))
t7 = Thread(target=ww, args=(7,))
t8 = Thread(target=ww, args=(8,))
t9 = Thread(target=ww, args=(9,))
t10 = Thread(target=ww, args=(10,))
t1.setDaemon(True) 
t2.setDaemon(True) 
t3.setDaemon(True) 
t4.setDaemon(True) 
t5.setDaemon(True) 
t6.setDaemon(True) 
t7.setDaemon(True) 
t8.setDaemon(True) 
t9.setDaemon(True) 
t10.setDaemon(True) 

t1.start()
t2.start()
t3.start()
t4.start()
t5.start()
t6.start()
t7.start()
t8.start()
t9.start()
t10.start()


t1.join()
t2.join()
t3.join()
t4.join()
t5.join()
t6.join()
t7.join()
t8.join()
t9.join()
t10.join()

end=datetime.datetime.now()
print 'finished! in time  ',end-begin

rr= np.sum(np.array(acc))/counter
print 'My average accuracy ',rr


# In[ ]:




# > проверим  поведение алгоритма 3.2 при параметре больше 9  - самый лучший результат при параметре 10

# In[67]:

acc=[0.0 for i in range(10,13)]
counter=0
begin=datetime.datetime.now()
for i in range(1,10):
    counter+=1
    plus, minus, x_test, y_test = read_files([i],[i])
    print i,' train, ', (i), ' test'
    for tresh in range(10,13):
        res_acc=is_in_int2(plus, minus, x_test, y_test, tresh)
        print 'tr ',tresh,'  and acc ', res_acc
        acc[tresh-10]+=res_acc
end=datetime.datetime.now()
print 'finished! in time  ',end-begin
print 'My average accuracy ',np.array(acc)/counter


# In[ ]:




# In[ ]:




# # Посмотрим популярные алгоритмы классификации.

# In[ ]:




# In[65]:

import pandas as pd
import numpy as np
import copy
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# >  код взят у YuliaNikonova. Спасибо ей!

# In[66]:


def dummy_encode_categorical_columns(data):
    result_data = copy.deepcopy(data)
    for column in data.columns.values:
        result_data = pd.concat([result_data, pd.get_dummies(result_data[column], prefix = column, prefix_sep = ': ')], axis = 1)
        del result_data[column]
    return result_data

def parse_file(name):
    df = pd.read_csv(name, sep=',')
    df = df.replace(to_replace='positive', value=1)
    df = df.replace(to_replace='negative', value=0)
    y = np.array(df['V10'])
    del df['V10']
    bin_df = dummy_encode_categorical_columns(df)
    return np.array(bin_df).astype(int), y
    


# >  дальше мой)

# In[84]:

res1=0.0
res2=0.0
res3=0.0

for i in range (1,11):
    str1='../train'+str(i)+'.csv'
    str2='../test'+str(i)+'.csv'
    X, y = parse_file(str1)
    X_test, y_test = parse_file(str2)

    clf1 = SVC(C=25,gamma=0.16)
    clf1.fit(X, y)
    y_pred1 = clf1.predict(X_test)
    acc1 = accuracy_score(y_test, y_pred1)
    
    clf2 = RandomForestClassifier(n_estimators=100, random_state=3, min_samples_leaf=1, n_jobs=2) 
    clf2.fit(X, y)
    y_pred2 = clf2.predict(X_test)
    acc2 = accuracy_score(y_test, y_pred2)
    

    clf3 =  KNeighborsClassifier(n_neighbors=18, n_jobs=2, p=1, weights='distance') 
    clf3.fit(X, y)
    y_pred3 = clf3.predict(X_test)
    acc3 = accuracy_score(y_test, y_pred3)
    
    
    res1+=acc1
    res2+=acc2
    res3+=acc3
    
    print 'test ', i , 'train', i
    print '   SVM ',"Accuracy: {}".format(acc1)
    print "   RF Accuracy: {}".format(acc2)
    print "   KNN Accuracy: {}".format(acc3)
print 
print 'svm avg acc ',res1/10
print 'rf avg acc ',res2/10
print 'knn avg acc ',res3/10


# In[ ]:




# In[ ]:



