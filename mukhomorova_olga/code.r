library(foreach)
library(doParallel)
setwd('C:/hse_homeworks_git/FCA homework/hw')
data<-read.table('hepatitis.data', sep=',', na.strings='?')
data<-na.omit(data)
names(data)<-c('Class','age','sex','steroid','antivirals','fatigue','malaise','anorexia','liver_big',
         'liver_firm','spleen_palpable','spiders','ascites','vaices','bilirubin','alk_phosphate',
         'sgot','albumin','protime','histology')
#steroid-на стероидах
#antivirals-принимает антивирусные
#fatigue - слабость
#malaise-недомогание,дискомфорт
#anorexia
#liver_big
#liver_firm - твердая,тугая печень
#spleen_palpable - селезенка пальпируема
#ascites - асцит
#vaices - варикоз
#histology - гистология

#all attrs are binary
#data1<-data[,c(1,3,4,5,6,7,8,9,10,11,12,13,14,20)]
#scaling (dihotomy)
data_scale<-data.frame()
for (i in 1:nrow(data)){if (data$age[i]<40){data_scale[i,'age<40']<-1;
data_scale[i,'age>=40']<-0}
  else{
    data_scale[i,'age<40']<-0;
    data_scale[i,'age>=40']<-1}}

for (i in 1:nrow(data)){if (data$bilirubin[i]<1){data_scale[i,'bilirubin<1']<-1;
data_scale[i,'bilirubin>=1']<-0}
  else{
    data_scale[i,'bilirubin<1']<-0;
    data_scale[i,'bilirubin>=1']<-1}}

for (i in 1:nrow(data)){if (data$alk_phosphate[i]<85){data_scale[i,'alk_ph<85']<-1;
data_scale[i,'alk_ph>=85']<-0}
  else{
    data_scale[i,'alk_ph<85']<-0;
    data_scale[i,'alk_ph>=85']<-1}}

for (i in 1:nrow(data)){if (data$sgot[i]<75){data_scale[i,'sgot<75']<-1;
data_scale[i,'sgot>=75']<-0}
  else{
    data_scale[i,'sgot<75']<-0;
    data_scale[i,'sgot>=75']<-1}}

for (i in 1:nrow(data)){if (data$albumin[i]<4){data_scale[i,'albumin<4']<-1;
data_scale[i,'albumin>=4']<-0}
  else{
    data_scale[i,'albumin<4']<-0;
    data_scale[i,'albumin>=4']<-1}}

#(ordinal)
for (i in 1:nrow(data)){if (data$protime[i]<40){data_scale[i,'protime<40']<-1;
data_scale[i,'protime<70']<-1;
data_scale[i,'protime<=100']<-1};
  if (data$protime[i]<70 & data$protime[i]>=40){data_scale[i,'protime<40']<-0;
  data_scale[i,'protime<70']<-1;
  data_scale[i,'protime<=100']<-1}
  if (data$protime[i]<=100 & data$protime[i]>=70){
    data_scale[i,'protime<40']<-0;
    data_scale[i,'protime<70']<-0;
    data_scale[i,'protime<=100']<-1}}

data1<-data
data1<-data1[!duplicated(data1), ]
data1<-data1[,!names(data1) %in% c('age','bilirubin','alk_phosphate','sgot','albumin','protime')]
data1<-data1-1
#mod 1
data1n<--(data1[,2:ncol(data1)]-1)
#mod 2
parameter1<-0.85
#0.7, 0.8375, 0.8125 0.7500 0.8125 1.0000 0.8125
#0.8, 0.8375, 0.9375 0.8125 0.8125 0.7500 0.8750
#0.85, 0.85, 0.9375 0.8750 1.0000 0.6875 0.7500
#0.9,  0.8375, 0.8125 0.8750 0.8750 0.8125 0.8125
#0.95, 0.8375, 0.8750 0.7500 0.8125 0.8750 0.8750
pos_attrs<-c()
for (i in 1:nrow(data1)){pos_attrs[i]<-sum(data1[i,3:14])}
treshold<-round(ncol(data1[i,3:14])*parameter1,0)
for (i in 1:nrow(data1)){if (pos_attrs[i]>=treshold){data1[i,'prop']<-1}else{data1[i,'prop']<-0}}
#mod 3
#basic hepatitis symptoms: loss of appetite (anorexia), fatigue, ascites.
pos_context<-split(data1, f = data1$Class)$`0`  #negative
pos_attrs<-c(); for (i in 1:ncol(data1)){pos_attrs[i]<-sum(pos_context[,i])}
pos_attrs
plot(x=c(1:15), y=pos_attrs)
for (i in 1:nrow(data1)){if (data1[i,'anorexia']==1 & data1[i,'fatigue']==1 & data1[i,'ascites']==1){data1[i,'main']<-1}else{data1[i,'main']<-0}}
#connecting all
data1<-cbind(data1, data1n, data_scale)
#parameter selection
parameter2<-3 #голосов за один вариант в 2 раза больше чем в другой
#10, 0.875, fpr=0.38
#1.1, 0.85, fpr=0.153
#cross validation
library(caret)
flds <- createFolds(c(1:nrow(data1)), k = 5, list = TRUE, returnTrain = FALSE)
accuracy<-c()
recall<-c()
precision<-c()
no_cores<-detectCores()-1
cl<-makeCluster(no_cores)
registerDoParallel(cl)
time<-Sys.time()
acc<-foreach(r=1:5,.combine='c') %do% {
      test<-data1[flds[[r]],]
      train<-data1[-flds[[r]],]
      pos_context<-split(train, f = train$Class)$`1`  #live
      #pos_context<-pos_context[1:round(nrow(pos_context)/6,0),]
      neg_context<-split(train, f = train$Class)$`0`  #die
      #classifying
      pos<-rep(0,nrow(test))
      neg<-rep(0,nrow(test))
      res_p<-c()
      res_n<-c()
      pos<-foreach(i=1:nrow(test), .combine = 'c') %:% 
        #positive
        foreach(j=1:nrow(pos_context), .combine = 'sum') %dopar% {
          intersection<-test[i,2:ncol(test)]*pos_context[j,2:ncol(pos_context)]
          for (k in 1:nrow(neg_context)){
            res_p[k]<-all(intersection*neg_context[k,2:ncol(neg_context)]==intersection)
          }
          if (sum(res_p)==0){1}
        }
      neg<-foreach(i=1:nrow(test), .combine = 'c') %:% 
        #negative
        foreach(j=1:nrow(neg_context), .combine = 'sum') %dopar% {
          intersection<-test[i,2:ncol(test)]*neg_context[j,2:ncol(neg_context)]
          for (k in 1:nrow(pos_context)){
            res_n[k]<-all(intersection*pos_context[k,2:ncol(pos_context)]==intersection)
          }
          if (sum(res_n)==0){1}
        }
      
      #prediction
      proportion<-data.frame()
      for (i in 1:nrow(test)){
        proportion[1,i]<-neg[i]/pos[i];
        proportion[2,i]<-pos[i]/neg[i];
        if (proportion[1,i]>=parameter2 | proportion[2,i]>=parameter2 | proportion[1,i]==0 | proportion[2,i]==0){
          
        }else{neg[i]<-1; pos[i]<-0}
      }
      for (i in 1:nrow(test)){
      if ((pos-neg)[i]>0){test[i,'class_predicted']<-1}else{test[i,'class_predicted']<-0}
      }
      
      #accuracy
      u = union(0, 1)
      t = table(factor(test$class_predicted, u), factor(test$Class, u))
      true_positive<-t[1,1]
      true_negative<-t[2,2]
      false_positive<-t[1,2]
      false_negative<-t[2,1]
      true_positive_rate<-true_positive/(sum(t[,1]))
      true_negative_rate<-true_negative/sum(t[,2])
      false_positive_rate<-false_positive/sum(t[,2])
      false_discovery_rate<-false_positive/sum(t[1,])
      negative_predictive_value<-true_negative/sum(t[2,])
      positive_predictive_value<-true_positive/sum(t[1,])
      accuracy[r]<-(true_positive+true_negative)/nrow(test);
      #recall[r]<-true_positive_rate
      #precision[r]<-positive_predictive_value
      #mean(accuracy);
      accuracy[r];
}
stopImplicitCluster()
Sys.time()-time