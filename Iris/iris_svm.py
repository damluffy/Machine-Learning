# -*- coding: utf-8 -*-
#                            支持向量机
########################################################################
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__=="__main__":
    path='iris.data'
    data=pd.read_csv(path,header=None) #data有5列
    data[4]=pd.Categorical(data[4]).codes# 将第5列data[4]的类别变成数字
    x,y=np.split(data.values,(4,),axis=1) #分割
    #print(x)
    #print(y)
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
    
    svm_model=svm.SVC(C=0.5,kernel='linear',decision_function_shape='ovr')
    #svm_model=svm.SVC(C=0.5,kernel='rbf',gamma=20,decision_function_shape='ovr')
    svm_model.fit(x_train,y_train.ravel())
    print(accuracy_score(y_train,svm_model.predict(x_train)))
    print("Accurary:",accuracy_score(y_test,svm_model.predict(x_test)))
    print(y_test.ravel())
    #print(svm_model.predict(x_test))
    
