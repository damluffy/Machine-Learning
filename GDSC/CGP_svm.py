# -*- coding: utf-8 -*-
"""
Created on Fri May 25 17:04:54 2018

@author: chn

能成功运行，但是数据y其实是不对的，应该把y中取NAN的都删除，此外，把x对应的样本也删除。

"""


import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__=="__main__":
    path1='/home/chn/DeepLearning/20180524GDSC/GDSC/data1.csv'
    path2='/home/chn/DeepLearning/20180524GDSC/GDSC/data2.csv'#data1 or data2 has no header
    data1=pd.read_csv(path1,header=None) #data1有0-3576列
    data2=pd.read_csv(path2,header=None) #data2有0-623列
    l1=np.array(data1)
    x=l1.tolist()
    l2=np.array(data2)
    y_list=l2.tolist()
    
    
    for y in y_list:
        y=pd.Categorical(y).codes
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
        svm_model=svm.SVC(C=0.5,kernel='linear',decision_function_shape='ovr')
        #svm_model=svm.SVC(C=0.5,kernel='rbf',gamma=20,decision_function_shape='ovr')
        svm_model.fit(x_train,y_train)
        print("Accurary_train:",accuracy_score(y_train,svm_model.predict(x_train)))
        print("Accurary_test:",accuracy_score(y_test,svm_model.predict(x_test)))
        print(y_test.ravel())
        #print(svm_model.predict(x_test))
    
