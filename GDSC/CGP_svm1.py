# -*- coding: utf-8 -*-
"""
Created on Fri May 27 17:04:54 2018

运行是错误的

@author: chn
将y中取NaN的值删除，对应x中值也删除。
注意到y=y_list[0]中的y[0]==nan，因此用y[0]作判断试试.试了不行。
因为读入pd后，nan是个float值，而奇怪的是float('nan')得到的float与读入的nan不能相等
因此，新建一个data2.csv的文件data2_num，其中nan取值作为0，代码见CGP_svm_2.py

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
    y0=y_list[0]
   
    for y in y_list:
        y_len=len(y0)
        x_select=[]
        y_select=[]
        for i in range(y_len):
            if (y0[i]!=y0[0]):
                x_select.append(x[i])
                y_select.append(y0[i])

        y_select=pd.Categorical(y_select).codes
        x_train,x_test,y_train,y_test=train_test_split(x_select,y_select,test_size=0.2,random_state=1)
        svm_model=svm.SVC(C=0.5,kernel='linear',decision_function_shape='ovr')
        #svm_model=svm.SVC(C=0.5,kernel='rbf',gamma=20,decision_function_shape='ovr')
        svm_model.fit(x_train,y_train)
        print("Accurary_train:",accuracy_score(y_train,svm_model.predict(x_train)))
        print("Accurary_test:",accuracy_score(y_test,svm_model.predict(x_test)))
        print(y_test.ravel())
        #print(svm_model.predict(x_test))
    
