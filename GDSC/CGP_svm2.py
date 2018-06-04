# -*- coding: utf-8 -*-
"""
Created on Fri May 28 10:27:54 2018

@author: chn
将y中取0的值删除，对应x中值也删除。
新建一个data2_num.csv的文件，其中nan取值作为0,true=1,false=-1

运行成功，只不过某些药物的数据特别有偏，会导致AUroc降低
最后将所有x_select预测后的值保存起来

"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__=="__main__":
    path1='/home/chn/DeepLearning/20180524GDSC/GDSC/data1.csv'
    path2='/home/chn/DeepLearning/20180524GDSC/GDSC/data2_num.csv'#data1 or data2 has no header
    data1=pd.read_csv(path1,header=None) #data1有0-3576列
    data2=pd.read_csv(path2,header=None) #data2有0-623列
    l1=np.array(data1)
    x=l1.tolist()
    l2=np.array(data2)
    y_list=l2.tolist()
    y_select_all_predict=[]
    y_select_all=[]
    
    for y in y_list:
        y_len=len(y)
        x_select=[]
        y_select=[]
        for i in range(y_len):
            if (y[i]!=0):
                x_select.append(x[i])
                y_select.append(y[i])

        x_train,x_test,y_train,y_test=train_test_split(x_select,y_select,test_size=0.2,random_state=1)
        svm_model=svm.SVC(C=0.5,kernel='linear',decision_function_shape='ovr')
        #svm_model=svm.SVC(C=0.5,kernel='rbf',gamma=20,decision_function_shape='ovr')
        svm_model.fit(x_train,y_train)
        print("Accurary_train:",accuracy_score(y_train,svm_model.predict(x_train)))
        print("Accurary_test:",accuracy_score(y_test,svm_model.predict(x_test)))
        a=svm_model.predict(x_select)
        y_select_all.append(y_select)
        y_select_all_predict.append(a.tolist())
    
    

        
    save_y_select_all=pd.DataFrame(columns=None,data=y_select_all)
    save_y_select_all_predict=pd.DataFrame(columns=None,data=y_select_all_predict)
    save_y_select_all.to_csv('/home/chn/DeepLearning/20180524GDSC/GDSC/y_select_all.csv')
    save_y_select_all_predict.to_csv('/home/chn/DeepLearning/20180524GDSC/GDSC/y_select_all_predict.csv')  
        
      
  
