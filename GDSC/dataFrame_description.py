# -*- coding: utf-8 -*-
"""
Created on Fri May 25 10:28:45 2018

@author: chn

DataFrame说明文件

path1='/home/chn/DeepLearning/20180524GDSC/GDSC/data1.csv.csv' #已去掉header了
data1=pd.read_csv(path1,header=None) #data1有0-3576列
得到一个dataFrame，（624,3577)


#data1_T=data1.T #转置
#c=data1_T[2]取第三列
#r=data1_T.iloc[4]取第五行



现在要转成一个list，有624个元素，每个元素是一个新的list，这样才能被sklearn使用
因为tolist()是按行取list，因此直接操作即可，不需要转置矩阵
l1=np.array(data1)
data1_list=l1.tolist()
之后可用data1_list[3]读取第4行，data1_list[3][0]读取第四行第一个元素

path2='/home/chn/DeepLearning/20180524GDSC/GDSC/data2.csv'
data2=pd.read_csv(path2,header=None) #data2有0-623列
得到一个dataFrame，（140,624）
因为140种药物要分开构建模型，因此转成一个140的list，每个元素是一个624的list
l2=np.array(data2)
data2_list=l2.tolist()



"""
