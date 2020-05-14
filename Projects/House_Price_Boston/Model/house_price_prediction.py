'''
@File    :   house_price_prediction.py
@Time    :   2020/05/14 13:48:52
@Author  :   Pan 
@Version :   1.0
@Contact :   pwd064@mail.ustc.edu.cn
@License :   (C)Copyright 2020-2025, USTC
@Desc    :   None
'''
# import os
# print(os.getcwd())

from   sklearn import datasets
from   sklearn import preprocessing
import pandas  as pd
import matplotlib.pyplot as plt 
from   sklearn import linear_model
from   sklearn.neighbors import KNeighborsRegressor
from   sklearn.tree      import DecisionTreeRegressor
from   sklearn.ensemble  import AdaBoostRegressor
from   sklearn.ensemble  import RandomForestRegressor
from   sklearn.model_selection import train_test_split
from   sklearn.model_selection import cross_val_score
from   sklearn.model_selection import learning_curve
from   sklearn.utils import shuffle
import numpy as np 

## ------------------------------------------------------
##                    特征工程
## ------------------------------------------------------ 
Boston = datasets.load_boston()
# print(dir(Boston))                # ['DESCR', 'data', 'feature_names', 'filename', 'target']
# print(Boston.DESCR)
# print(Boston.feature_names)         # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
# print(Boston.DESCR)
X, Y = Boston.data, Boston.target
X,Y  = preprocessing.scale(X), preprocessing.scale(Y)
df   = pd.DataFrame(X,columns=Boston.feature_names)
df['target'] = Y
df   = df[df.target!=2.99]
df   = shuffle(df)
# df.to_csv('.\Data\data.csv')
# print(df.info())                  # 506 non-null    float64
# df.describe().to_csv('.\Data\describe.csv')
# plt.scatter(df['LSTAT'],Y,c='m',linewidths=0.5)
# plt.title('LSTAT'); plt.xlabel('LSTAT'); plt.ylabel('House price')
# plt.show()
## ------------------------------------------------------
##                    定义模型
## ------------------------------------------------------ 
# model = linear_model.Ridge()
# model = KNeighborsRegressor()
# model = DecisionTreeRegressor()
# model = AdaBoostRegressor()
model = RandomForestRegressor()
## ------------------------------------------------------
##                    预测/验证
## ------------------------------------------------------ 
# ['B','PTRATIO','TAX','RAD','NOX','INDUS','ZN','CHAS']
# df = df.drop(['B'],axis=1)
X, Y = df.iloc[:,0:-1], df.iloc[:,-1]
score = cross_val_score(model, X, Y, cv=5)
print(score, np.mean(score))
train_sizes, train_scores, test_scores = learning_curve(model,X,Y,cv=5)
train_scores_mean = np.mean(train_scores,axis=1)
test_scores_mean  = np.mean(test_scores,axis=1)
plt.plot(train_sizes,train_scores_mean,'o-',label='train')
plt.plot(train_sizes,test_scores_mean,'o-',label='test')
plt.xlabel('samples')
plt.ylabel('train/cv accuracy')
plt.legend()
plt.show()
