'''
@File    :   Titanic_survive_prediction.py
@Time    :   2020/05/18 15:51:27
@Author  :   Pan
@Version :   1.0
@Contact :   pwd064@mail.ustc.edu.cn
@License :   (C)Copyright 2020-2025, USTC
@Desc    :   None
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
import pickle #pickle模块

def nametonum(name: str):
    if 'Mr.' in name:
        return 1
    elif 'Miss.' in name:
        return 2
    elif 'Mrs.' in name:
        return 3
    else:
        return 0

##---------------------------------------------------
##         特征工程：导入数据集并做基本处理
##---------------------------------------------------
##['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
df = pd.DataFrame(pd.read_csv('./data/test.csv'))
## 用0、1来表示女性和男性
df['Sex'].replace('female', 0,inplace=True)
df['Sex'].replace('male', 1,inplace=True)
## 用名字中的尊称作区分
df['Name'] = df['Name'].apply(nametonum)
# print(df['Name'])
df['Fare'].replace(np.nan, df['Fare'].mean(),inplace=True)
## 用平均年龄填充年龄缺失值
df['Age'].replace(np.nan, df['Age'].mean(),inplace=True)
# print(df['Age'].mean())
df['Age'] = pd.cut(df['Age'],6,labels=list(range(1,7)))
# df['Age'] = preprocessing.maxabs_scale(df['Age'],axis=0)
# print(df['Age'])
df = df.drop(columns='Ticket')
# print(df.describe()['Fare'])
# df['Fare'] = preprocessing.scale(df['Fare'],axis=0,with_mean=False,with_std=True)
# print(df['Fare'].describe())
df['Fare'] = pd.cut(df['Fare'],10,labels=list(range(1,11)))
# df['Fare'] = preprocessing.maxabs_scale(df['Fare'],axis=0)
# df['Fare'].to_csv('aa.csv')
# print(set(df['Fare']))
df['Cabin'] = df['Cabin'].fillna(0)  
df['Cabin'] = df['Cabin'].apply(lambda x: 1 if x!=0 else 0)
# df['Cabin'] = df['Cabin'].apply(lambda x: x[0] if x!=0 else 0)
# mapclass = {0:0,'T':1,'F':2,'E':3,'B':4,'D':5,'A':6,'G':7,'C':8}
# df['Cabin'] = df['Cabin'].map(mapclass)
# print(set(df['Cabin']))
df['Embarked'] = df['Embarked'].fillna('C')
mapclass2 = {'C':1,'Q':2,'S':3}
df['Embarked'] = df['Embarked'].map(mapclass2)
# key = pd.pivot_table(df,values='Survived',index='Name')      # aggfunc=np.sum
# # print(key)
# plt.bar(key.index,np.reshape(key.values,key.index.shape),width=0.4)
# plt.xlabel('Name')
# plt.ylabel('Survival rate')
# plt.show()
# df.to_csv('./data/processed.csv')
# print(df)
# X, Y = df.iloc[:,2:].values, df['Survived'].values
X = df.iloc[:,1:]
# X.to_csv('aa.csv')
# print(X.info())
# # #读取Model
# Y = pd.read_csv('./data/gender_submission.csv')
with open('./data/model.pickle', 'rb') as f:
    model = pickle.load(f)
Y = pd.DataFrame(list(range(892,892+418,1)),columns = ['PassengerId',])
Y['Survived'] = model.predict(X)
# print(Y[0:2])
Y.to_csv('./data/gender_submission.csv')
    
# print(X,Y)
##---------------------------------------------------
##         选择模型
##---------------------------------------------------
# X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
# model = linear_model.RidgeClassifierCV()
# model = KNeighborsClassifier()
# model = SVC(C=5,kernel='rbf')
# model = AdaBoostClassifier()
# model = RandomForestClassifier(criterion='gini')
# estimator = AdaBoostClassifier(model,learning_rate=0.001,n_estimators=10)
# estimator = BaggingClassifier(model,n_estimators=10,max_samples=700, max_features=8)
# model.fit(X_train,Y_train)
# model.fit(X_train,Y_train)
# print(model.score(X_test,Y_test))
# a = model.predict(X_test)
# print(a[0:20])
# print(cross_val_score(model,X,Y,cv=5).mean())
# model.fit(X,Y)
##---------------------------------------------------
##         保存模型
##---------------------------------------------------
# with open('./data/model.pickle', 'wb') as f:
#     pickle.dump(model, f)

