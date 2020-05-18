'''
@File    :   Titanic_survive_prediction.py
@Time    :   2020/05/18 15:51:27
@Author  :   Pan 
@Version :   1.0
@Contact :   pwd064@mail.ustc.edu.cn
@License :   (C)Copyright 2020-2025, USTC
@Desc    :   None
'''

import pandas as pd
import numpy  as np 
import matplotlib.pyplot as plt 
import math

##---------------------------------------------------
##               导入数据集
##---------------------------------------------------
df = pd.read_csv('./data/train.csv')
pclass = pd.pivot_table(df,values='Survived',index='Pclass')
plt.bar(pclass.index,np.reshape(pclass.values,(3,)))
plt.show()