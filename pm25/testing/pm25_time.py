#!/usr/bin/python3
# -*- coding: utf-8 -*-
import time
import pandas as pd
import numpy as np 

train = pd.read_csv("./train.csv") #載入檔案 排除亂碼問題
pm25 = train[train['Observation']=='PM2.5'].drop(['Date', 'Location','Observation'], axis=1)
#print(pm25.head(3)) #檢查點
pm25 = pm25.stack().values
#print(pm25, len(pm25)-9)

x = [] #x y此時為list
y = []
tStart = time.time()
for i in range(len(pm25)-9):
	temx = pm25[i:i+9]
	temy = pm25[i+9]
	x.append(temx)
	y.append(temy)
x = np.array(np.concatenate(x).reshape((len(pm25)-9, 9)),float) #x,y為nd_array
y = np.array(y,float)
tEnd = time.time()
print(x, y, tEnd - tStart)
