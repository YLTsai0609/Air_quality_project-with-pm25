#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np 

train = pd.read_csv('./train.csv') 
#提取pm2.5
pm25 = train[train['Observation']=='PM2.5'].drop(['Date', 'Location','Observation'], axis=1)
#print(pm25.head(3)) 
pm25 = pm25.stack().values
#print(pm25)

#製作inputs和labels
x = [] #x y此時為list
y = []
for i in range(len(pm25)-9):
	temx = pm25[i:i+9]
	temy = pm25[i+9]
	x.append(temx)
	y.append(temy)
x = np.array(np.concatenate(x).reshape((len(pm25)-9, 9))).astype('float32') #x,y為nd_array
y = np.array(y).astype('float32')
#print(x)
#print(y)
#切分資料 驗證資料1/5
x_train = x[:4*len(x)//5 + 1]
#x_std_train = x_standardization[:4*len(x)//5 + 1]
y_train = y[:4*len(y)//5 + 1]
x_validation =  x[-(len(x)//5):]
#x_std_validation= x_standardization[-(len(x)//5):]
y_validation = y[-(len(x)//5):]
x_final_train = x
#x_std_final_train = x_standardization
y_final_train = y

#存取檔案
np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)
np.save("x_validation.npy", x_validation)
np.save("y_validation.npy",y_validation)
np.save("x_final_train.npy", x_final_train)
np.save("y_final_train.npy", y)
