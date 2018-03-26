#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np 
#提取pm2.5為x
test = pd.read_csv("./test_X.csv") 
x_test = test[test['AMB_TEMP']=='PM2.5'].iloc[:, 2:]
x_test = np.array(x_test,'float32')
#print(x_test.shape)
#print(x_test)
#增加一行1 對齊bias term
x_test_b = np.concatenate((np.ones((x_test.shape[0], 1)), x_test),axis=1)
#print(x_test_b.shape)
#預測
w = np.load('model_best.npy')
#print(x_test_b, w.shape)
y_hat = np.dot(x_test_b, w)
y_pre = pd.read_csv("./sampleSubmission.csv")
y_pre.value = y_hat
y_pre.to_csv("Submission3.csv")

