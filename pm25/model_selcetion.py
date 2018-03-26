#!/usr/bin/python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np  
#讀取 訓練/驗證 資料
x_val = np.load('x_validation.npy')
x_train = np.load('x_train.npy')
y_val = np.load('y_validation.npy')
y_train = np.load('y_train.npy')
#增加一行1 對齊bias term
x_val = np.concatenate((np.ones((x_val.shape[0], 1)), x_val),axis=1)
x_train = np.concatenate((np.ones((x_train.shape[0], 1)), x_train),axis=1)
#計算Ein,Eval
Lambda = [1000,100,10,1, 0.1, 0.01,0.001,0.0001]
Eval = []
Ein = []
for j in range(len(Lambda)):
    w = np.load("model_lambda_"+str(Lambda[j])+".npy")
    y_hat_val = np.dot(x_val, w)
    y_hat_train = np.dot(x_train, w)
    err1 = abs(y_val - y_hat_val).sum()/len(y_val)
    err2 = abs(y_train - y_hat_train).sum()/len(y_train)
    Eval.append(err1)
    Ein.append(err2)
print(Eval)
print("min of Eval :", np.min(np.array(Eval)))

#plot
Eval = np.array(Eval)
Ein = np.array(Ein)
Lambda = np.array(Lambda)
plt.plot(Lambda, Eval, label=u'Eval')
plt.plot(Lambda, Ein, label=u'Ein')
plt.legend()
plt.xlabel("Lambda")
plt.ylabel("ave_abs_Error")
plt.ylim(4.5, 4.65)
plt.xscale('log')
plt.show()
#結論為lambda = 0.001 時驗證資料錯誤最低
