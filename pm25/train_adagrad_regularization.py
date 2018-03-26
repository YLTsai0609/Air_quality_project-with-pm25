#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
x = np.load('x_train.npy')
y = np.load('y_train.npy')
#增加一行1 對齊bias term
x = np.concatenate((np.ones((x.shape[0], 1)), x),axis=1)
#print(x.shape, y)
#regularization
Lambda = [1000,100,10,1, 0.1, 0.01,0.001,0.0001]
#print(len(Lambda))
for j in range(len(Lambda)):
#initial condition
	w = np.zeros((len(x[0])))
	lr = 1
	iteration = 700
	accum_square_grad = np.zeros((len(x[0])))

#train 
	for i in range(iteration):
		socre = np.dot(x, w)
		loss = y - socre 
		grad = np.dot(x.transpose(),loss)*(-2) + 2*Lambda[j]*(w**2)  #Loss = MSE , L1 Regularizer 
		accum_square_grad += grad**2
		ada = np.sqrt(accum_square_grad)
		w = w - lr*grad/ada
		np.save("model_lambda_"+str(Lambda[j])+".npy", w)
#print(loss, loss.shape, grad, grad.shape)