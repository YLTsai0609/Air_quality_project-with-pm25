#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
x = np.load('x_final_train.npy')
y = np.load('y_final_train.npy')
#增加一行1 對齊bias term
x = np.concatenate((np.ones((x.shape[0], 1)), x),axis=1)
#print(x.shape, y)
#initial condition
w = np.zeros((len(x[0])))
lr = 1
iteration = 700
accum_square_grad = np.zeros((len(x[0])))

#print(w, accum_square_grad)

#train 
#model_selection最佳結果為lambda = 0.0001
for i in range(iteration):
	socre = np.dot(x, w)
	loss = y - socre 
	grad = np.dot(x.transpose(),loss)*(-2) +  2*(0.0001)*(w**2)  #Loss = MSE
	accum_square_grad += grad**2
	ada = np.sqrt(accum_square_grad)
	w = w - lr*grad/ada
np.save("model_best.npy", w)