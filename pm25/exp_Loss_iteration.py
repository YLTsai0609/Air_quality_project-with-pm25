import matplotlib.pyplot as plt
import numpy as np


x = np.load('x_train.npy')
y = np.load('y_train.npy')

# loding data 


w = np.zeros((len(x[0])))
lr = 1
iteration = 700
accum_square_grad = np.zeros((len(x[0])))
iter = []
loss_func = []

#print(w, accum_square_grad)

#train 
for i in range(iteration):
	socre = np.dot(x, w)
	loss = y - socre 
	grad = np.dot(x.transpose(),loss)*(-2)  #Loss = (y-yhat)**2
	accum_square_grad += grad**2
	ada = np.sqrt(accum_square_grad)
	w = w - lr*grad/ada
	iter.append(i)
	loss_func.append(np.sqrt((loss**2).sum()))
	#iter = np.array(iter, float)
	#loss_func = np.array(loss_func, float)
	#print(loss_func.shape)

#plot
plt.plot(iter, loss_func)

plt.xlim(0, iteration)
plt.ylim(0, 1000)
plt.xlabel("iteration")
plt.ylabel("Loss funciotn")
plt.title("Loss function vs iteration")
plt.show()
#經實驗測試 loss function在 550 iterations之後穩定的收斂