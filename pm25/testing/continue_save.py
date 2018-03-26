import numpy as np

for i in range(3):
	x = i 
	print(x)
	np.save("test"+str(x)+".npy", np.array(x))
k = np.load('test2.npy')
print(k)