import matplotlib.pyplot as plt
import numpy as np
# loding data 
x1 = np.arange(0,360)
y1 = np.sin(x1 * np.pi / 180.0)
y2 = np.cos(x1 * np.pi / 180.0 )
#print(x.shape, y.shape)
#plot
plt.plot(x1, y1, label=u'y=sin(x)')
plt.plot(x1, y2, label=u'y=cos(x)')
plt.legend()
'''
plt.xlim(-30, 390)
plt.ylim(-1.5, 1.5)
'''
plt.xlabel("x")
plt.ylabel("y")
#plt.title("y = sin(x)")
plt.show()

