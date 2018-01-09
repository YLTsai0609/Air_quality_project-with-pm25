import numpy as np
x = int(input("請問要取出哪一行"))

#讀取檔案並取出 x 行排序
f = open('hw0_data.dat')
data = np.loadtxt(f,usecols= (x))
data = np.sort(data)
#寫入文件
f1 = open("Ans1.txt","w")
data = data.tolist()
f1.write(str(data))
f1.close()