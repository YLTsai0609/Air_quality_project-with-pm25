import pandas as pd
import numpy as np 
'''
讀取csv檔案 utf-8-sig解碼
其他解決讀取亂碼方法 https://code.i-harness.com/zh-TW/q/2040ba5 
google 'pandas讀取csv亂碼' 

train = pd.read_csv("./train.csv",encoding='utf-8-sig')
train1 = train.drop(['Date', 'Location'], 1)
'''
  #丟掉Date及Location那行
#print(train.head()) #顯示前面的區塊
#print(train.tail())  #顯示後面的區塊
#print(train.index,train.columns) #顯示index 橫的 直的
#print(train['Observation'],train[0:5]) #選出直的及橫的 0:5不包含5
#print(train['Observation'][0:3]) #多次選取
#print(train[:5]) #0可以不寫
#print(train[:-5]) #後面5列不要 其他寫出
df1 = pd.DataFrame(np.random.randn(6,5),columns=list('ABCDE'))
'''
k =[]
for i in range(2):
	k1 = df1.iloc[i:i+1,:].values
	k.append(k1)

print(k) #stack 堆疊 但是index會重新排序
'''
df2 = df1.stack().values
#df2 = np.array(range(30))
print(df2.shape)
#http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-label pandas document