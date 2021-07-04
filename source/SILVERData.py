# -*- coding: utf-8 -*-
"""
Created on Sun May 16 18:07:27 2021

@author: hatice.cardak
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

X=pd.read_csv('../data/SILVER_Histdata.csv')
X.columns = ["Date","Open","High","Low","Close","Adj Close","Volume"]
X.head()
print(X)

class SilverData():
    def __init__(self):
        self.folderName='../data/SILVER_Histdata.csv'
        self.data=pd.read_csv(self.folderName)
        self.columns=self.data.columns
  
    def prepareTimeSeriesData(self,method='regression',time=30):
        # a=
        X=np.zeros((self.data.shape[0]-time,time))
        y=np.zeros((X.shape[0],1))
        
        for i in range(y.shape[0]):
            X[i,:]=self.data['Open'][i:i+time]
            if method=='regression':
                y[i]=self.data['Open'][i+time]
            if method=='classification':
                y[i]=self.data['High'][i+time]/self.data['Open'][i+time-1]
                
                
        return np.flip(X,axis=0),np.flip(y,axis=0)
        
    def getCloseData(self):
        return np.flip(self.data['Open'].values)
    def calculateRSI(self,x,n):
         
        rsi=50*np.ones(x.shape)
        
        for i in range(n,x.shape[0]):
            diff=(np.diff(x[i-n-1:i]))
            
            pos=np.sum(diff[diff>0])/n
            neg=-np.sum(diff[diff<0])/n
            rsi[i]=100-100/(1+pos/neg)
            
        return rsi

###########################
print('\n Open Prices: \n' ,np.flip(X['Open'].values))
silver=SilverData()
X,y=silver.prepareTimeSeriesData(method='classification',time = 30)
X2,y2=silver.prepareTimeSeriesData(method='regression',time = 30) 
print('\n The average of the predicted price data set: ',(y>1.01).mean()) #predicted price of the data set average

print('\n Classification Time Data Set: \n', X,
      '\n Predicted Prices: \n', y)

print(' \n Regression Time Data set: \n',X2,
      '\n Open Price Data set: \n',y2)
############################

rsiSilver=silver.getCloseData()
rsiSilver=silver.calculateRSI(rsiSilver,14)
price=silver.getCloseData()
print('RSI :', rsiSilver)

plt.figure(figsize=(15,5))
plt.title('RSI')
ax1=plt.subplot(211)
ax1.plot(rsiSilver)

ax2=plt.subplot(212,sharex=ax1)
ax2.plot(rsiSilver)


plt.axhline(0, linestyle='--', alpha=0.1)
plt.axhline(20, linestyle='--', alpha=0.5)
plt.axhline(30, linestyle='--')

plt.axhline(70, linestyle='--')
plt.axhline(80, linestyle='--', alpha=0.5)
plt.axhline(100, linestyle='--', alpha=0.1)
plt.show()