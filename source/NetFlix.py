# -*- coding: utf-8 -*-
"""
Created on Sun May 16 18:02:45 2021

@author: hatice.cardak
"""

import numpy as np 
import pandas as pd

X=pd.read_csv('../data/NFLX_Histdata.csv')
X.columns = ["Date","Open","High","Low","Close","Adj Close","Volume"]
X.head()
print(X)

class NetFlixDataset():
    def __init__(self):
        self.folderName='../data/NFLX_Histdata.csv'
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
    
    

print('\n Open Prices: \n' ,np.flip(X['Open'].values))
    
netflix=NetFlixDataset()
X,y=netflix.prepareTimeSeriesData(method='regression',time = 30)

print(' \n Regression Time Data set: \n',X,
      '\n Open Price Data set: \n',y)


X2,y2=netflix.prepareTimeSeriesData(method='classification',time = 30)
print('\n Classification Time Data Set: \n', X2,
      '\n Predicted Prices: \n', y2)

print('\n The average of the predicted price data set: ',(y2>1.01).mean())
