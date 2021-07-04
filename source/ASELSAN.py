# -*- coding: utf-8 -*-
"""
Created on Sat May 22 19:09:43 2021

@author: hatice.cardak
"""

import numpy as np 
import pandas as pd

X=pd.read_csv('../data/ASELS_Histdata.csv')
X.columns = ["Date","Open","High","Low","Close","Adj Close", "Volume"]
X.head() # first  and last 5 data are shown 
print(X)

class ASELSANDataset():
    
    def __init__(self):
        
        self.folderName='../data/ASELS_Histdata.csv'
        self.data=pd.read_csv(self.folderName)
        self.columns=self.data.columns
        
    def prepareTimeSeriesData(self, method,time):
        
        '''
        This method prepares time series data
        Input Arguments:
        Time: 30
        Method: Regression and Classification
        X= Feature Matrix
        y=target Matrix
        
        Regression method aims to predict 31st day data with respect to 30 days data 
        
        ***********
        
        Classification method aims to predict 31st day price that will be %1 higher than previous day Open Price
        this method takes into consideration with respect to 31st day High price over the 30rd day Open price.
        If the price is higher than previous day price by 1.01, buying stock will be good decision.         
        '''
        self.time=30
        self.method=['regression', 'classification']
                
        X=np.zeros((self.data.shape[0]-time,time))
        y=np.zeros((X.shape[0],1))
        
        for i in range(y.shape[0]):
            
            X[i,:]=self.data['Open'][i:i+time]
            
            if method=='regression':
                y[i]=self.data['Open'][i+time]
                
            if method=='classification':
                y[i]=self.data['High'][i+time]/self.data['Open'][i+time-1] # looking at 1.01 rise in prices 
                
        # reverse order the data set to be shown as descending day            
        return np.flip(X,axis=0),np.flip(y,axis=0)
    
    # Open prices is listed by 30rd day to 1st day    
    def getCloseData(self):
        
        return np.flip(self.data['Open'].values) 
    def calculateRSI(self,x,n):
        
        rsi=50*np.ones(x.shape)
        
        for i in range(n,x.shape[0]):
            diff=np.diff(x[i-n-1:i])
            
            pos=np.sum(diff[diff>0])/n
            neg=-np.sum(diff[diff<0])/n
            
            rsi[i]=100-100/(1+pos/neg)
            
        return rsi

print('\n Open Prices: \n' ,np.flip(X['Open'].values))
    
aselsan=ASELSANDataset()
X,y=aselsan.prepareTimeSeriesData(method='classification',time = 30)
print('\n Classification Time Data Set: \n', X,
      '\n Predicted Prices: \n', y)
X2,y2=aselsan.prepareTimeSeriesData(method='regression',time = 30)
print(' \n Regression Time Data set: \n',X2,
      '\n Open Price Data set: \n',y2)
print('\n The average of the predicted price data set: ',(y>1.01).mean()) #predicted price of the data set average