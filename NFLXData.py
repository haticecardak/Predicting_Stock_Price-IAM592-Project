
import numpy as np 
import pandas as pd
import os

X=pd.read_csv('../dataset/NFLX_Histdata.csv')

class NetFlixData():
    def __init__(self):
        self.folderName='../dataset/NFLX_Histdata.csv'
        self.data=pd.read_csv(self.folderName)
        self.columns=self.data.columns
  
    def prepareTimeSeriesData(self,method='regression',time): 
     
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
    
nflx=NetFlixData()
X,y=nflx.prepareTimeSeriesData(method='regression',time = 30)
