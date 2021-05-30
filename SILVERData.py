import numpy as np 
import pandas as pd

X=pd.read_csv('../data/SILVER_Histdata.csv')

class SilverData():
    def __init__(self):
        self.folderName='../data/SILVER_Histdata.csv'
        self.data=pd.read_csv(self.folderName)
        self.columns=self.data.columns
  
    def prepareTimeSeriesData(self,method='regression',timeWindow=30):
        # a=
        X=np.zeros((self.data.shape[0]-timeWindow,timeWindow))
        y=np.zeros((X.shape[0],1))
        
        for i in range(y.shape[0]):
            X[i,:]=self.data['Open'][i:i+timeWindow]
            if method=='regression':
                y[i]=self.data['Open'][i+timeWindow]
            if method=='classification':
                y[i]=self.data['High'][i+timeWindow]/self.data['Open'][i+timeWindow-1]
                
                
        return np.flip(X,axis=0),np.flip(y,axis=0)
        
    def getCloseData(self):
        return np.flip(self.data['Open'].values)
     
si=SilverData()

X,y=si.prepareTimeSeriesData(method='classification',timeWindow = 30)
