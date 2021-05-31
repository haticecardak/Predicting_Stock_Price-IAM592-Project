import numpy as np 
import pandas as pd

X=pd.read_csv('../data/SILVER_Histdata.csv')

class SilverData():
    def __init__(self):
        self.folderName='../data/SILVER_Histdata.csv'
        self.data=pd.read_csv(self.folderName)
        self.columns=self.data.columns
  
    def prepareTimeSeriesData(self,model='regression',time=30):
       
        X=np.zeros((self.data.shape[0]-time,time))
        y=np.zeros((X.shape[0],1))
        
        for i in range(y.shape[0]):
            X[i,:]=self.data['Open'][i:i+time]
            if method=='regression':
                y[i]=self.data['Open'][i+time]
                
        return np.flip(X,axis=0),np.flip(y,axis=0)
        
    def getCloseData(self):
        return np.flip(self.data['Open'].values)
     
si=SilverData()

X,y=si.prepareTimeSeriesData(model='regression',time = 30)
