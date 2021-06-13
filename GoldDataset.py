import numpy as np 
import pandas as pd

X=pd.read_csv('../data/Gold Futures Historical Data.csv')

class GoldDataset():
    
    def __init__(self):
        
        self.folderName='../dataset/Gold Futures Historical Data.csv'
        self.data=pd.read_csv(self.folderName)
        self.columns=self.data.columns
        
    def prepareTimeSeriesData(self,model='regression',time=30):
        
        
        X=np.zeros((self.data.shape[0]-time,time))
        y=np.zeros((X.shape[0],1))
        
        for i in range(y.shape[0]):
            X[i,:]=self.data['Price'][i:i+time]
            if method=='regression':
                y[i]=self.data['Price'][i+time]
