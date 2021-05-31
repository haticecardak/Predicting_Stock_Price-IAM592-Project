import numpy as np 
import pandas as pd
import os

X2=pd.read_csv('../dataset/ASELS_Histdata.csv')

class AselsanData():

    def __init__(self):
        self.folderName='../dataset/ASELS_Histdata.csv'
        self.data=pd.read_csv(self.folderName)
        self.columns=self.data.columns
     
    def TimeSeriesData(self,model='regression',time=60):
    
    # 60 days historical data used to estimate 61th day price 
    # 30 days also will be checked the change in price 
    
        X2=np.zeros((self.data.shape[0]-time,time))
        y=np.zeros((X2.shape[0],1))
        
        for i in range(y.shape[0]):
            X2[i,:]=self.data['Open'][i:i+time] # open price is considered 
            if method=='regression':
                y[i]=self.data['Open'][i+time] 
              
asl=AselsanData()
X2,y=asl.TimeSeriesData(model='regression',time = 60)
