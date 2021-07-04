import numpy as np 
import pandas as pd

X2=pd.read_csv('../dataset/ASELS_Histdata.csv')
X2.columns = ["Tarih","Şimdi","Açılış","Yüksek","Düşük","Hacim", "Fark %"] #coloumn headings
X2.head() #shown first and last 5 data

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
                y[i]=self.data['High'][i+time]/self.data['Open'][i+time-1] 
                
        # reverse order the data set to be shown as descending day            
        return np.flip(X,axis=0),np.flip(y,axis=0)
    
    # Open prices is listed by 30rd day to 1st day    
    def getCloseData(self):
        
        return np.flip(self.data['Open'].values) 

print('\n Open Prices: \n' ,np.flip(X['Open'].values))
    
aselsan=ASELSANDataset()
X,y=aselsan.prepareTimeSeriesData(method='classification',time = 30)
print('\n Classification Time Data Set: \n', X,
      '\n Predicted Prices: \n', y)
X2,y2=aselsan.prepareTimeSeriesData(method='regression',time = 30)
print(' \n Regression Time Data set: \n',X2,
      '\n Open Price Data set: \n',y2)
print('\n The average of the predicted price data set: ',(y>1.01).mean()) #predicted price of the data set average
