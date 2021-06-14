import pandas as pd
import numpy as np 


X=pd.read_csv('../data/Gold_Histdata.csv')
X.columns = ["Date" , "Price","Open","High","Low","Vol.","Change "]
X.head()
print(X)
class GoldDataset():
    
    def __init__(self):
        
        self.folderName='../data/Gold_Histdata.csv'
        self.data=pd.read_csv(self.folderName)
        self.columns=self.data.columns
        
       
        for column in ['Price','High']: #read only Price and High Column from the data set
            self.data[column]=self.data[column].str.replace(',','' ).astype(float) # replacing comma with space(" ") and making these column float      
    
    def prepareTimeSeriesData(self,method, time):
            
    # '''  
    # Parameters
    #     ----------
    #     X : feature matrix
    #     y : target
        
    #     Defining method that refers to regression and classification. 
        
    #     The aim of the regression is to forecast 31th day price by using past 30 days data.
        
    #     The aim of classification  could the 31st day price be gaining %1 profit?
    #     Returns
    #     -------  
    # '''           
        X=np.zeros((self.data.shape[0]-time,time))
        y=np.zeros((X.shape[0],1))
        
        for i in range(y.shape[0]):
            X[i,:]=self.data['Price'][i:i+time]
            
            if method=='regression':
                y[i]=self.data['Price'][i+time]
                
            if method=='classification':
                #31st day high price is divided into 30th day closed price
                y[i]=self.data['High'][i+time]/self.data['Price'][i+time-1]
        
        return np.flip(X,axis=0),np.flip(y,axis=0)
    
  
    def getCloseData(self):
        return np.flip(self.data['Price'].values)

gold=GoldDataset()
X,y=gold.prepareTimeSeriesData(method='classification',time= 30) 

