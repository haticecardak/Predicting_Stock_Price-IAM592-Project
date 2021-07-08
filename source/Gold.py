
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

X=pd.read_csv('../data/Gold_Histdata.csv')
X.columns = ["Date" , "Price","Open","High","Low","Vol","Change %"]
X.head()
print(X)

class GoldDataset():
    
    def __init__(self):
        
        self.folderName='../data/Gold_Histdata.csv'
        self.data=pd.read_csv(self.folderName)
        self.columns=self.data.columns
        
       
        for column in ['Price','High']: #read only Price and High Column from the data set
            self.data[column]=self.data[column].str.replace(',','' ).astype(float) # replacing comma with space(" ") and making these column float      
    
    def prepareTimeSeriesData(self,method='regression', time=30):
            
        '''
        Parameters
        ----------
        X : feature matrix
        y : target
        
        Defining method that refers to regression and classification. 
        
        The aim of the regression is to forecast 31th day price by using past 30 days data.
        
        The aim of classification  could the 31st day price be gaining %1 profit?
        
        '''   
        X=np.zeros((self.data.shape[0]-time,time))
        y=np.zeros((X.shape[0],1))
        
        for i in range(y.shape[0]):
            
            X[i,:]=self.data['Price'][i:i+time]
            
            if method=='regression':
                y[i]=self.data['Price'][i+time]
                
            if method=='classification':
                y[i]=self.data['High'][i+time]/self.data['Price'][i+time-1]
        
        return np.flip(X,axis=0),np.flip(y,axis=0)
     
  
    def getCloseData(self):
        return np.flip(self.data['Price'].values)
    
    def calculateRSI(self,x,n):
         
        rsi=50*np.ones(x.shape)
        
        for i in range(n,x.shape[0]):
            diff=(np.diff(x[i-n-1:i]))
            
            pos=np.sum(diff[diff>0])/n
            neg=-np.sum(diff[diff<0])/n
            rsi[i]=100-100/(1+pos/neg)
            
        return rsi



print('\n Open Prices: \n' ,np.flip(X['Price'].values))
gold=GoldDataset()
X,y=gold.prepareTimeSeriesData(method='classification',time = 30) 



goldPrice=gold.getCloseData()
rsiGold=gold.calculateRSI(goldPrice,14)
price=gold.getCloseData()
print('RSI :', rsiGold)


# plot correspondingRSI values and significant levels
plt.figure(figsize=(16,6))
ax1=plt.subplot(211)#produces subaxes command
ax1.set_title('Prices')
ax1.plot(goldPrice)
ax2=plt.subplot(212,sharex=ax1)
ax2.set_title('RSI')
ax2.plot(rsiGold)

plt.axhline(0, linestyle='--', alpha=0.1)
plt.axhline(20, linestyle='--', alpha=0.5)
plt.axhline(30, linestyle='--')

plt.axhline(70, linestyle='--')
plt.axhline(80, linestyle='--', alpha=0.5)
plt.axhline(100, linestyle='--', alpha=0.1)
plt.show()

