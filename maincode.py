import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd

from matplotlib.figure import Figure, figaspect

from ASELSAN import ASELSANDataset
from SILVERData import SilverData
from GoldDataset import GoldDataset
from NFLXData import NetFlixData

aselsan=ASELSANDataset()
silver=SilverData()
golden=GoldDataset()
netflix=NetFlixData()
data={'Gold':golden, 'Aselsan': aselsan, 'Silver':silver, 'NetFlix': netflix}

def splitDataset(X,y,ratio=0.9): # ratio=0.9 the set of data %90: train , %10 test
    test=np.floor(X.shape[0]*ratio).astype('int')
    print('Train Samples : {} over all Samples : {}'.format(test,X.shape[0]))
    xtrain=X[:test,:]
    xtest=X[test:,:]
    ytrain=y[:test,:]
    ytest=y[test:,:]
    return xtrain,ytrain,xtest,ytest
def calculateLoss(y,ypred):
    return np.mean((ypred-y)**2)

#--------------------


for key in data.keys():
    print('Key : {} , Samples : {}'.format(key,data[key].data.shape[0]))


#--Linear Regression 
  
  lrOLS=LinearReg(standardize=True,method='OLS',lam=0)
  lrRidge=LinearReg(standardize=True,method='Ridge',lam=10)
  
    for key in data.keys():
        
        X,y = data[key].prepareTimeSeriesData(model='regression',time=30)
        xtrain,ytrain,xtest,ytest = splitDataset(X,y,ratio=0.9)
          xtrain,ytrain,xtest,ytest = splitDataset(X,y,ratio=0.9)
        sTime=time.time() #measure the time from the time library
        lrOLS.fit(xtrain,ytrain)
        print('OLS Training ({} Samples) Time  : {:.3f} seconds'.format(xtrain.shape[0],time.time()-sTime))
        ypred=lrOLS.predict(xtest)
        
        print('{} OLS test MSE : {}'.format(key,np.mean((ypred-ytest)**2)))
        plt.figure()
        plt.plot(ytest)
        plt.plot(ypred)
        plt.title('{} Test Data OLS Estimation'.format(key))
        plt.savefig('../images/{} OLS Estimation '.format(key))
         
        plt.show()
        
        sTime=time.time() 
        lrRidge.fit(xtrain,ytrain)
        print('Ridge Training ({} Samples) Time  : {:.3f} seconds'.format(xtrain.shape[0],time.time()-sTime))
        ypred=lrRidge.predict(xtest)
        
        print('{} Ridge test MSE : {:.3f}'.format(key,np.mean((ypred-ytest)**2)))
        plt.figure()
        plt.plot(ytest)
        plt.plot(ypred)
        plt.title('{} Test Data Ridge Estimation'.format(key))
        plt.savefig('../images/{} Ridge Estimation '.format(key))
        plt.show()

