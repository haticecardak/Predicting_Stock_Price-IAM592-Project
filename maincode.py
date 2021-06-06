import matplotlib.pyplot as plt
import time
import numpy as np
import time
from matplotlib.figure import Figure, figaspect

from ASELSAN import ASELSANDataset
from SILVERData import SilverData
from GoldDataset import GoldDataset
from NFLXData import NetFlixData

aselsan=ASELSANDataset()
si=SilverData()
gd=GoldDataset()
nflx=NetFlixData()
data={'Gold':gd, 'Aselsan': aselsan, 'Silver':si, 'NetFlix': nflx}

def splitDataset(X,y,ratio=0.9): # ratio=0.9 the set of data %90: train , %10 test
    test=np.floor(X.shape[0]*ratio).astype('int')
    print('Train Samples : {} over all Samples : {}'.format(test,X.shape[0]))
    xtrain=X[:test,:]
    xtest=X[test:,:]
    ytrain=y[:test,:]
    ytest=y[test:,:]
    return xtrain,ytrain,xtest,ytest
#--------------------


for key in data.keys():
    print('Key : {} , Samples : {}'.format(key,data[key].data.shape[0]))


#--Linear Regression 
 
   for key in data.keys():
        
        X,y = data[key].prepareTimeSeriesData(model='regression',time=30)
        xtrain,ytrain,xtest,ytest = splitDataset(X,y,ratio=0.9)
