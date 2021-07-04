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
#------------RLAgentEnv----------------
from RLAgentEnv import *

for key in data.keys():
    price=data[key].getCloseData()
    rlEnv=Env(price)
    
        rlAgent=RLAgent()
    numEpisode=1000
    rewards=np.zeros((numEpisode,price.shape[0]))
    for episode in range(numEpisode):
        if (1+episode)%100==0:
            print('{} Episode {} Completed'.format(key,episode+1))
        rlAgent.resetStates()
        rlAgent.setEps(np.maximum(0.5/(1+episode),0.001))#epsilon update
        for timeInd in range(price.shape[0]):
        # for timeInd in range(100):
        info=rlEnv.getInfo(timeInd)
            # print(info)
            # print(rlAgent.asset)
            asset=rlAgent.asset
            action=rlAgent.getAction(info)
            # print(action)
            # print(rlAgent.asset)
            bp=rlAgent.getBoughtPrice()
            # print(bp)
            reward=rlEnv.getReward(timeInd,action,bp,asset)
            rewards[episode,timeInd] = reward
            rlAgent.updateQtable(action,info,reward)
            if action=='sell':
                 # print('Sell Reward : {:.3f}'.format(reward))
                 
    # print('{} Q Table'.format(key))
    print(rlAgent.Qtable) 
    
    plt.figure()
    plt.plot(np.sum(rewards,axis=1))
    plt.title('{} Rewards Per Episode'.format(key))
    plt.show(block=False)
