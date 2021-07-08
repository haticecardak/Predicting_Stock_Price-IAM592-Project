# -*- coding: utf-8 -*-
"""
Created on Sat May 22 19:13:44 2021

@author: hatice.cardak
"""


import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd


from matplotlib.figure import *

from ASELSAN import ASELSANDataset
from SILVERData import SilverData
from Gold import GoldDataset
from NetFlix import NetFlixDataset
from LinearReg import LinearReg


def splitDataset(X,y,ratio=0.9): # ratio=0.9 the set of data %90: train , %10 test
    test=np.floor(X.shape[0]*ratio).astype('int')
    print('Train Samples : {} over all Samples : {}'.format(test,X.shape[0]))
    
 # data set is divided into two parts train and test data set
    xtrain=X[:test,:]
    xtest=X[test:,:]
    ytrain=y[:test,:]
    ytest=y[test:,:]
    return xtrain,ytrain,xtest,ytest
    print(xtrain)
    
    
def calculateLoss(y,ypred):
    return np.mean((ypred-y)**2)
        
gold=GoldDataset()
aselsan=ASELSANDataset()
silver=SilverData()
netflix=NetFlixDataset()


data={'Aselsan': aselsan, 'NetFlix': netflix, 'Silver': silver, 'Gold':gold} # definig dictionary and elements are called as a key
# %%%%%%%%%%%%%

for key in data.keys():
    print('Key : {} , Samples : {}'.format(key,data[key].data.shape[0]))


#%% Linear Regression

if True:
    lrOLS=LinearReg(standardize=True,method='OLS',lam=0)
    lrRidge=LinearReg(standardize=True,method='Ridge',lam=10) # lambda is defined as a 10 for testing the code 
    
    for key in data.keys(gold):
        
        X,y = data[key].prepareTimeSeriesData(method='regression',time=30) #calling preparetimeseries for using data set.
        
        xtrain,ytrain,xtest,ytest = splitDataset(X,y,ratio=0.9)

        #measure the time from the time library calculating the method for example OLS method fitting time
        lrOLS.fit(xtrain,ytrain)
        print('OLS Training ({} Samples) Time  : {:.3f} seconds'.format(xtrain.shape[0],time.time()-sTime))
        ypred=lrOLS.predict(xtest)
        
        print('{} Ridge test OLS : {:.3f}'.format(key,np.mean((ypred-ytest)**2)))
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
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from RLAgentEnv import *

# for key in data.keys():
for key in ['BTC']:
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
                 print('Sell Reward : {:.3f}'.format(reward))
                
    print('{} Q Table'.format(key))
    print(rlAgent.Qtable) 
    
    plt.figure()
    plt.plot(np.sum(rewards,axis=1))
    plt.title('{} Rewards Per Episode'.format(key))
    plt.show(block=False)
      
rsiGold=gd.calculateRSI(goldPrice,14)
rsiGold[np.isnan(rsiGold)]=50

# plt.figure()
# ax1=plt.subplot(211)
# ax1.plot(goldPrice)
# ax2=plt.subplot(212,sharex=ax1)
# ax2.plot(rsiGold)

# plt.show()
