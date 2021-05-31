import matplotlib.pyplot as plt
import time
import numpy as np
import time
from matplotlib.figure import Figure, figaspect

from ASELSAN import ASELSANDataset
from SILVERData import SilverData
from GoldDataset import GoldDataset
from NFLXData import NetFlixData

asl=ASELSANDataset()
si=SilverData()
gd=GoldDataset()
nflx=NetFlixData()
data={'Gold':gd, 'Aselsan': asl, 'Silver':si, 'NetFlix': nflx}

 train_x=X[:testInd,:]
 train_y=y[:testInd,:]
 test_x=X[testInd:,:]
 test_y=y[testInd:,:]


for key in data.keys():
    print('Key : {} , Samples : {}'.format(key,data[key].data.shape[0]))


#--Linear Regression 
 
   for key in data.keys():
        
        X,y = data[key].prepareTimeSeriesData(model='regression',time=30)
