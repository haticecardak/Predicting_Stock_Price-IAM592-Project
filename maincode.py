import matplotlib.pyplot as plt
import time
import numpy as np
import time
from matplotlib.figure import Figure, figaspect

from ASELSAN import ASELSANDataset
from SILVERData import SilverData
from GoldDataset import GoldDataset

asl=ASELSANDataset()
si=SilverData()
gd=GoldDataset()

data={'Gold':gd, 'Aselsan': asl, 'Silver':si}

#--Linear Regression 
 
   for key in data.keys():
        
        X,y = data[key].prepareTimeSeriesData(method='regression',timeWindow=30)
