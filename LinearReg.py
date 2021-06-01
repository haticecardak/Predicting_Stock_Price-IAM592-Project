import numpy as np
import pandas as pd
import statsmodels.api as sm

class LinearReg():
    
    def __init__(self,standardize,method,lam):
        self.standardize=standardize
        self.method=method
        self.lam=lam
        def fit(self,X,y):
        '''
        
        Parameters
        ----------
        X : feature matrix
        y : target
This object makes the model fit by using given data and find beta.
If we need to standardize the data, it would hold the data mean and standard deviation
        -----------

        '''
        if self.standardize:
            self.mx=X.mean(axis=0) # calculate the average of X matris axis=0 means that only rows average taking into considiretion
            self.sx=X.std(axis=0) #calculate the standar deviation of the X matrix
            
            X=(X-self.mx)/self.sx #normalization
            
            self.my=y.mean() #calculate the target mean
            y=y-self.my #make centeralization
            
