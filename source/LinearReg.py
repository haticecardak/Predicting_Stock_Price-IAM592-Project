
import numpy as np

class LinearReg():
    
    def __init__(self,standardize,method,lam=0):
        self.standardize=standardize
        self.method=method
        self.lam=lam
        
    def fit(self,X,y,B):
        
        '''
        
        Parameters
        ----------
        X : feature
        y : target
        This object makes the model fit by using given data and find beta.
        If we need to standardize the data, it would hold the data mean and standard deviation

        Returns
        -------
        None.

        '''
        if self.standardize:
            self.mx=X.mean(axis=0) # calculate the average of X matris axis=0 means that only rows average taking into considiretion
            self.sx=X.std(axis=0) #calculate the standard deviation of the X matrix
            
            X=(X-self.mx)/self.sx #normalization
            
            self.my=y.mean() #calculate the target mean
            y=y-self.my #make centeralization
            
        if self.method=='OLS': #ordinary least square
            self.B=np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y)) # beta coefficient of i nth day 
        
        if self.method=='Ridge':
            self.B=np.dot(np.linalg.inv(np.dot(X.T,X)+self.lam*np.eye(X.shape[1])),np.dot(X.T,y))
        
    def predict(self,X):
        if self.standardize:
            X=(X-self.mx)/self.sx #normalization 
        
            return self.my+ np.dot(X,self.B)
        
        else:
            return np.dot(X,self.B)
        
        
        