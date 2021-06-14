import numpy as np

class Agent():
  def __init__(self,gamma=0.90,epsilon=0.1):
       self.gamma=gamma #with respect to rsi value, getting the how much reward i won and multiply the reward value with gamma then adding the result Q-table
       self.epsilon=epsilon # new decision reward multiplication rate 
       self.actions = ['buy', 'hold', 'sell']
       self.asset='cash'
       self.Qtable = np.zeros((10,len(self.actions)))
  
        
        if asset == 'cash':
            if action == 'buy':
                return -0.001 #komisyon alırken kesiliyor
            if action == 'hold':
                return 0
            if action == 'sell':
                return -1
        if asset == 'stock':
            if action == 'buy':
                return -1
            if action == 'hold':
                return 0
            if action == 'sell':
              
  def getAction(self,info): #info enviromenttn gelen bilgi price ve rsi değeri 
        
        p,rsi=info #aldığı p ve rsi değerine göre q-tableın hangi katmanına bakmasına karar veriyor
        stateInd=np.floor((rsi-0.001)/10).astype(int)
        
        if np.random.rand()<self.eps: #random hareketi %10 olasılıkla
            
            if self.asset=='cash': 
                action = self.actions[np.random.randint(2)]#2den küçük sayılardan birini seç demek
                if action=='buy':
                    self.asset='stock'
                    self.bought_price=p
                    self.boughtRSIind=stateInd
                    # print('Buy Order Price : {:.4f}'.format(p))
                # print('Random Selection : {}'.format(action))
                return action
            else:
                action = self.actions[np.random.randint(1,3)]
                if action=='sell':
                    self.asset='cash'
                    # print('Sell Order Price : {:.4f}'.format(p))
                # print('Random Selection : {}'.format(action))
                return action
        else:
            if self.asset=='cash':

                action = self.actions[np.argmax(self.Qtable[stateInd,:2])]
                if action=='buy':
                    self.asset='stock'
                    self.bought_price=p
                    self.boughtRSIind=stateInd
                    # print('Buy Order Price : {:.4f}'.format(p))
                return action
            else:
                action = self.actions[1+np.argmax(self.Qtable[stateInd,1:3])]
                if action=='sell':
                    self.asset='cash'
                    # print('Sell Order Price : {:.4f}'.format(p))
                return action
                
              
    def getBoughtPrice(self):
       self.bought_price=0
       return self.bought_price 
    
class Environment():  
    def __init__(self,price): #read price
            self.price=price
            self.rsi=self.calculateRSI(self.price,14)   # calculate the RSI with respect to 14 days    
    
    def calculateRSI(self,x,n):        
             rsi=50*np.ones(x.shape)
             for i in range(n,p):
               diff=np.diff(x[i-n-1:i])
                positive=np.sum(diff[diff>0])/n # son 14 günde artışların ortlaması
                negative=-np.sum(diff[diff<0])/n # son 14 günde azalışların ortalaması
                rs=positive/negative
                    if negative==0:
                       negative=0.001
                       rsi[i]=100-100/(1+rs)
                       rsi[np.isnan(rsi)]=50
                       rsi[rsi<1]=1
                       rsi[rsi>99]=99
                       return rsi
