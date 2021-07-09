
import numpy as np

class RLAgent():
    
    def __init__(self,gamma=0.95,eps=0.1): #agent tablo tutar Q-table
    #RSI şu kadarken ne kadar ödül aldım tutuyor eski değeri 0.95 ile çarpır reward ekler
       #Q-table update eden bir gama değeri var
       # epsilon yeni karar alma 0.1 exploration or explanation 
        self.gamma=gamma
        self.eps=eps
        self.actions = ['buy', 'hold', 'sell']
        self.Qtable = np.zeros((10,len(self.actions))) #RSI her 10luk dilimi içimi oluşturuldu
        self.asset='cash'
        self.bought_price=0
        self.boughtRSIind=0
    
    def resetStates(self):
        self.asset='cash'
        self.bought_price=0
        self.boughtRSIind=0
    
    def setEps(self,eps2):
        self.eps=eps2
    
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
                   
                return action
            else:
                action = self.actions[np.random.randint(1,3)]
                if action=='sell':
                    self.asset='cash'
                   
                return action
        else:
            if self.asset=='cash':

                action = self.actions[np.argmax(self.Qtable[stateInd,:2])]
                if action=='buy':
                    self.asset='stock'
                    self.bought_price=p
                    self.boughtRSIind=stateInd
                   
                return action
            else:
                action = self.actions[1+np.argmax(self.Qtable[stateInd,1:3])]
                if action=='sell':
                    self.asset='cash'
                   
                return action
    
    def getBoughtPrice(self):
        return self.bought_price
    
    def updateQtable(self, action,info,reward):
        p,rsi=info
        if rsi<0:
            rsi=0.1
        if rsi>100:
            rsi=99
        stateInd=np.floor((rsi-0.001)/10).astype(int)
        
        if action=='buy':
            actionInd=0
        if action=='hold':
            actionInd=1
        if action=='sell':
            actionInd=2
        
        self.Qtable[stateInd,actionInd]=reward + self.gamma*self.Qtable[stateInd,actionInd]
        if actionInd==2:
            self.Qtable[self.boughtRSIind,0]=reward + self.gamma*self.Qtable[self.boughtRSIind,0]
    

class Env():
    
    def __init__(self,price): #read price
        self.price=price
        self.rsi=self.calculateRSI(self.price,14)   # calculate the RSI with respect to 14 days      
        
    def getInfo(self,time):
        
        return self.price[time],self.rsi[time]
    
    
    def calculateRSI(self,x,n):
                
        rsi=50*np.ones(x.shape)
        for i in range(n):
           diff=np.diff(x[i-n-1:i])
           pos=np.sum(diff[diff>0])/n # son 14 günde artışların ortlaması
           neg=-np.sum(diff[diff<0])/n # son 14 günde azalışların ortalaması
           rs=pos/neg
           if neg==0:
               neg=0.001
        rsi[i]=100-100/(1+rs)
        rsi[np.isnan(rsi)]=50
        rsi[rsi<1]=1
        rsi[rsi>99]=99
        return rsi
        
    def getReward(self, time,action,bought_price,asset): # zaman indexi verilince o price ve zamanı agent vercek agent aksiyona bakıp ödüle karar  vericek
        
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
                return self.price[time]/bought_price -1 #satarken karlı mı sattı zararlı mı onu kontrol ediyor
            
        return 0
