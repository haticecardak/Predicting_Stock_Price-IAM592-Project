import numpy as np

class Agent():
  def __init__(self,gamma=0.90,eps=0.1):
       self.gamma=gamma
        self.eps=eps
        self.actions = ['buy', 'hold', 'sell']
        self.asset='cash'
        
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
