# Predicting Stock Price in Pyhton Project

I would like to study stock price estimation on the financial market. My project consists of 3 parts that are Dataset, Linear Regression and Reinforcement Learning (RL) Environment.

Firstly, I prepare a timeseries data with respect to looking at the historical data, which enables me to query all dataset in a same way. Data set information is stated "Data Set Description". 

Secondly, Linear Regression focuses on two variables that are dependent and independent. Dependent variable implies the target and the independent variable implies feature. I would like to estimate the price of stock in financial market with the daily data and then make a restriction such as 30 days, 60 days and etc., which enables to forecast future price movement up or down. It can be seen as a signal. 

As for RL, I researched the method that is used to estimate stock price. I found  the RL  method being an area of Machine Learning. This method is mainly talk about the interaction between environment and the result. The idea behind the RL interacting with the environment many times and getting the reward or penalty with respect to action result, then it will find the optimal way and result by gaining experience. In this method has so many submethod but i want to work Quality-table (Q-table).
 --- Elements of RL beyond agent and environment:
  -> policy
  -> reward signal
  -> value function
  -> model
  
  policy: it stands for the learning agent's way of behaving 
  reward signal: it stands for the RL. Each step of the environment sends a single number that can be accepted as a reward.
  value function: it stands for the what is good in the long process and an action would be accumulate over the next steps.
  model: it stands for mimics the behaviour of the environment which allows interfaces to be made a decision about how it can behave.
  
  *** the learner or decision maker is called agent. 
  
  
 
 

# Data Set Description:

I used different assets from the stock market and also included silver and gold prices. I prepare dataset for querying all data in a same way.

1- Stock Market Data Set 
I have downloaded historical data (daily based 5Y) set from the yahoo finance. First stock is ASELSAN from the Borsa Istanbul and the other is NetFlix from the NASDAQ.

2- Gold and Silver Data Set
I have dowloaded daily basis historical data(daily based 5Y) from investing.com website. 
