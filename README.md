# Predicting Stock Price in Pyhton Project

I would like to study stock price estimation on the financial market. My project consists of 3 parts that are Dataset, Linear Regression and Reinforcement Learning (RL) Environment.
## Data Set
Firstly, I prepare a timeseries data with respect to looking at the historical data, which enables me to query all dataset in a same way. Data set information is stated "Data Set Description". 

## Linear Regression 
Secondly, Linear Regression focuses on two variables that are dependent and independent. Dependent variable implies the target and the independent variable implies feature. I would like to estimate the price of stock in financial market with the daily data and then make a restriction such as 30 days, 60 days and etc., which enables to forecast future price movement up or down. It can be seen as a signal. 

Methods that are used to predict stock price listed below:

 Ridge Regression : it works with penalty terms. The aim is that penalty coefficient is added to loss function so as to minimize variance.
  
 Ordinary Least Square: it estimates the relationship between independent variables (features) and a dependent variable (target).

 ## Reinforcement Learning
 
As for RL, I researched the method that is used to estimate stock price. I found  the RL  method being an area of Machine Learning. This method is mainly talk about the interaction between environment and the result. The idea behind the RL interacting with the environment many times and getting the reward or penalty with respect to action result, then it will find the optimal way and result by gaining experience. In this method has so many submethod but i want to work Quality-table (Q-table).

--- Elements of RL beyond agent and environment:
  
   -> policy
   
   -> reward signal
   
   -> value function
   
   -> model
  
  
  policy: it stands for the learning agent's way of behaving.
  
  reward signal: it stands for the RL. Each step of the environment sends a single number that can be accepted as a reward.
  
  value function: it stands for the what is good in the long process and an action would be accumulate over the next steps.
  
  model: it stands for mimics the behaviour of the environment which allows interfaces to be made a decision about how it can behave.
  
  *** the learner or decision maker is called agent. 
  ### Q-Table  
  Q-Learning seeks to find best action to take given current state. Q stands for the Quality.
  In this project, values [State, Action] initialized to zero and then update and store q-values after an episode. 
  Q-table can be seen as a reference point for our agent to select the best way. 
  

 
 
 

# Data Set Description:

I used different assets from the stock market and also included silver and gold prices. I prepare dataset for querying all data in a same way.

1- Stock Market Data Set 
I have downloaded historical data (daily based 5Y) set from the yahoo finance. First stock is ASELSAN from the Borsa Istanbul and the other is NetFlix from the NASDAQ.

2- Gold and Silver Data Set
I have dowloaded daily basis historical data(daily based 5Y) from investing.com website. 
-------------------

 # Data Set Stock Result Sample- Aselsan: 
 
        Date    Open    High     Low   Close  Adj Close    Volume
        
   0     2016-06-14   4.705   4.710   4.550   4.590   4.453227   2862848
 
   1     2016-06-15   4.600   4.655   4.585   4.610   4.472632   2108200
 
   2     2016-06-16   4.605   4.610   4.510   4.530   4.395016   1331068
 
   3     2016-06-17   4.535   4.590   4.535   4.575   4.438675   1211294
 
   4     2016-06-20   4.610   4.775   4.610   4.760   4.618162   3244210
         ...     ...     ...     ...     ...        ...       ...
         
 1283  2021-06-07  15.150  15.580  15.140  15.460  15.460000  34057364

 1284  2021-06-08  15.470  15.520  15.320  15.430  15.430000  20037369
 
 1285  2021-06-09  15.430  15.470  15.280  15.300  15.300000  14455072
 
 1286  2021-06-10  15.360  15.750  15.320  15.700  15.700000  30898798
 
1 287  2021-06-11  15.760  15.950  15.610  15.610  15.610000  38839794

 Open Prices: 

[15.76  15.36  15.43  ...  4.605  4.6    4.705]

 Classification Time Data Set: 

[[14.74  15.35  14.96  ... 15.47  15.43  15.36 ]
 [14.5   14.74  15.35  ... 15.15  15.47  15.43 ]
 [14.23  14.5   14.74  ... 15.08  15.15  15.47 ]
 ...
 [ 4.605  4.535  4.61  ...  4.5    4.58   4.54 ]
 [ 4.6    4.605  4.535 ...  4.55   4.5    4.58 ]
 [ 4.705  4.6    4.605 ...  4.64   4.55   4.5  ]] 
 Predicted Prices: 
 [[1.03841146]
 [1.02073882]
 [1.        ]
 ...
 [1.05286344]
 [1.01965066]
 [1.02444444]]
 
 Regression Time Data set: 
 [[14.74  15.35  14.96  ... 15.47  15.43  15.36 ]
 [14.5   14.74  15.35  ... 15.15  15.47  15.43 ]
 [14.23  14.5   14.74  ... 15.08  15.15  15.47 ]
 ...
 [ 4.605  4.535  4.61  ...  4.5    4.58   4.54 ]
 [ 4.6    4.605  4.535 ...  4.55   4.5    4.58 ]
 [ 4.705  4.6    4.605 ...  4.64   4.55   4.5  ]] 
 Open Price Data set: 
 [[15.76 ]
 [15.36 ]
 [15.43 ]
 ...
 [ 4.695]
 [ 4.54 ]
 [ 4.58 ]]

 The average of the predicted price data set:  0.4968203497615262
 
 # Data Set Sample Result -Gold:
 
  Date     Price      Open      High       Low      Vol Change %
0     Jun 14, 2021  1,868.60  1,875.60  1,875.85  1,846.20        -   -0.37%
1     Jun 13, 2021  1,875.55  1,879.35  1,879.75  1,875.25        -   -0.22%
2     Jun 11, 2021  1,879.60  1,901.90  1,906.20  1,876.10  220.46K   -0.89%
3     Jun 10, 2021  1,896.40  1,891.40  1,903.00  1,871.80  250.55K    0.05%
4     Jun 09, 2021  1,895.50  1,894.40  1,901.70  1,889.30  147.33K    0.06%
           ...       ...       ...       ...       ...      ...      ...
1341  May 20, 2016  1,305.90  1,305.90  1,305.90  1,305.90    0.04K    0.00%
1342  May 19, 2016  1,305.90  1,305.90  1,305.90  1,305.90    0.07K   -1.39%
1343  May 18, 2016  1,324.30  1,324.30  1,324.30  1,324.30    0.02K   -0.05%
1344  May 17, 2016  1,324.90  1,324.90  1,324.90  1,324.90    0.02K    0.23%
1345  May 16, 2016  1,321.80  1,321.80  1,321.80  1,321.80    0.04K    0.17%

 Open Prices: 
 ['1,321.80' '1,324.90' '1,324.30' ... '1,879.60' '1,875.55' '1,868.60']
RSI : [50.         50.         50.         ... 54.23211169 41.11394558   38.06506126]

![image](https://user-images.githubusercontent.com/78654515/124862601-5e12c200-dfbe-11eb-8e9c-27b1009c44d1.png)

First part of the image open prices are taken into consideration and the second part RSI index significant levels are taken into consideration.
