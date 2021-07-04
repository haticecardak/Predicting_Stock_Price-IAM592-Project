# Predicting Stock Price in Pyhton Project

 I would like to study stock price estimation on the financial market. Firstly, I prepare a timeseries data with respect to looking at the historical data by using Linear Regression method. Linear Regression focuses on two variables that are dependent and independent. Dependent variable implies the target and the independent variable implies feature. I would like to estimate the price of stock in financial market with the daily data and then make a restriction such as 30 days, 60 days and etc., which enables to forecast future price movement up or down. It can be seen as a signal. 

 Moreover, I researched the method that is used to estimate stock price. I found  the Reinforcement Learning (RL) method being an area of Machine Learning. This method is mainly talk about the interaction between environment and the result. The idea behind the RL interacting with the environment many times and getting the reward or penalty with respect to action result, then it will find the optimal way and result by gaining experience. In this method has so many submethod but i want to work Quality-table (Q-table).
 

# Data Set Description:

I used different assets from the stock market and also included silver and gold prices. I prepare dataset for querying all data in a same way.

1- Stock Market Data Set 
I have downloaded historical data (daily based 5Y) set from the yahoo finance. First stock is ASELSAN from the Borsa Istanbul and the other is NetFlix from the NASDAQ.

2- Gold and Silver Data Set
I have dowloaded daily basis historical data(daily based 5Y) from investing.com website. 


# ASELSAN Dataset Result
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
1287  2021-06-11  15.760  15.950  15.610  15.610  15.610000  38839794

[1288 rows x 7 columns]

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
