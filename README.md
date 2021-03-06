# Predicting Stock Price in Pyhton Project

   I would like to study stock price estimation on the financial market. My project consists of 3 parts that are Dataset, Linear Regression and Reinforcement Learning (RL) Environment.
   
## Data Set
      Firstly, I prepare a timeseries data with respect to looking at the historical data, which enables me to query all dataset in a same way. Data set information and sample results are stated below "Data Set Description". 
      My data set is composed of 4 financial market instruments that are ASELSAN, NETFLİX, Gold and Silver.
      I really wonder about the Gold and Silver price movement also adding two company stocks in order to analyze their predicting stock prices. 
      In my data set file,  I prepared timeseries with using regression and classification method. 

      In the regression task, feature matrix  is formed by using first 30 day close price of the stock and this feature try yo estimate the 31st day Close Price that is called target. 
      For instance, from 1st day to 30rd day Close Prices are called feature matrix, and 31st day Close Price is called target.
      In the classification task,  feature matrix  is formed by using first 30 day close price of the stock that is the same as regression feature matrix, but target plays a deterministic role. 
      Target matrix is considering raito of 31 st day high price  and  30 rd day close price. 

## Linear Regression 
      
      Secondly, Linear Regression focuses on two variables that are dependent and independent. Dependent variable implies the target and the independent variable implies feature. I would like to estimate the price of stock in financial market with the daily data and then make a restriction such as 30 days, 60 days and etc., which enables to forecast future price movement up or down. It can be seen as a signal. 

      Methods that are used to predict stock price listed below:

      Ridge Regression : it works with penalty terms. The aim is that penalty coefficient is added to loss function so as to minimize variance.
  
      Ordinary Least Square: it estimates the relationship between independent variables (features) and a dependent variable (target).

 ## Reinforcement Learning
 
      As for RL, I researched the method that is used to estimate stock price. I found  the RL  method being an area of Machine Learning. 
      This method is mainly talk about the interaction between environment and the result. 
      The idea behind the RL interacting with the environment many times and getting the reward or penalty with respect to action result then it will find the optimal way and result by gaining experience.
      In this method has so many submethod but i want to work Quality-table (Q-table).

      --- Elements of RL beyond agent and environment:
          -> policy
          -> reward signal
          -> value function
          -> model 
    The learner or decision maker is called agent.     
   
      policy: it stands for the learning agent's way of behaving at a given time.
      
      reward signal: it stands for the RL. Each step of the environment sends a single number that can be accepted as a reward. 
    This signal defines what are the good and bad events for the agent.
    
       value function: it stands for the what is good in the long process and an action would be accumulate over the next steps. 
    The value of the state expect to accumulate over the future starting from the state.
      
      model: it stands for mimics the behaviour of the environment which allows interfaces to be made a decision about how it can behave. 
    The behaviour of the environment that allows inferences to be made about how the environment will behave. 
    
    
    The thing it interacts with comprising everything outside the agent is called environment in this project stock price results.
    These interact continuously, the agent select the actions and environment responds the action result and present the new case for the agent.
    Environment gives the reward of the action that is a special numerical value multipling with gamma and agent tries to maximize this value over time. 
    
    In this project our actions buy, sell and hold.
    When agent will select the actions that depend on the information such as dataset result and the next step is following by a reward can be given by the environment. 
    In the state of the datasets, Relative Strength Index (RSI) indicator being used for technical analysis in the financial market is also used to make decision about the action. 
    RSI takes into consideration 14 day price movement. It calculates the 14 day average of upward price and average of the downward price and then dividing the upward result and downward result.  
    After the calculation it makes sense about the taking actions. RSI value range is [0,100]. 
    According to RSI, if the value is less than 30 the action would be buy and if the value is higher than 70 the action would be sell. 
    In this case this rule is not used entirely because agent will give a decision based on reward signal.
    
    
  ### Q-Table  
  
    
     Q-Learning seeks to find best action to take given current state. Q stands for the Quality and also it seeks to learn a policy maximizing the reward.
     At begining, Q-table or it can be called as matrix is initiliazed to zero.
     When the state is done, then the Q-table would be updated. 
     The updates occur after each step or action and ends when state is done. The agent will not learn much after a single step. 
     After updating, Q-table would be getting reference point for the agent.
     In this project, values [State, Action] initialized to zero and then update and store q-values after an state that is set 100 action. 
     
     The sample run is stated below belongs to GoldDataSet and it shows us selling case profit with respect to state period.
     After the period result, Q-table is updated. 
     Q-table can be seen as a reference point for our agent to select the best way. 
  
![image](https://user-images.githubusercontent.com/78654515/125017686-5154a380-e07c-11eb-94ed-ba3577ed931a.png)
    ![image](https://user-images.githubusercontent.com/78654515/125018257-6b42b600-e07d-11eb-9445-e52c6b6210a3.png)
 ![image](https://user-images.githubusercontent.com/78654515/125018282-7695e180-e07d-11eb-9804-52d4ebfd3614.png)
      ![image](https://user-images.githubusercontent.com/78654515/125018304-801f4980-e07d-11eb-80ea-0f8f4fa6bc2d.png)

     
         Gold State 800 Completed
         Sell Reward : -0.002
         Sell Reward : -0.000
         Sell Reward : 0.003
         Sell Reward : 0.002
         Sell Reward : -0.002
         Sell Reward : -0.003
         Sell Reward : 0.000
         Sell Reward : 0.010
         Sell Reward : -0.005
         Sell Reward : -0.005
         Sell Reward : -0.009
         Sell Reward : 0.368
         Sell Reward : -0.019
         Sell Reward : -0.013
         Sell Reward : 0.005
         Sell Reward : -0.006
         Sell Reward : -0.005
         Sell Reward : 0.005
         Sell Reward : 0.002
         Sell Reward : 0.023
         Sell Reward : -0.010
         Sell Reward : 0.005
         Sell Reward : 0.009
         Sell Reward : 0.007
         Sell Reward : -0.018
         Sell Reward : -0.023
         Sell Reward : 0.011
         Sell Reward : 0.002
         Sell Reward : -0.000
         Sell Reward : -0.015
         Sell Reward : -0.025
         Sell Reward : -0.004
         Sell Reward : 0.003
         Sell Reward : -0.061
         Sell Reward : -0.002
         Sell Reward : 0.007
         Sell Reward : 0.001
         Sell Reward : -0.003
         Sell Reward : -0.001
         Sell Reward : -0.001
         Sell Reward : -0.014
         Sell Reward : 0.027
         Sell Reward : 0.002
         Sell Reward : -0.011
         Sell Reward : -0.005
         Sell Reward : -0.002
         Sell Reward : 0.004
         Sell Reward : 0.001
         Sell Reward : 0.007
         Sell Reward : -0.047
         Sell Reward : 0.028
         Sell Reward : -0.010
         Sell Reward : 0.033
         Sell Reward : -0.000
         Sell Reward : 0.021
         Sell Reward : 0.003
         Sell Reward : 0.002
         Sell Reward : 0.001
         Sell Reward : 0.000
         Sell Reward : -0.007
         Sell Reward : 0.002
         Sell Reward : -0.007
         Sell Reward : 0.003
         Sell Reward : 0.015
         Sell Reward : 0.007
         Sell Reward : 0.062
         Sell Reward : -0.008
         Sell Reward : 0.004
         Sell Reward : -0.007
         Sell Reward : 0.001
         Sell Reward : -0.009
         Sell Reward : -0.006
         Sell Reward : -0.001
         Sell Reward : -0.008
         Sell Reward : 0.004
         Sell Reward : 0.002
         Sell Reward : 0.017
         Sell Reward : 0.001
         Sell Reward : -0.004
         Sell Reward : 0.003
         Sell Reward : -0.057
         Sell Reward : -0.014
         Sell Reward : 0.351
         Sell Reward : 0.002
         Sell Reward : 0.006
         Sell Reward : -0.001
         Sell Reward : 0.004
         Sell Reward : 0.009
         Sell Reward : -0.004
         Sell Reward : -0.005
         Sell Reward : -0.006
         Sell Reward : 0.003
         Sell Reward : 0.001
         Sell Reward : 0.007
         Sell Reward : -0.001
         Sell Reward : 0.002
         Sell Reward : -0.013
         Sell Reward : 0.000
         Sell Reward : -0.017
         Sell Reward : 0.006
         Sell Reward : 0.009
         Sell Reward : 0.027
         Sell Reward : -0.050
         Sell Reward : -0.008
         Sell Reward : -0.003
         Sell Reward : 0.001
         Sell Reward : -0.003
         Sell Reward : -0.005
         Sell Reward : 0.004
         Sell Reward : 0.002
         Sell Reward : 0.023
         Sell Reward : -0.000
         Sell Reward : 0.008
         Sell Reward : 0.008
         Sell Reward : -0.000
         Sell Reward : -0.002
         Sell Reward : -0.015
         Sell Reward : 0.016
         Sell Reward : 0.007
         Sell Reward : -0.003
         Sell Reward : 0.007
         Sell Reward : -0.003
         Sell Reward : 0.001
         Sell Reward : -0.003
         Sell Reward : -0.019
         Sell Reward : 0.004
         Sell Reward : 0.002
         Gold State 900 Completed
         Sell Reward : 0.005
         Sell Reward : -0.017
         Sell Reward : -0.002
         Sell Reward : 0.034
         Sell Reward : 0.005
         Sell Reward : 0.016
         Sell Reward : -0.005
         Sell Reward : -0.048
         Sell Reward : 0.003
         Sell Reward : -0.010
         Sell Reward : 0.218
         Sell Reward : 0.034
         Sell Reward : 0.011
         Sell Reward : -0.009
         Sell Reward : 0.003
         Sell Reward : -0.004
         Sell Reward : -0.002
         Sell Reward : -0.029
         Sell Reward : 0.025
         Sell Reward : 0.005
         Sell Reward : 0.003
         Sell Reward : 0.001
         Sell Reward : -0.001
         Sell Reward : -0.016
         Sell Reward : 0.005
         Sell Reward : 0.003
         Sell Reward : 0.001
         Sell Reward : 0.004
         Sell Reward : 0.001
         Sell Reward : 0.005
         Sell Reward : 0.017
         Sell Reward : -0.016
         Sell Reward : -0.001
         Sell Reward : 0.015
         Sell Reward : -0.002
         Sell Reward : 0.001
         Sell Reward : -0.009
         Sell Reward : -0.031
         Sell Reward : 0.005
         Sell Reward : 0.011
         Sell Reward : 0.000
         Sell Reward : -0.008
         Sell Reward : -0.007
         Sell Reward : 0.006
         Sell Reward : -0.000
         Sell Reward : -0.000
         Sell Reward : 0.001
         Sell Reward : -0.001
         Sell Reward : -0.007
         Sell Reward : -0.001
         Sell Reward : -0.003
         Sell Reward : -0.002
         Sell Reward : 0.001
         Sell Reward : -0.000
         Sell Reward : 0.004
         Sell Reward : -0.001
         Sell Reward : 0.008
         Sell Reward : -0.008
         Sell Reward : -0.007
         Sell Reward : 0.002
         Sell Reward : -0.002
         Sell Reward : -0.022
         Sell Reward : 0.381
         Sell Reward : 0.004
         Sell Reward : 0.002
         Sell Reward : 0.003
         Sell Reward : 0.010
         Sell Reward : -0.002
         Sell Reward : 0.000
         Sell Reward : -0.002
         Sell Reward : 0.002
         Sell Reward : -0.014
         Sell Reward : -0.001
         Sell Reward : -0.004
         Sell Reward : -0.005
         Sell Reward : 0.047
         Sell Reward : -0.002
         Sell Reward : 0.004
         Sell Reward : 0.012
         Sell Reward : 0.002
         Sell Reward : -0.002
         Sell Reward : 0.008
         Sell Reward : 0.001
         Sell Reward : -0.002
         Sell Reward : 0.045
         Sell Reward : -0.005
         Sell Reward : -0.005
         Sell Reward : -0.035
         Sell Reward : 0.046
         Sell Reward : -0.005
         Sell Reward : -0.000
         Sell Reward : 0.006
         Sell Reward : -0.003
         Sell Reward : 0.005
         Sell Reward : 0.009
         Sell Reward : -0.003
         Sell Reward : 0.007
         Sell Reward : 0.012
         Sell Reward : 0.009
         Sell Reward : 0.003
         Sell Reward : -0.002
         Sell Reward : 0.004
         Sell Reward : -0.005
         Sell Reward : 0.006
         Sell Reward : 0.006
         Sell Reward : -0.002
         Sell Reward : -0.012
         Sell Reward : -0.001
         Sell Reward : -0.007
         Sell Reward : -0.001
         Sell Reward : 0.006
         Sell Reward : -0.003
         Sell Reward : 0.003
         Sell Reward : 0.051
         Sell Reward : 0.018
         Sell Reward : -0.007
         Sell Reward : -0.003
         Sell Reward : 0.005
         Sell Reward : -0.006
         Sell Reward : 0.041
         Sell Reward : 0.014
         Sell Reward : -0.014
         Sell Reward : -0.019
         Sell Reward : -0.013
         Sell Reward : 0.005
         Sell Reward : -0.006
         Sell Reward : -0.005
         Sell Reward : 0.000
         Sell Reward : 0.000
         Sell Reward : 0.002
         Sell Reward : 0.010
         Sell Reward : 0.005
         Sell Reward : -0.001
         Sell Reward : -0.003
         Sell Reward : -0.010
         Sell Reward : 0.002
         Sell Reward : -0.006
         Sell Reward : 0.003
         Sell Reward : -0.004
         Sell Reward : 0.011
         Sell Reward : 0.007
         Sell Reward : -0.003
         Sell Reward : -0.056
         Sell Reward : 0.047
         Sell Reward : -0.010
         Sell Reward : -0.005
         Gold State 1000 Completed
         Gold Q Table
         [[ 0.          0.          0.        ]
          [ 0.          0.          0.        ]
          [ 0.          0.          0.        ]
          [ 0.          0.          0.        ]
          [-0.01623284  0.          0.0062734 ]
          [ 0.          0.          0.        ]
          [ 0.          0.          0.        ]
          [ 0.          0.          0.        ]
          [ 0.          0.          0.        ]
          [ 0.          0.          0.        ]]




# Data Set Description:

      I used different assets from the stock market and also included silver and gold prices. I prepare dataset for querying all data in a same way.

      1- Stock Market Data Set 
      I have downloaded historical data (daily based 5Y) set from the yahoo finance. First stock is ASELSAN from the Borsa Istanbul and the other is NetFlix from the NASDAQ.

      2- Gold and Silver Data Set
      I have dowloaded daily basis historical data(daily based 5Y) from investing.com website. 
      -------------------
# Number of Data set samples: 
      Key : Aselsan , Samples : 1288
      Key : NetFlix , Samples : 1258
      Key : Silver , Samples : 1583
      Key : Gold , Samples : 1346

 # Data Set Stock Result Sample- Aselsan: 
 
           Date        Open    High     Low   Close  Adj Close    Volume
        
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

     Open Prices:   [15.76  15.36  15.43  ...  4.605  4.6    4.705]

     Classification Time Data Set:   [[14.74  15.35  14.96  ... 15.47  15.43  15.36 ]
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
 
               Date     Price      Open      High       Low        Vol    Change %
    
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

    Open Prices: ['1,321.80' '1,324.90' '1,324.30' ... '1,879.60' '1,875.55' '1,868.60']
   
    RSI : [50.         50.         50.         ... 54.23211169 41.11394558   38.06506126]

  ![image](https://user-images.githubusercontent.com/78654515/124975503-c2716800-e036-11eb-9e35-38514d5cbc5e.png)

First part of the image open prices are taken into consideration and the second part RSI index significant levels are taken into consideration.
