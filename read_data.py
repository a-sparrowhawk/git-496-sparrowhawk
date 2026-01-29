import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks #reference: https://plotly.com/python/peak-finding/

DATA_URL = 'https://raw.githubusercontent.com/plotly/datasets/master/2014_apple_stock.csv'

print("\n") #formatting 


#plotting the data  
def read_and_visualize():
    # You'll use pandas to read the CSV data from the URL.
    # The read_csv function from pandas can accept a URL directly.
    DATA_CSV = pd.read_csv(DATA_URL)

    # Print the total number of rows and columns in the dataset.
    df = pd.DataFrame(DATA_CSV)
    num_rows = df.shape[0]
    num_cols = df.shape[1]
    print("There are", num_rows, "rows, and", num_cols, "columns. \n")
    
    # Plot the results and save the plot as 'apple_stock.png'.
    # X is the date, and Y is the value of the stock on that date 
    time = df.iloc[:, 0]
    values = df.iloc[:, 1]
    
    plt.plot(time, values, color = "black")
    plt.title("Apple Stock Prices over Time")
    plt.xlabel("Date")
    plt.ylabel("Stock Value")
    plt.savefig('C:/Users/aysha/OneDrive/Desktop/apple_stock.png')
    
    return DATA_CSV 

var = read_and_visualize()
print(var)


#moving/rolling average 
values = var.iloc[:, 1]
time = var.iloc[:, 0]

w_size = input("\nPlease enter the size of the rolling average you would like to compute: ")
w_size = int(w_size)
print('\n')

def calculate_moving_average(values, w_size):
    if(w_size > len(values)):
        print("You entered a length that is bigger than the number of rows in the dataframe. I am going to assume you meant to enter the length of the dataframe. \n")
        w_size = len(values)
        roll = values.rolling(window = w_size).mean()
    else:    
        roll = values.rolling(window = w_size).mean() #referenced: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
        
    #call the function and then show the plt.show() to have it display both of the plot components
    plt.plot(time, roll, color = "red") #changed from values to time
    
    return roll

rollAvg = calculate_moving_average(values, w_size)
rollAvg
    

#bollinger bands 
def calculate_bollinger_bands(rollAvg, window_size = 30):
    upper = rollAvg + 2*np.std(rollAvg)
    lower = rollAvg - 2*np.std(rollAvg)
    plt.plot(time, upper, color = "blueviolet")
    plt.plot(time, lower, color = "fuchsia")
    
bbands = calculate_bollinger_bands(rollAvg, window_size = 30)
bbands
    

#maximum drawdown and daily drawdown 
def max_and_daily_drawdown(values):
    #print the maximum drawdown over the entire series 
    P = find_peaks(values)[0] #referenced stack overflow
    L = find_peaks(-values)[0] #referenced stack overflow

    MDD = max((P-L)/P)
    print("The maximum drawdown over the entire series is:", MDD)
    
    #plot the daily drawdown over the entire series 
    
    #referenced: https://stackoverflow.com/questions/22607324/start-end-and-duration-of-maximum-drawdown-in-python
    max_daily = [values[0]]
    for i in range(1, len(values)):
        if values[i-1] > values[i]:
            max_daily.append(values[i-1])
        else:
            max_daily.append(values[i]) #changing from 0 to values[i]    
    #end of reference 
    
    '''
    #values = values.pct_change()
    print(values)
    #referenced: https://www.youtube.com/watch?v=TI9f9jIO41I
    cumulative = (values + 1).cumsum()
    running_max = np.maximum.accumulate(cumulative)
    max_daily = (cumulative - running_max)/running_max
    '''
    
    plt.plot(time, max_daily, color = "green")
    plt.show() 
    #displaying the plot now because the RSI in the next step is wrong and you cannot see these lines otherwise.
    
    return max_daily  

madd = max_and_daily_drawdown(values)
madd
print("\n")

#working up until here

#relative strength index 
def relative_strength_index(values):
    #calculate the day to day price differences and store in variable called diff
    values = np.array(values)
    
    diff = np.zeros(len(values))
    for i in range(1, len(values)):
        diff[i] = values[i] - values[i-1]
            
    #separate the two series
    #   one for positive differences (Gains)         
    #   one for negative differences (Losses)
    
    Gains = []
    Losses = []
    for i in range(0, len(diff)):
        if diff[i] < 0:
            Losses.append(diff[i])
            Gains.append(0)
        elif diff[i] >= 0:
            Gains.append(diff[i])
            Losses.append(0)
            
    Gains = pd.DataFrame(Gains)
    Losses = pd.DataFrame(Losses)
    
    Gains = Gains.loc[:, 0]
    Losses = Losses.loc[:, 0]
        
    #take the rolling average of gains vs losses and derive the Relative strength.
    #   take a three day rolling avg for this
    rollAvg_gains = Gains.rolling(3).mean()
    rollAvg_losses = Losses.rolling(3).mean()
        
    #compute RS doing rolling avg gains / rolling avg losses
    RS = rollAvg_gains/rollAvg_losses
    RS[0] = 0
    RS[1] = 0

    #compute RSI as a time series using formula
    RSI = 100 - (100 / (1 + RS))
    
    plt.plot(time, RSI, color = "gray")
    plt.title("RSI Over Time") #this is very incorrect.
    plt.ylabel("RSI")
    plt.xlabel("Time")
    plt.show() 
    #I am displaying RSI separately because there is something wrong with the calculation and the values are too big. 
    
    return RSI

result = relative_strength_index(values)
result

