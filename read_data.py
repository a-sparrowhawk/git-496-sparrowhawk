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

w_size = input("\nPlease enter the size of the rolling average you would like to compute: ")
w_size = int(w_size)
print('\n')

def calculate_moving_average(values, w_size):
    if(w_size > len(values)):
        print("You entered a length that is bigger than the number of rows in the dataframe. I am going to assume you meant to enter the length of the dataframe. \n")
        w_size = len(values)
        roll = values.rolling(w_size).sum()
        roll = roll / w_size
    else:    
        roll = values.rolling(w_size).sum() #referenced: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
        roll = roll / w_size
        
    #call the function and then show the plt.show() to have it display both of the plot components
    #read_and_visualize() #this is printing in the correct place 
    plt.plot(values, roll, color = "red")
    
    return roll

rollAvg = calculate_moving_average(values, w_size)
rollAvg
    
    
#bollinger bands 
def calculate_bollinger_bands(rollAvg, window_size = 30):
    upper = rollAvg + np.std(rollAvg)
    lower = rollAvg - np.std(rollAvg)
    plt.plot(values, upper, color = "blueviolet")
    plt.plot(values, lower, color = "fuchsia")
    #plt.show()
    
bbands = calculate_bollinger_bands(rollAvg, window_size = 30)
bbands
    
    
#maximum drawdown and daily drawdown 
def max_and_daily_drawdown(values, window):
    #print the maximum drawdown over the entire series 
    P = find_peaks(values)[0] #referenced stack overflow
    L = find_peaks(-values)[0] #referenced stack overflow

    MDD = max((P-L)/P)
    print("The maximum drawdown over the entire series is:", MDD)
    
    #plot the daily drawdown over the entire series 
    #reference: https://medium.com/cloudcraftz/measuring-maximum-drawdown-and-its-python-implementation-99a3963e158f

    roll_max = values.rolling(window, min_periods =1).max()
    daily_drawdown = values/roll_max - 1
    max_daily = daily_drawdown.rolling(window, min_periods = 1).min()
    #end of reference 
    
    plt.plot(values, max_daily, color = "green")
    plt.show()
    
    return max_daily  

madd = max_and_daily_drawdown(values, window = len(values))
madd
print("\n")
