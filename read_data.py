import pandas as pd
import matplotlib.pyplot as plt
import os
DATA_URL = 'https://raw.githubusercontent.com/plotly/datasets/master/2014_apple_stock.csv'

print("\n") #formatting 

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
    
    plt.plot(time, values)
    plt.title("Apple Stock Prices over Time")
    plt.xlabel("Date")
    plt.ylabel("Stock Value")
    #plt.show()
    plt.savefig('C:/Users/aysha/OneDrive/Desktop/apple_stock.png')
    
    
    return DATA_CSV # or whatever variable you use to store the dataframe

var = read_and_visualize()
print(var)

values = var.iloc[:, 1]

w_size = input("Please enter the size of the rolling average you would like to compute:")
w_size = int(w_size)

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
    read_and_visualize() #this is printing in the correct place 
    plt.plot(values, roll)
    plt.show()
    
    return roll

rollAvg = calculate_moving_average(values, w_size)
print(rollAvg)
    