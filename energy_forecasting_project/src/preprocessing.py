#These are the imported libraries that I will use for my feature engineering
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def create_features(df):
    #Print initial shape of the dateframe 
    print('Initial:', df.shape)

    #So in the csv that was given to me I was give date and time, of which I translated that to extract the actual hour, so that I can use the hours in the seasonality patterns 
    df['hour'] = df['date'].dt.hour
    print('After hour:', df.shape)

    #I also wanted to capture the days of the wrrk from the dateytime column that was provided to me. 
    df['dayofweek'] = df['date'].dt.dayofweek
    print('After dayofweek:', df.shape)

    #So, in order to have the full picture in our predictability model. We also need to factor in the actual months. That also plays a role in energy generation and how much of energy is produced. 
    df['month'] = df['date'].dt.month
    print('After month:', df.shape)

     #This is how much of energy that was produced an hour or 2 ago. So from my analysis, I understand that it would be extremely beneficial if we were to have those figures so that we can use them when we are predicting how much will have in 6 hours time. 
    df['lag_1'] = df['power_x'].shift(1)
    print('After lag_1:', df.shape)

    df['lag_2'] = df['power_x'].shift(2)
    print('After lag_2:', df.shape)

     #Here I calculatedd the rolling mean, because predicting the generation of energy from just one hour is not accurate. We need the mean to establish the actual/ predicted pattern
    df['rolling_mean_3'] = df['power_x'].rolling(3).mean()
    print('After rolling_mean_3:', df.shape)

     #In this code snippet, I was basically checking for missing values which would be introduced by the lab
    print('NaN count before dropna:\n', df.isna().sum())


    # Only drop NaN in relevant columns
    df.dropna(subset=['power_x', 'lag_1', 'lag_2', 'rolling_mean_3'], inplace=True)
    print('After dropna:', df.shape)
    return df

#In this case, I basically created "6" boxes. These are not actual boxes, but what im trying to explain here is that in the previous code we have an idea as to how much energy was generated in the last 1 or 2 hours. Now, I tell the model what was henerated per hour and I expect the model to generate for the next 6 hours. The XG Boost will learn from the patterns to show me how much more can be generated in the future. 
def create_targets(df, n_steps=6):
    target_cols = [f'target_t+{i}' for i in range(1, n_steps + 1)]
    for i in range(1, n_steps + 1):


#In this final code snippet, I noticed that we willnot have any containers or values of generated energy for the future. I then added this piece of code that tells my model which is still being built to throw away those values. 
        df[f'target_t+{i}'] = df['power_x'].shift(-i)
    df.dropna(subset=target_cols, inplace=True)
    return df

