# Import libraries for data manipulation and splitting
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def create_features(df):
    # Print the initial shape of the DataFrame to confirm data load
    print('Initial:', df.shape)

    # Extract the hour from the datetime column for capturing daily patterns
    df['hour'] = df['date'].dt.hour
    print('After hour:', df.shape)

    # Extract day of the week to capture weekly patterns (e.g., weekdays vs weekends)
    df['dayofweek'] = df['date'].dt.dayofweek
    print('After dayofweek:', df.shape)

    # Extract month to capture seasonal or monthly variations in energy production
    df['month'] = df['date'].dt.month
    print('After month:', df.shape)

    # Create lag features representing energy production 1 and 2 hours before
    # These provide recent context for the model to learn from short-term trends
    df['lag_1'] = df['power_x'].shift(1)
    print('After lag_1:', df.shape)

    df['lag_2'] = df['power_x'].shift(2)
    print('After lag_2:', df.shape)

    # Calculate rolling mean of the past 3 hours to smooth short-term fluctuations
    df['rolling_mean_3'] = df['power_x'].rolling(3).mean()
    print('After rolling_mean_3:', df.shape)

    # Display count of missing values introduced by lag and rolling calculations
    print('NaN count before dropna:\n', df.isna().sum())

    # Drop rows with NaN values in relevant columns to ensure clean training data
    df.dropna(subset=['power_x', 'lag_1', 'lag_2', 'rolling_mean_3'], inplace=True)
    print('After dropna:', df.shape)

    return df

def create_targets(df, n_steps=6):
    """
    Create future target columns for multi-step forecasting.
    Each target_t+i column represents energy production i hours ahead.
    """
    target_cols = [f'target_t+{i}' for i in range(1, n_steps + 1)]

    # For each step ahead, create a shifted column as the target
    for i in range(1, n_steps + 1):
        df[f'target_t+{i}'] = df['power_x'].shift(-i)

    # Drop rows where future targets are not available (occur at the end of dataset)
    df.dropna(subset=target_cols, inplace=True)

    return df
