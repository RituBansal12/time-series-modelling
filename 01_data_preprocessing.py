import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Create visualizations directory if it doesn't exist
os.makedirs('visualizations', exist_ok=True)
os.makedirs('data', exist_ok=True)

def load_data():
    """Load the PJM load data from CSV file."""
    try:
        df = pd.read_csv('data/PJM_Load_hourly.csv')
        # Convert datetime column to datetime object if it exists
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df.set_index('Datetime', inplace=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def plot_time_series(df, title, filename):
    """Plot time series data and save to file."""
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df.iloc[:, 0], linewidth=0.5)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Load (MW)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'visualizations/{filename}.png')
    plt.close()

def preprocess_data(df):
    """Preprocess the data by handling missing values and outliers."""
    # Make a copy of the dataframe
    df = df.sort_values('Datetime')
    df_processed = df.copy()
    
    # 1. Handle missing values using moving average of window 3
    column_name = df_processed.columns[0]  # Get the name of the load column
    df_processed[column_name] = df_processed[column_name].fillna(
        df_processed[column_name].rolling(window=3, min_periods=1).mean()
    )
    
    # If there are still NaN values (can happen at the start), forward fill them
    df_processed[column_name] = df_processed[column_name].fillna(method='ffill')
    
    # 2. Handle outliers
    # Calculate rolling statistics
    rolling_mean = df_processed[column_name].rolling(window=5, center=True).mean()
    rolling_std = df_processed[column_name].rolling(window=5, center=True).std()
    
    # Identify outliers (points that are 3 standard deviations from the rolling mean)
    threshold = 3
    outliers = (df_processed[column_name] < (rolling_mean - threshold * rolling_std)) | \
               (df_processed[column_name] > (rolling_mean + threshold * rolling_std))
    
    # Replace outliers with the average of neighboring points
    for i in np.where(outliers)[0]:
        if i > 0 and i < len(df_processed) - 1:
            df_processed.iloc[i] = (df_processed.iloc[i-1] + df_processed.iloc[i+1]) / 2
    
    return df_processed

def main():
    print("Starting data preprocessing...")
    
    # 1. Load the data
    print("Loading data...")
    df = load_data()
    if df is None:
        print("Failed to load data. Exiting...")
        return
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    
    # 2. Visualize raw data
    print("Visualizing raw data...")
    plot_time_series(df, 'Raw PJM Load Data', 'raw_data')
        
    # 3. Handle outliers, missing values and visualize
    print("Handling outliers and missing values...")
    df_processed = preprocess_data(df)
    plot_time_series(df_processed, 'PJM Load Data (After Preprocessing)', 'preprocessed_data')
    
    # 4. Save preprocessed data
    print("Saving preprocessed data...")
    df_processed.to_csv('data/PJM_Load_hourly_preprocessed.csv')
    
    print("Data preprocessing completed successfully!")
    print(f"Preprocessed data saved to: data/PJM_Load_hourly_preprocessed.csv")

if __name__ == "__main__":
    main()
