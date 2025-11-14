import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import os
import sys
import datetime
from datetime import datetime as dt

# Set up logging
def setup_logging():
    os.makedirs('results', exist_ok=True)
    log_file = f'results/time_series_prep_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'a')
            
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
            
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(log_file)
    return log_file

def create_exogenous_features(df):
    """Create exogenous features from datetime index."""
    # Ensure index is datetime
    df.index = pd.to_datetime(df.index)
    
    # Time of day features (already one-hot encoded)
    df['hour'] = df.index.hour
    time_bins = [-1, 6, 12, 18, 23]
    time_labels = ['night', 'morning', 'afternoon', 'evening']
    df['time_of_day'] = pd.cut(df['hour'], bins=time_bins, labels=time_labels)
    time_dummies = pd.get_dummies(df['time_of_day'], prefix='time')
    
    # Day of week features
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Month and season
    seasons = {1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring', 
               6: 'summer', 7: 'summer', 8: 'summer', 9: 'fall', 10: 'fall', 
               11: 'fall', 12: 'winter'}
    df['season'] = df.index.month.map(seasons)
    season_dummies = pd.get_dummies(df['season'], prefix='season')
    
    # Combine all features and ensure they're all numeric
    exog = pd.concat([
        time_dummies,
        season_dummies,
        df[['is_weekend']]
    ], axis=1)
    
    # Convert all columns to float to ensure numeric type
    exog = exog.astype(float)
    
    return exog

def load_preprocessed_data():
    """Load the preprocessed data and create exogenous features."""
    try:
        # Load the main data
        df = pd.read_csv('data/PJM_Load_hourly_preprocessed.csv', index_col=0, parse_dates=True)
        
        # Create and save exogenous features
        exog = create_exogenous_features(df.copy())
        exog.to_csv('data/exogenous_features.csv')
        print("Saved exogenous features to data/exogenous_features.csv")
        
        return df
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        return None

def test_stationarity(timeseries, window=24):
    """Test stationarity using Augmented Dickey-Fuller test."""
    print('\nResults of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    
    # Print detailed test results
    print("\n" + "="*50)
    print("Stationarity Test Results")
    print("="*50)
    print(dfoutput)
    print("\nInterpretation:")
    print(f"- p-value = {dftest[1]:.6f}")
    if dftest[1] <= 0.05:
        print("- The time series is likely stationary (p-value <= 0.05).")
    else:
        print("- The time series is likely non-stationary (p-value > 0.05).")
    print("="*50 + "\n")
    
    return dftest[1]  # Return p-value

def make_stationary(series):
    """Apply transformations to make the series stationary."""
    # First difference
    diff = series.diff().dropna()
    p_value = test_stationarity(diff)
    
    if p_value > 0.05:
        # If still not stationary, apply seasonal difference (24 hours)
        print("First difference not sufficient, applying seasonal difference...")
        diff = diff.diff(24).dropna()
        p_value = test_stationarity(diff)
        
        if p_value > 0.05:
            # If still not stationary, apply log transformation
            print("Seasonal difference not sufficient, applying log transformation...")
            log_series = np.log1p(series)
            diff = log_series.diff().dropna()
            p_value = test_stationarity(diff)
            
            if p_value > 0.05:
                print("Warning: Series may not be perfectly stationary after transformations.")
    
    return diff

def plot_decomposition(series, period=24*7):
    """Plot time series decomposition."""
    print(f"\nPerforming time series decomposition with period={period}...")
    try:
        decomposition = seasonal_decompose(series, period=period)
        
        plt.figure(figsize=(15, 12))
        
        plt.subplot(411)
        plt.plot(series, label='Original')
        plt.legend(loc='best')
        plt.title('Original Time Series')
        
        plt.subplot(412)
        plt.plot(decomposition.trend, label='Trend')
        plt.legend(loc='best')
        plt.title('Trend')
        
        plt.subplot(413)
        plt.plot(decomposition.seasonal, label='Seasonality')
        plt.legend(loc='best')
        plt.title('Seasonality')
        
        plt.subplot(414)
        plt.plot(decomposition.resid, label='Residuals')
        plt.legend(loc='best')
        plt.title('Residuals')
        
        plt.tight_layout()
        
        # Create visualizations directory if it doesn't exist
        os.makedirs('visualizations', exist_ok=True)
        output_path = 'visualizations/time_series_decomposition.png'
        plt.savefig(output_path)
        plt.close()
        print(f"Decomposition plot saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during time series decomposition: {e}")
        print("Trying with a different period...")
        # Try with default period if custom period fails
        if period != 24:  # Avoid infinite recursion
            plot_decomposition(series, period=24)

def plot_acf_pacf(series, lags=50):
    """Plot ACF and PACF for the series."""
    # Calculate ACF and PACF
    lag_acf = acf(series, nlags=lags)
    lag_pacf = pacf(series, nlags=lags, method='ols')
    
    # Plot ACF
    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    plt.stem(lag_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(series)), linestyle='--', color='gray')
    plt.axhline(y=1.96/np.sqrt(len(series)), linestyle='--', color='gray')
    plt.title('Autocorrelation Function')
    
    # Plot PACF
    plt.subplot(122)
    plt.stem(lag_pacf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(series)), linestyle='--', color='gray')
    plt.axhline(y=1.96/np.sqrt(len(series)), linestyle='--', color='gray')
    plt.title('Partial Autocorrelation Function')
    
    plt.tight_layout()
    plt.savefig('visualizations/acf_pacf_plots.png')
    plt.close()

def main():
    print("Starting time series preparation...")
    print(f"Process started at: {datetime.datetime.now()}")
    
    # Set up logging
    log_file = setup_logging()
    print(f"Logging to: {log_file}")
    
    # 1. Load preprocessed data
    print("\n=== Loading Data ===")
    print("Loading preprocessed data...")
    df = load_preprocessed_data()
    if df is None:
        print("Failed to load preprocessed data. Exiting...")
        return
    
    # Get the column name (in case it's not the default)
    column_name = df.columns[0]
    series = df[column_name]
    
    # 2. Check stationarity
    print("\n=== Testing Stationarity ===")
    print("Performing Augmented Dickey-Fuller test...")
    p_value = test_stationarity(series)
    
    # 3. Make the series stationary if needed
    if p_value > 0.05:
        print("\n=== Making Series Stationary ===")
        print(f"Time series is not stationary (p-value: {p_value:.6f}). Applying transformations...")
        stationary_series = make_stationary(series)
        
        # Save the stationary series
        stationary_series.to_csv('data/PJM_Load_hourly_SARIMA.csv', header=['Load_Diff'])
        print("\nStationary data saved to: data/PJM_Load_hourly_SARIMA.csv")
        
        # Use the stationary series for further analysis
        series = stationary_series
    else:
        print("\nTime series is already stationary.")
    
    # 4. Time series decomposition
    print("\n=== Time Series Decomposition ===")
    print("Performing time series decomposition...")
    plot_decomposition(series)
    
    # 5. Plot ACF and PACF
    print("\n=== ACF and PACF Analysis ===")
    print("Plotting ACF and PACF...")
    plot_acf_pacf(series)
    
    print("\n=== Time Series Preparation Completed Successfully! ===")
    print(f"\n=== Output Files ===")
    print("- Stationary data: data/PJM_Load_hourly_SARIMA.csv (if created)")
    print("- Visualizations: visualizations/")
    print(f"- Log file: {log_file}")
    print(f"\nProcess completed at: {datetime.datetime.now()}")

if __name__ == "__main__":
    main()
