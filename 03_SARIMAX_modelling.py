import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error
import json
import os
import sys
import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def setup_logging():
    os.makedirs('results', exist_ok=True)
    log_file = f'results/sarimax_modeling_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
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

def load_data():
    """Load the time series data and exogenous features."""
    try:
        # Try to load preprocessed data
        if os.path.exists('data/PJM_Load_hourly_SARIMA.csv'):
            df = pd.read_csv('data/PJM_Load_hourly_SARIMA.csv', 
                           index_col=0, 
                           parse_dates=True)
            is_stationary = True
        else:
            # Fall back to preprocessed data
            df = pd.read_csv('data/PJM_Load_hourly_preprocessed.csv', 
                           index_col=0, 
                           parse_dates=True)
            is_stationary = False
        
        # Load exogenous features
        exog = pd.read_csv('data/exogenous_features.csv', 
                          index_col=0, 
                          parse_dates=True)
        
        # Ensure the index is a datetime and set frequency
        df.index = pd.to_datetime(df.index)
        exog.index = pd.to_datetime(exog.index)
        
        # Align indices
        df = df.reindex(exog.index)
        df = df.sort_index()
        df = df.asfreq('H')
        
        # Ensure all data is numeric
        df = df.apply(pd.to_numeric, errors='coerce')
        exog = exog.apply(pd.to_numeric, errors='coerce')
        
        # Forward fill any missing values
        df = df.ffill()
        exog = exog.ffill()
        
        return df, exog, is_stationary
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, False

def split_data(series, test_size=365*24):  # Default: last year of hourly data
    """Split data into train and test sets."""
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    return train, test

def find_best_arima(train, exog=None, subset_frac=0.2):
    """Find the best ARIMA parameters using auto_arima with optional exogenous variables."""
    print(f"Searching for best ARIMA parameters using most recent {subset_frac*100:.0f}% of data...")
    
    # Take the most recent subset of the data
    subset_size = int(len(train) * subset_frac)
    train_subset = train.iloc[-subset_size:]
    exog_subset = exog.iloc[-subset_size:] if exog is not None else None
    
    print(f"Using most recent {len(train_subset)} out of {len(train)} points for parameter search")
    print(f"Date range for subset: {train_subset.index.min()} to {train_subset.index.max()}")
    
    try:
        # Convert to numpy arrays to avoid pandas type issues
        y = train_subset.values
        X = exog_subset.values if exog_subset is not None else None
        
        # Use a more constrained parameter space
        stepwise_fit = auto_arima(
            y,
            X=X,                   # Use numpy array instead of pandas
            start_p=1, start_q=1,
            max_p=2, max_q=2,
            start_P=0, start_Q=0,
            max_P=1, max_Q=1,
            m=24,                  # Daily seasonality for hourly data
            d=None, D=None,        # Let model decide differencing
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            information_criterion='aic',
            max_order=5,
            n_jobs=1
        )
        
        print("\n=== Best Parameters Found ===")
        print(stepwise_fit.summary())
        return stepwise_fit.order, stepwise_fit.seasonal_order if hasattr(stepwise_fit, 'seasonal_order') else (0, 0, 0, 0)
        
    except Exception as e:
        print(f"Error in auto_arima: {e}")
        print("Using default SARIMA(1,1,1)(1,1,1,24) parameters.")
        return (1, 1, 1), (1, 1, 1, 24)

def train_sarimax(train, order, seasonal_order, exog=None):
    """Train SARIMAX model with given parameters and optional exogenous variables."""
    print("\nTraining SARIMAX model...")
    print(f"Using order: {order}, seasonal_order: {seasonal_order}")
    if exog is not None:
        print(f"Using {exog.shape[1]} exogenous features")
    
    try:
        # Convert to numpy arrays to avoid pandas type issues
        y = train.values if hasattr(train, 'values') else train
        X = exog.values if exog is not None else None
        
        model = SARIMAX(
            y,
            exog=X,  # Use numpy array instead of pandas
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        model_fit = model.fit(disp=False)
        print(model_fit.summary())
        return model_fit
        
    except Exception as e:
        print(f"Error fitting SARIMAX model: {e}")
        print("Trying a simpler model...")
        try:
            model = SARIMAX(
                y,
                exog=X,
                order=(1, 1, 1),
                seasonal_order=(0, 1, 1, 24),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            model_fit = model.fit(disp=False)
            print(model_fit.summary())
            return model_fit
        except Exception as e2:
            print(f"Failed to fit fallback model: {e2}")
            raise

def evaluate_model(model, train, test, exog_test=None, is_stationary=False):
    """Evaluate the model and make predictions."""
    # Make predictions
    start = len(train)
    end = len(train) + len(test) - 1
    
    # Get the corresponding exogenous variables for the test period
    exog_forecast = exog_test.values if exog_test is not None else None
    
    # Make predictions with exogenous variables if available
    preds = model.predict(start=start, end=end, exog=exog_forecast, dynamic=False)
    
    # If we differenced the data, we need to reverse it
    if not is_stationary:
        # This is a simplified version - you might need to adjust based on your differencing
        preds = train.iloc[-1] + preds.cumsum()
    
    # Calculate MAPE
    mape = np.mean(np.abs((test - preds) / test)) * 100
    
    # Store results
    results = {
        'mape': mape,
        'order': model.order,
        'seasonal_order': model.seasonal_order,
        'model_type': 'SARIMAX' if exog_test is not None else 'SARIMA'
    }
    
    return results, preds

def plot_predictions(train, test, preds, is_stationary):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(15, 6))
    
    # Plot training data
    plt.plot(train.index[-24*30:], train[-24*30:], label='Training Data')
    
    # Plot test data
    plt.plot(test.index, test, label='Actual', color='blue', alpha=0.6)
    
    # Plot predictions
    plt.plot(test.index, preds, label='Predicted', color='red', linestyle='--')
    
    plt.title('SARIMAX Model: Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Load (MW)' if not is_stationary else 'Differenced Load')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizations/sarimax_predictions.png')
    plt.close()

def main():
    print("Starting SARIMAX modeling...")
    print(f"Process started at: {datetime.datetime.now()}")
    
    # Set up logging
    log_file = setup_logging()
    print(f"Logging to: {log_file}")
    
    # 1. Load data and exogenous features
    print("\n=== Loading data ===")
    df, exog, is_stationary = load_data()
    if df is None or exog is None:
        print("Failed to load data or exogenous features. Exiting...")
        return
    
    # Get the column name (in case it's not the default)
    column_name = df.columns[0]
    series = df[column_name]
    
    # 2. Split data and exogenous features
    print("\n=== Splitting data ===")
    print("Splitting data into train and test sets...")
    train, test = split_data(series)
    
    # Split exogenous variables to match train/test split
    exog_train = exog.iloc[:-len(test)]
    exog_test = exog.iloc[-len(test):]
    
    print(f"Training data: {len(train)} samples")
    print(f"Test data: {len(test)} samples")
    print(f"Exogenous features: {exog.shape[1]} features"
          f" ({', '.join(exog.columns.tolist()[:5])}{', ...' if len(exog.columns) > 5 else ''})")
    
    # 3. Find best SARIMAX parameters with exogenous variables
    print("\n=== Finding best SARIMAX parameters ===")
    order, seasonal_order = find_best_arima(train, exog=exog_train, subset_frac=0.2)
    print(f"\n=== Best parameters found ===")
    print(f"Order: {order}")
    print(f"Seasonal Order: {seasonal_order}")
    
    # 4. Train SARIMAX model with exogenous variables
    print("\n=== Training SARIMAX model ===")
    print(f"Using order: {order}")
    print(f"Using seasonal_order: {seasonal_order}")
    model = train_sarimax(train, order, seasonal_order, exog=exog_train)
    
    # 5. Evaluate model with exogenous variables
    print("\n=== Evaluating model ===")
    results, preds = evaluate_model(model, train, test, exog_test=exog_test, is_stationary=is_stationary)
    
    # 6. Plot results
    print("Generating prediction plots...")
    plot_predictions(train, test, preds, is_stationary)
    
    # 7. Save results
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'sarimax_results.json')
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n=== SARIMAX modeling completed successfully! ===")
    print(f"\n=== Results Summary ===")
    print(f"Best Model: SARIMAX{order}x{seasonal_order}")
    print(f"Exogenous Features: {exog.shape[1]} features")
    print(f"MAPE: {results['mape']:.2f}%")
    print(f"\n=== Output Files ===")
    print(f"- Model results: {results_file}")
    print(f"- Prediction plot: visualizations/sarimax_predictions.png")
    print(f"- Log file: {log_file}")
    print(f"\nProcess completed at: {datetime.datetime.now()}")

if __name__ == "__main__":
    main()