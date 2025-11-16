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
import logging
import pickle

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def setup_logging():
    """Set up comprehensive logging with both file and console handlers."""
    os.makedirs('logs', exist_ok=True)
    log_file = f'logs/sarimax_modeling_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # Create logger
    logger = logging.getLogger('sarimax_modeling')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler (logs everything)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler (logs INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger, log_file

def load_data():
    """Load the time series data and exogenous features."""
    logger = logging.getLogger('sarimax_modeling')
    logger.info("Starting data loading process...")
    
    try:
        # Try to load preprocessed data
        if os.path.exists('data/PJM_Load_hourly_SARIMA.csv'):
            logger.debug("Loading stationary data from PJM_Load_hourly_SARIMA.csv")
            df = pd.read_csv('data/PJM_Load_hourly_SARIMA.csv', 
                           index_col=0, 
                           parse_dates=True)
            is_stationary = True
            logger.info("Loaded stationary SARIMA data")
        else:
            logger.debug("Loading preprocessed data from PJM_Load_hourly_preprocessed.csv")
            # Fall back to preprocessed data
            df = pd.read_csv('data/PJM_Load_hourly_preprocessed.csv', 
                           index_col=0, 
                           parse_dates=True)
            is_stationary = False
            logger.info("Loaded preprocessed data")
        
        # Load exogenous features
        logger.debug("Loading exogenous features from exogenous_features_original.csv")
        exog = pd.read_csv('data/exogenous_features_original.csv', 
                          index_col=0, 
                          parse_dates=True)
        
        # Ensure the index is a datetime and set frequency
        df.index = pd.to_datetime(df.index)
        exog.index = pd.to_datetime(exog.index)
        
        # Sort both datasets
        df = df.sort_index()
        exog = exog.sort_index()
        
        # Only keep common indices to ensure alignment
        common_index = df.index.intersection(exog.index)
        logger.debug(f"Aligning datasets with {len(common_index)} common timestamps")
        df = df.loc[common_index]
        exog = exog.loc[common_index]
        
        # Ensure all data is numeric
        df = df.apply(pd.to_numeric, errors='coerce')
        exog = exog.apply(pd.to_numeric, errors='coerce')
        
        # Forward fill any missing values
        missing_before = df.isnull().sum().sum() + exog.isnull().sum().sum()
        df = df.ffill()
        exog = exog.ffill()
        missing_after = df.isnull().sum().sum() + exog.isnull().sum().sum()
        
        if missing_before > 0:
            logger.debug(f"Filled {missing_before - missing_after} missing values using forward fill")
        
        logger.info(f"Data loaded successfully. Shape: {df.shape}, Exogenous features: {exog.shape}")
        logger.debug(f"Date range: {df.index.min()} to {df.index.max()}")
        logger.debug(f"Exogenous features: {', '.join(exog.columns.tolist())}")
        
        return df, exog, is_stationary

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        logger.error("Please ensure data files exist in the 'data' directory")
        return None, None, False

def split_data(series, test_size=365*24):  # Default: last year of hourly data
    """Split data into train and test sets."""
    logger = logging.getLogger('sarimax_modeling')
    logger.debug(f"Splitting data with test_size={test_size} samples")
    
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    
    logger.info(f"Data split: Train={len(train)} samples, Test={len(test)} samples")
    logger.debug(f"Train range: {train.index.min()} to {train.index.max()}")
    logger.debug(f"Test range: {test.index.min()} to {test.index.max()}")
    
    return train, test

def find_best_arima(train, exog=None, subset_frac=0.2):
    """Find the best ARIMA parameters using auto_arima with optional exogenous variables."""
    logger = logging.getLogger('sarimax_modeling')
    logger.info(f"Searching for best ARIMA parameters using most recent {subset_frac*100:.0f}% of data...")
    
    # Take the most recent subset of the data
    subset_size = int(len(train) * subset_frac)
    train_subset = train.iloc[-subset_size:]
    exog_subset = exog.iloc[-subset_size:] if exog is not None else None
    
    logger.info(f"Using most recent {len(train_subset)} out of {len(train)} points for parameter search")
    logger.info(f"Date range for subset: {train_subset.index.min()} to {train_subset.index.max()}")
    
    if exog_subset is not None:
        logger.debug(f"Using {exog_subset.shape[1]} exogenous features for parameter search")
    
    try:
        logger.debug("Starting auto_arima parameter search")
        # Convert to numpy arrays to avoid pandas type issues
        y = train_subset.values
        X = exog_subset.values if exog_subset is not None else None
        
        # Use a more constrained parameter space
        stepwise_fit = auto_arima(
            y,
            X=X,                   # Use numpy array instead of pandas
            start_p=1, start_q=1,
            max_p=3, max_q=3,
            start_P=1, start_Q=1,
            max_P=2, max_Q=2,
            m=24,                  # Daily seasonality for hourly data
            d=None, D=None,        # Let model decide differencing
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            information_criterion='aic',
            max_order=None
        )
        
        logger.info("=== Best Parameters Found ===")
        logger.info(f"ARIMA order: {stepwise_fit.order}")
        if hasattr(stepwise_fit, 'seasonal_order'):
            logger.info(f"Seasonal order: {stepwise_fit.seasonal_order}")
        logger.info(f"AIC: {stepwise_fit.aic():.2f}")
        
        return stepwise_fit.order, stepwise_fit.seasonal_order if hasattr(stepwise_fit, 'seasonal_order') else (0, 0, 0, 0)
        
    except Exception as e:
        logger.error(f"Error in auto_arima: {e}")
        logger.warning("Using default SARIMA(1,1,1)(1,1,1,24) parameters.")
        return (1, 1, 1), (1, 1, 1, 24)

def train_sarimax(train, order, seasonal_order, exog=None):
    """Train SARIMAX model with given parameters and optional exogenous variables."""
    logger = logging.getLogger('sarimax_modeling')
    logger.info("Training SARIMAX model...")
    logger.info(f"Using order: {order}, seasonal_order: {seasonal_order}")
    
    if exog is not None:
        logger.info(f"Using {exog.shape[1]} exogenous features")
    
    try:
        logger.debug("Preparing data for SARIMAX model")
        # Convert to numpy arrays to avoid pandas type issues
        y = train.values if hasattr(train, 'values') else train
        X = exog.values if exog is not None else None
        
        logger.debug("Creating SARIMAX model")
        model = SARIMAX(
            y,
            exog=X,  # Use numpy array instead of pandas
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        logger.info("Fitting SARIMAX model...")
        model_fit = model.fit(disp=False)
        logger.info("Model fitting completed successfully")
        
        logger.debug(f"Model AIC: {model_fit.aic:.2f}")
        logger.debug(f"Model BIC: {model_fit.bic:.2f}")
        
        return model_fit
        
    except Exception as e:
        logger.error(f"Error fitting SARIMAX model: {e}")
        logger.warning("Trying a simpler model...")
        try:
            logger.debug("Attempting fallback model with simpler parameters")
            model = SARIMAX(
                y,
                exog=X,
                order=(1, 1, 1),
                seasonal_order=(0, 1, 1, 24),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            model_fit = model.fit(disp=False)
            logger.info("Fallback model fitted successfully")
            logger.debug(f"Fallback model AIC: {model_fit.aic:.2f}")
            return model_fit
        except Exception as e2:
            logger.error(f"Failed to fit fallback model: {e2}")
            raise

def evaluate_model(model_fit, train, test, exog_test=None, order=None, seasonal_order=None):
    """Evaluate the model and make predictions."""
    logger = logging.getLogger('sarimax_modeling')
    logger.info("Evaluating model performance...")
    
    # Make predictions
    start = len(train)
    end = len(train) + len(test) - 1
    
    logger.debug(f"Making predictions for {len(test)} samples")
    logger.debug(f"Prediction range: index {start} to {end}")
    
    # Get the corresponding exogenous variables for the test period
    exog_forecast = exog_test.values if exog_test is not None else None
    
    if exog_forecast is not None:
        logger.debug(f"Using {exog_forecast.shape[1]} exogenous features for predictions")
    
    try:
        # Make predictions with exogenous variables if available
        logger.debug("Generating predictions...")
        preds = model_fit.predict(start=start, end=end, exog=exog_forecast, dynamic=False)
        logger.info("Predictions generated successfully")
        
        # Calculate MAPE
        logger.debug("Calculating performance metrics")
        mape = np.mean(np.abs((test - preds) / test)) * 100
        
        logger.info(f"Model Performance - MAPE: {mape:.2f}%")
        
        # Store results
        results = {
            'mape': mape,
            'order': order,
            'seasonal_order': seasonal_order,
            'model_type': 'SARIMAX' if exog_test is not None else 'SARIMA'
        }
        
        logger.debug(f"Results stored: {results}")
        return results, preds
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

def plot_predictions(train, test, preds, is_stationary):
    """Plot actual vs predicted values."""
    logger = logging.getLogger('sarimax_modeling')
    logger.info("Generating prediction plots...")
    
    try:
        logger.debug("Creating plot figure")
        plt.figure(figsize=(15, 6))
        
        # Plot training data
        logger.debug(f"Plotting last {24*30} training samples")
        plt.plot(train.index[-24*30:], train[-24*30:], label='Training Data')
        
        # Plot test data
        logger.debug(f"Plotting {len(test)} test samples")
        plt.plot(test.index, test, label='Actual', color='blue', alpha=0.6)
        
        # Plot predictions
        logger.debug(f"Plotting {len(preds)} predicted samples")
        plt.plot(test.index, preds, label='Predicted', color='red', linestyle='--')
        
        plt.title('SARIMAX Model: Actual vs Predicted')
        plt.xlabel('Date')
        plt.ylabel('Load (MW)' if not is_stationary else 'Differenced Load')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot
        plot_file = 'visualizations/sarimax_predictions.png'
        plt.savefig(plot_file)
        plt.close()
        
        logger.info(f"Prediction plot saved to: {plot_file}")
        
    except Exception as e:
        logger.error(f"Error generating prediction plot: {e}")
        raise

def plot_last_7_days(test, preds, is_stationary):
    """Plot the last 7 days of actuals vs predictions."""
    logger = logging.getLogger('sarimax_modeling')
    logger.info("Generating 7-day prediction plot...")
    
    try:
        # Get the last 7 days (168 hours for hourly data)
        last_7_days = 7 * 24
        if len(test) < last_7_days:
            last_7_days = len(test)
            logger.warning(f"Test data has only {len(test)} samples, plotting all available data")
        
        logger.debug(f"Plotting last {last_7_days} samples (7 days of hourly data)")
        
        # Create figure
        plt.figure(figsize=(15, 8))
        
        # Slice the last 7 days
        test_last_7 = test[-last_7_days:]
        preds_last_7 = preds[-last_7_days:]
        
        # Plot actual values
        plt.plot(test_last_7.index, test_last_7, label='Actual', color='blue', linewidth=2, alpha=0.8)
        
        # Plot predicted values
        plt.plot(test_last_7.index, preds_last_7, label='Predicted', color='red', linestyle='--', linewidth=2, alpha=0.8)
        
        # Formatting
        plt.title('SARIMAX Model: Last 7 Days - Actual vs Predicted', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Load (MW)' if not is_stationary else 'Differenced Load', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Save the plot
        plot_file = 'visualizations/sarimax_last_7_days.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"7-day prediction plot saved to: {plot_file}")
        
        # Calculate and log MAPE for the last 7 days
        mape_7_days = np.mean(np.abs((test_last_7 - preds_last_7) / test_last_7)) * 100
        logger.info(f"Last 7 days MAPE: {mape_7_days:.2f}%")
        
    except Exception as e:
        logger.error(f"Error generating 7-day prediction plot: {e}")
        raise

def save_model(model_fit, order, seasonal_order, exog_features=None, results=None, is_stationary=False):
    """Save the trained SARIMAX model and its metadata.
    
    Args:
        model_fit: Trained SARIMAX model object
        order: ARIMA order tuple (p, d, q)
        seasonal_order: Seasonal order tuple (P, D, Q, s)
        exog_features: List of exogenous feature names (optional)
        results: Model evaluation results (optional)
        is_stationary: Whether the data was stationary (bool)
    
    Returns:
        model_file: Path to the saved model file
        metadata_file: Path to the saved metadata file
    """
    logger = logging.getLogger('sarimax_modeling')
    logger.info("Saving trained model and metadata...")
    
    try:
        # Create models directory if it doesn't exist
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Generate timestamp for model files
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Model file paths
        model_file = os.path.join(models_dir, f'sarimax_model_{timestamp}.pkl')
        metadata_file = os.path.join(models_dir, f'sarimax_metadata_{timestamp}.json')
        
        # Save the model using pickle
        logger.debug(f"Saving model to: {model_file}")
        with open(model_file, 'wb') as f:
            pickle.dump(model_fit, f)
        
        # Create metadata dictionary
        metadata = {
            'model_type': 'SARIMAX' if exog_features else 'SARIMA',
            'order': order,
            'seasonal_order': seasonal_order,
            'exogenous_features': exog_features if exog_features is not None else [],
            'num_exogenous_features': len(exog_features) if exog_features is not None else 0,
            'is_stationary_data': is_stationary,
            'model_timestamp': timestamp,
            'model_creation_date': datetime.datetime.now().isoformat(),
            'model_aic': float(model_fit.aic) if hasattr(model_fit, 'aic') else None,
            'model_bic': float(model_fit.bic) if hasattr(model_fit, 'bic') else None,
            'model_log_likelihood': float(model_fit.llf) if hasattr(model_fit, 'llf') else None,
            'training_samples': len(model_fit.data.endog) if hasattr(model_fit, 'data') else None,
            'evaluation_results': results if results else {},
            'software_info': {
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'statsmodels_version': getattr(model_fit, '_results', None).__class__.__module__.split('.')[0] if hasattr(model_fit, '_results') else 'unknown'
            }
        }
        
        # Save metadata
        logger.debug(f"Saving metadata to: {metadata_file}")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved successfully: {model_file}")
        logger.info(f"Metadata saved successfully: {metadata_file}")
        
        return model_file, metadata_file
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def load_model(model_file):
    """Load a saved SARIMAX model and its metadata.
    
    Args:
        model_file: Path to the model file.
    
    Returns:
        tuple: (model_fit, metadata) or (None, None) if loading fails
    """
    logger = logging.getLogger('sarimax_modeling')
    
    try:
        # Derive metadata file path from model file path
        base_name = os.path.splitext(model_file)[0]
        metadata_file = f"{base_name.replace('_model_', '_metadata_')}.json"
        logger.info(f"Loading model from: {model_file}")
        
        # Load the model
        logger.debug("Loading model object...")
        with open(model_file, 'rb') as f:
            model_fit = pickle.load(f)
        
        # Load metadata
        logger.debug("Loading model metadata...")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Model loaded successfully: {metadata.get('model_type', 'Unknown')}")
        logger.info(f"Model parameters: {metadata.get('order', 'Unknown')}x{metadata.get('seasonal_order', 'Unknown')}")
        logger.info(f"Model created: {metadata.get('model_creation_date', 'Unknown')}")
        
        return model_fit, metadata
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None

def main():
    """Main function to run SARIMAX modeling pipeline."""
    # Set up logging first
    logger, log_file = setup_logging()
    
    logger.info("=== Starting SARIMAX Modeling Pipeline ===")
    logger.info(f"Process started at: {datetime.datetime.now()}")
    
    try:
        # 1. Load data and exogenous features
        logger.info("=== Loading Data ===")
        df, exog, is_stationary = load_data()
        if df is None or exog is None:
            logger.error("Failed to load data or exogenous features. Exiting...")
            return
        
        # Get the column name (in case it's not the default)
        column_name = df.columns[0]
        series = df[column_name]
        
        # 2. Split data and exogenous features
        logger.info("=== Splitting Data ===")
        test_size = 365*24  # Use the same test_size as split_data function
        train, test = split_data(series, test_size=test_size)
        
        # Split exogenous variables using the same test_size to ensure alignment
        exog_train = exog.iloc[:-test_size]
        exog_test = exog.iloc[-test_size:]
        
        logger.info(f"Training data: {len(train)} samples")
        logger.info(f"Test data: {len(test)} samples")
        logger.info(f"Exogenous features: {exog.shape[1]} features"
                   f" ({', '.join(exog.columns.tolist()[:5])}{', ...' if len(exog.columns) > 5 else ''})")
        
        # 3. Find best SARIMAX parameters with exogenous variables
        logger.info("=== Finding Best SARIMAX Parameters ===")
        order, seasonal_order = find_best_arima(train, exog=exog_train, subset_frac=0.2)
        logger.info("=== Best Parameters Found ===")
        logger.info(f"Order: {order}")
        logger.info(f"Seasonal Order: {seasonal_order}")
        
        # 4. Train SARIMAX model with exogenous variables
        logger.info("=== Training SARIMAX Model ===")
        model = train_sarimax(train, order, seasonal_order, exog=exog_train)
        
        # 5. Evaluate model with exogenous variables
        logger.info("=== Evaluating Model ===")
        results, preds = evaluate_model(model, train, test, exog_test=exog_test, order=order, seasonal_order=seasonal_order)
        
        # 5.5. Save the trained model with evaluation results
        logger.info("=== Saving Model ===")
        exog_feature_names = exog.columns.tolist() if exog is not None else None
        model_file, metadata_file = save_model(
            model_fit=model,
            order=order,
            seasonal_order=seasonal_order,
            exog_features=exog_feature_names,
            results=results,
            is_stationary=is_stationary
        )
        
        # 6. Plot results
        logger.info("=== Generating Visualizations ===")
        plot_predictions(train, test, preds, is_stationary)
        plot_last_7_days(test, preds, is_stationary)
        
        # 7. Save results
        logger.info("=== Saving Results ===")
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, 'sarimax_results.json')
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        
        # Final summary
        logger.info("=== SARIMAX Modeling Completed Successfully! ===")
        logger.info("=== Results Summary ===")
        logger.info(f"Best Model: SARIMAX{order}x{seasonal_order}")
        logger.info(f"Exogenous Features: {exog.shape[1]} features")
        logger.info(f"MAPE: {results['mape']:.2f}%")
        logger.info("=== Output Files ===")
        logger.info(f"- Model results: {results_file}")
        logger.info(f"- Trained model: {model_file}")
        logger.info(f"- Model metadata: {metadata_file}")
        logger.info(f"- Prediction plot: visualizations/sarimax_predictions.png")
        logger.info(f"- 7-day prediction plot: visualizations/sarimax_last_7_days.png")
        logger.info(f"- Log file: {log_file}")
        logger.info(f"Process completed at: {datetime.datetime.now()}")
        
    except Exception as e:
        logger.error(f"Fatal error in main pipeline: {e}")
        logger.error("SARIMAX modeling pipeline failed")
        raise

if __name__ == "__main__":
    main()