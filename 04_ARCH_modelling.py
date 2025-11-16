import json
import os
import sys
import datetime
import joblib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj

def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

def rescale_residuals(residuals):
    """Rescale residuals to optimal range for ARCH modeling (1-1000)."""
    residuals = pd.Series(residuals).dropna()
    current_scale = np.abs(residuals).max()
    
    # Target scale between 1 and 1000
    if current_scale > 1000:
        scale_factor = 100 / current_scale  # Scale to around 100
        scaled_residuals = residuals * scale_factor
        print(f"Rescaled residuals from {current_scale:.2e} to {np.abs(scaled_residuals).max():.2f}")
        return scaled_residuals, scale_factor
    elif current_scale < 1:
        scale_factor = 100 / current_scale  # Scale to around 100
        scaled_residuals = residuals * scale_factor
        print(f"Rescaled residuals from {current_scale:.2e} to {np.abs(scaled_residuals).max():.2f}")
        return scaled_residuals, scale_factor
    else:
        print(f"Residuals already in good range: {current_scale:.2f}")
        return residuals, 1.0

def load_sarimax_model():
    """Load SARIMAX model and extract residuals."""
    # Try to load a saved SARIMAX model
    for file in os.listdir('models'):
        if file.startswith('sarimax_model_') and file.endswith('.pkl'):
            file_name = os.path.join('models', file)
            print(f"Loading SARIMAX model from: {file_name}")
            sarimax_model = joblib.load(file_name)
            
            # Get residuals from the model
            residuals = pd.Series(sarimax_model.resid)
            # Create a simple date range for the residuals
            start_date = pd.Timestamp('1998-04-01')
            residuals.index = pd.date_range(start=start_date, periods=len(residuals), freq='H')
            
            print(f"Loaded {len(residuals)} residual observations")
            print(f"Residuals date range: {residuals.index[0]} to {residuals.index[-1]}")
            
            # Rescale residuals for ARCH modeling
            scaled_residuals, scale_factor = rescale_residuals(residuals)
            
            return sarimax_model, scaled_residuals, scale_factor
    
    # If no saved model, create from data
    print("No SARIMAX model found. Creating from data...")
    try:
        # Load data
        data_files = ['data/PJM_Load_hourly_SARIMA.csv', 'data/PJM_Load_hourly_preprocessed.csv']
        df = None
        for file_path in data_files:
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                break
            except FileNotFoundError:
                continue
        
        if df is None:
            raise FileNotFoundError("No data files found")
        
        # Load exogenous variables
        try:
            exog = pd.read_csv('data/exogenous_features_original.csv', index_col=0, parse_dates=True)
        except FileNotFoundError:
            print("Warning: Exogenous variables file not found. Using dummy exog.")
            exog = pd.DataFrame(index=df.index, data=np.ones((len(df), 1)))
        
        # Create simple SARIMAX model
        column_name = df.columns[0]
        series = df[column_name]
        test_size = 365*24
        train = series.iloc[:-test_size]
        exog_train = exog.iloc[:len(train)]
        
        # Fit simplified SARIMAX
        simple_order = [1, 1, 1]
        sarimax_model = SARIMAX(train, exog=exog_train, order=simple_order, seasonal_order=[0, 0, 0, 0])
        sarimax_fit = sarimax_model.fit(disp=False)
        
        residuals = pd.Series(sarimax_fit.resid)
        residuals.index = train.index
        
        # Rescale residuals for ARCH modeling
        scaled_residuals, scale_factor = rescale_residuals(residuals)

        # Save the model
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'models/sarimax_model_{timestamp}.pkl'
        joblib.dump(sarimax_fit, model_path)
        print(f"SARIMAX model saved to: {model_path}")
        
        return sarimax_fit, scaled_residuals, scale_factor
        
    except Exception as e:
        print(f"Error creating SARIMAX model: {e}")
        print("Using differenced series as residuals fallback...")
        # Final fallback: use differenced series
        if df is not None:
            residuals = df.diff().dropna()
            residuals = residuals.iloc[-min(len(residuals), 1000):]
            scaled_residuals, scale_factor = rescale_residuals(residuals)
            return None, scaled_residuals, scale_factor
        return None, None, 1.0

def load_residuals():
    """Load residuals from SARIMA model."""
    try:
        # Load data with fallback
        data_files = ['data/PJM_Load_hourly_SARIMA.csv', 'data/PJM_Load_hourly_preprocessed.csv']
        df = None
        for file_path in data_files:
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                break
            except FileNotFoundError:
                continue
        
        if df is None:
            raise FileNotFoundError("No data files found")
        
        # Use differenced series as residuals placeholder
        return df.diff().dropna()
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def test_arch_effect(residuals, lags=40):
    """Test for ARCH effects in residuals using Ljung-Box test."""
    print("\n=== Testing for ARCH Effects ===")
    
    # Check if residuals are numeric
    residuals = pd.Series(residuals).dropna()
    if not pd.api.types.is_numeric_dtype(residuals):
        print("Warning: Residuals are not numeric. Converting to numeric.")
        residuals = pd.to_numeric(residuals, errors='coerce').dropna()
    
    if len(residuals) == 0:
        print("Error: No valid numeric residuals after cleaning.")
        return False, 0.0, None
    
    # Test for ARCH effects on squared residuals
    squared_residuals = residuals**2
    
    try:
        # Use Ljung-Box test on squared residuals
        lb_test = acorr_ljungbox(squared_residuals, lags=min(lags, len(squared_residuals)//4), return_df=True)
        
        # Check if any lags are significant (p < 0.05)
        significant_lags = lb_test[lb_test['lb_pvalue'] < 0.05]
        has_arch_effect = len(significant_lags) > 0
        
        # Calculate ratio of significant lags
        significant_ratio = len(significant_lags) / len(lb_test) if len(lb_test) > 0 else 0
        
        print(f"Number of lags tested: {len(lb_test)}")
        print(f"Number of significant lags (p < 0.05): {len(significant_lags)}")
        print(f"Ratio of significant lags: {significant_ratio:.1%}")
        
        if has_arch_effect:
            print("ARCH effects detected in residuals.")
        else:
            print("No significant ARCH effects detected.")
        
        return has_arch_effect, significant_ratio, lb_test
        
    except Exception as e:
        print(f"Error in ARCH effect test: {e}")
        return False, 0.0, None

def fit_arch_model_iterative(residuals, max_attempts=6, model_type='both'):
    """Fit ARCH/GARCH model with simplified iterative approach."""
    print(f"\n=== Iterative {model_type.upper()} Fitting ===")
    
    best_model = None
    best_results = None
    best_score = float('inf')
    
    # Simplified model configurations
    model_configs = []
    if model_type in ['arch', 'both']:
        model_configs.extend([
            {'type': 'arch', 'p': 1, 'q': 0, 'differenced': False, 'name': 'ARCH(1)'},
            {'type': 'arch', 'p': 2, 'q': 0, 'differenced': False, 'name': 'ARCH(2)'},
            {'type': 'arch', 'p': 3, 'q': 0, 'differenced': False, 'name': 'ARCH(3)'},
            {'type': 'arch', 'p': 5, 'q': 0, 'differenced': False, 'name': 'ARCH(5)'},
        ])
    if model_type in ['garch', 'both']:
        model_configs.extend([
            {'type': 'garch', 'p': 1, 'q': 1, 'differenced': False, 'name': 'GARCH(1,1)'},
            {'type': 'garch', 'p': 1, 'q': 2, 'differenced': False, 'name': 'GARCH(1,2)'},
            {'type': 'garch', 'p': 2, 'q': 1, 'differenced': False, 'name': 'GARCH(2,1)'},
            {'type': 'garch', 'p': 2, 'q': 2, 'differenced': False, 'name': 'GARCH(2,2)'},
            {'type': 'garch', 'p': 1, 'q': 1, 'differenced': True, 'name': 'GARCH(1,1)-diff'},
        ])
    # Add EGARCH and TGARCH models for better asymmetry handling
    if model_type in ['both']:
        model_configs.extend([
            {'type': 'egarch', 'p': 1, 'q': 1, 'differenced': False, 'name': 'EGARCH(1,1)'},
            {'type': 'garch', 'p': 1, 'q': 1, 'o': 1, 'differenced': False, 'name': 'GJR-GARCH(1,1)'},
        ])
    
    # Limit to max_attempts
    model_configs = model_configs[:max_attempts]
    
    for attempt, config in enumerate(model_configs):
        print(f"\nAttempt {attempt + 1}/{len(model_configs)}: {config['name']}")
        
        residuals_to_use = residuals.diff().dropna() if config['differenced'] else residuals
        
        try:
            # Fit the model
            if config['type'] == 'arch':
                model = arch_model(residuals_to_use, vol='Arch', p=config['p'])
            elif config['type'] == 'egarch':
                model = arch_model(residuals_to_use, vol='EGarch', p=config['p'], q=config['q'])
            elif config['type'] == 'garch' and 'o' in config:
                # GJR-GARCH model
                model = arch_model(residuals_to_use, vol='Garch', p=config['p'], q=config['q'], o=config['o'])
            else:  # Standard GARCH
                model = arch_model(residuals_to_use, vol='Garch', p=config['p'], q=config['q'])
            
            model_fit = model.fit(disp='off')
            
            # Simple evaluation
            std_residuals = residuals_to_use / model_fit.conditional_volatility
            has_arch_effect, significant_ratio, _ = test_arch_effect(std_residuals)
            
            # Calculate score (lower is better)
            score = model_fit.aic
            if has_arch_effect:
                # Heavily penalize models that still have ARCH effects
                score += 10000 * significant_ratio
            else:
                # Reward models that eliminate ARCH effects
                score -= 1000
            
            # Also consider BIC and log-likelihood in the score
            score += 0.5 * model_fit.bic
            score -= 0.1 * model_fit.loglikelihood
            
            print(f"Score: {score:.2f}, ARCH effects remaining: {has_arch_effect}")
            
            # Update best model
            if score < best_score:
                best_score = score
                best_model = model_fit
                best_results = {
                    'model': config['type'].upper(),
                    'p': config['p'],
                    'q': config['q'],
                    'attempt': attempt + 1,
                    'differenced': config['differenced'],
                    'aic': float(model_fit.aic),
                    'has_arch_effect_after_fit': bool(has_arch_effect),
                    'significant_ratio': float(significant_ratio),
                    'score': float(score)
                }
                print("New best model found!")
                
        except Exception as e:
            print(f"Error in attempt {attempt + 1}: {e}")
            continue
    
    if best_model is None:
        print("All fitting attempts failed!")
        return None, None
    
    print(f"\n=== Best Model: {best_results['model']}({best_results['p']}")
    if best_results['q'] > 0:
        print(f",{best_results['q']}")
    print(f") - Score: {best_results['score']:.2f}")
    
    return best_model, best_results

def evaluate_volatility_model(model, residuals):
    """Comprehensive evaluation of ARCH/GARCH model fit."""
    print("\n=== Comprehensive Volatility Model Evaluation ===")
    
    # Get standardized residuals
    std_residuals = residuals / model.conditional_volatility
    
    # 1. ARCH effects test (more nuanced)
    has_arch_effect, significant_ratio, lb_test = test_arch_effect(std_residuals)
    
    # 2. Normality test on standardized residuals
    from scipy import stats
    normality_stat, normality_p = stats.jarque_bera(std_residuals.dropna())
    print(f"\nNormality test (Jarque-Bera):")
    print(f"Statistic: {normality_stat:.2f}, p-value: {normality_p:.4f}")
    if normality_p < 0.05:
        print("Standardized residuals are not normal")
    else:
        print("Standardized residuals appear normal")
    
    # 3. Volatility clustering check
    vol_clustering = (std_residuals.abs() > std_residuals.abs().mean()).rolling(10).mean()
    clustering_ratio = (vol_clustering > 0.6).mean()
    print(f"\nVolatility clustering: {clustering_ratio:.1%} of periods show persistent volatility")
    
    # 4. Information criteria
    print(f"\nModel Information Criteria:")
    print(f"AIC: {model.aic:.2f}")
    print(f"BIC: {model.bic:.2f}")
    
    return {
        'arch_removed': not has_arch_effect,
        'significant_ratio': significant_ratio,
        'normality_p': normality_p,
        'volatility_clustering': clustering_ratio,
        'aic': model.aic,
        'bic': model.bic
    }
    
def evaluate_sarimax_vs_sarimax_volatility(sarimax_model, arch_model, test_data, exog_test, arch_results):
    """Compare performance between SARIMAX and SARIMAX+Volatility models."""
    print("\n=== Performance Comparison: SARIMAX vs SARIMAX+Volatility ===")
    
    # Get SARIMAX predictions
    print("Getting SARIMAX predictions...")
    try:
        # Check if model was trained with exogenous variables
        # SARIMAX models may have exog in different attributes
        has_exog = False
        if hasattr(sarimax_model, 'exog') and sarimax_model.exog is not None:
            has_exog = True
        elif hasattr(sarimax_model, 'model') and hasattr(sarimax_model.model, 'exog'):
            has_exog = True
        elif hasattr(sarimax_model, 'data') and hasattr(sarimax_model.data, 'exog'):
            has_exog = True
        
        if has_exog and exog_test is not None:
            print("Using exogenous variables for SARIMAX forecast")
            # Ensure exog_test has the right number of observations
            if len(exog_test) >= len(test_data):
                exog_forecast = exog_test.iloc[:len(test_data)]
            else:
                # If exog_test is shorter, repeat the last row
                last_exog = exog_test.iloc[-1:].values
                exog_forecast = np.tile(last_exog, (len(test_data), 1))
                exog_forecast = pd.DataFrame(exog_forecast, index=test_data.index)
            
            sarimax_preds = sarimax_model.get_forecast(steps=len(test_data), exog=exog_forecast)
        else:
            print("Using SARIMAX forecast without exogenous variables")
            sarimax_preds = sarimax_model.get_forecast(steps=len(test_data))
        sarimax_mean = sarimax_preds.predicted_mean
        sarimax_conf_int = sarimax_preds.conf_int()
    except Exception as e:
        print(f"Error getting SARIMAX predictions: {e}")
        # Use in-sample predictions as fallback
        fitted_vals = sarimax_model.fittedvalues
        if hasattr(fitted_vals, 'iloc'):
            sarimax_mean = fitted_vals.iloc[-len(test_data):]
        else:
            # It's a numpy array
            sarimax_mean = fitted_vals[-len(test_data):]
        sarimax_conf_int = None
    
    # Calculate SARIMAX metrics
    sarimax_mape = mean_absolute_percentage_error(test_data, sarimax_mean) * 100
    print(f"SARIMAX MAPE: {sarimax_mape:.2f}%")
    
    # For GARCH-enhanced predictions, we need to simulate volatility-adjusted forecasts
    print("Calculating GARCH volatility-adjusted predictions...")
    
    # Get the last few residuals from training to start GARCH forecasting
    train_residuals = pd.Series(sarimax_model.resid)
    
    # Forecast volatility using GARCH model
    garch_forecast_horizon = len(test_data)
    # Use dynamic forecasting for multi-step ahead
    garch_forecast = arch_model.forecast(horizon=1, start=train_residuals.index[-1], reindex=True, method='simulation')
    
    # Get conditional volatility forecasts
    # For multi-step forecasts, we need to simulate or use iterative approach
    volatility_forecast = np.array([])
    last_residuals = train_residuals.copy()
    
    # Simple iterative forecasting approach
    for i in range(len(test_data)):
        if i == 0:
            # Use the last forecast for the first step
            vol = garch_forecast.variance.iloc[-1].values[0]
        else:
            # For subsequent steps, use a simple persistence approach
            vol = volatility_forecast[-1] * 0.95  # Slight decay
        volatility_forecast = np.append(volatility_forecast, vol)
    
    # Create volatility-adjusted predictions (same mean as SARIMAX, different confidence intervals)
    # The point forecasts remain the same, only confidence intervals change
    if isinstance(sarimax_mean, pd.Series):
        sarimax_garch_mean = sarimax_mean.copy()
        mean_index = sarimax_mean.index
    else:
        # It's a numpy array
        sarimax_garch_mean = sarimax_mean.copy()
        mean_index = test_data.index
    
    # Calculate GARCH-adjusted confidence intervals
    # Use the predicted volatility to adjust the confidence intervals
    if sarimax_conf_int is not None:
        # Handle both DataFrame and numpy array formats
        if isinstance(sarimax_conf_int, pd.DataFrame):
            sarimax_ci_width = sarimax_conf_int.iloc[:, 1] - sarimax_conf_int.iloc[:, 0]
        else:
            # It's a numpy array
            sarimax_ci_width = sarimax_conf_int[:, 1] - sarimax_conf_int[:, 0]
        
        # Adjust the confidence interval width based on predicted volatility
        # Scale factor based on the ratio of predicted volatility to historical volatility
        historical_vol = np.std(train_residuals)
        volatility_scale = np.sqrt(volatility_forecast) / historical_vol
        
        # Apply volatility scaling to confidence intervals with a minimum scale factor
        # to ensure the confidence intervals are visible
        min_scale = 0.5  # Minimum 50% of original CI width
        volatility_scale = np.maximum(volatility_scale, min_scale)
        garch_ci_width = sarimax_ci_width * volatility_scale
        
        # Create GARCH-adjusted confidence intervals
        sarimax_garch_conf_int = pd.DataFrame({
            'lower': sarimax_garch_mean - garch_ci_width / 2,
            'upper': sarimax_garch_mean + garch_ci_width / 2
        }, index=mean_index)
    else:
        # If no SARIMAX confidence intervals, create basic ones using volatility
        garch_std = np.sqrt(volatility_forecast)
        sarimax_garch_conf_int = pd.DataFrame({
            'lower': sarimax_garch_mean - 1.96 * garch_std,
            'upper': sarimax_garch_mean + 1.96 * garch_std
        }, index=mean_index)
    
    # Calculate SARIMAX+GARCH metrics (same mean, so same MAPE)
    sarimax_garch_mape = mean_absolute_percentage_error(test_data, sarimax_garch_mean) * 100
    print(f"SARIMAX+GARCH MAPE: {sarimax_garch_mape:.2f}%")
    
    # Calculate improvement
    improvement = ((sarimax_mape - sarimax_garch_mape) / sarimax_mape) * 100
    print(f"Improvement: {improvement:.2f}%")
    
    if improvement > 0:
        print(f"SARIMAX+{arch_results['model']} performs better by {improvement:.2f}%")
    else:
        print(f"SARIMAX alone performs better by {-improvement:.2f}%")
    
    comparison_results = {
        'sarimax_mape': sarimax_mape,
        'sarimax_garch_mape': sarimax_garch_mape,
        'improvement_percent': improvement,
        'sarimax_better': improvement <= 0
    }
    
    return comparison_results, sarimax_mean, sarimax_garch_mean, sarimax_conf_int, sarimax_garch_conf_int

def save_arch_model(model, p, q=0):
    """Save the trained ARCH/GARCH model."""
    print("\n=== Saving ARCH model ===")
    
    # Determine model type based on q parameter
    model_type = 'ARCH' if q == 0 else f'GARCH({q})'
    
    # Generate filename with timestamp and model parameters
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'models/arch_model_{timestamp}_{model_type.lower()}({p}'
    if q > 0:
        filename += f',{q}'
    filename += ').pkl'
    
    # Save the model
    joblib.dump(model, filename)
    print(f"ARCH model saved successfully to: {filename}")
    
    return filename

def plot_confidence_intervals_7days(test_data, sarimax_mean, sarimax_garch_mean, sarimax_conf_int, sarimax_garch_conf_int):
    """Plot confidence intervals for the last 7 days comparing SARIMAX and SARIMAX+GARCH."""
    print("\nGenerating 7-day confidence interval comparison...")
    
    # Get the last 7 days (168 hours) of data
    last_7_days = 7 * 24  # 7 days * 24 hours
    test_data_7d = test_data.iloc[-last_7_days:]
    
    # Handle both pandas Series and numpy arrays for mean predictions
    if isinstance(sarimax_mean, pd.Series):
        sarimax_mean_7d = sarimax_mean.iloc[-last_7_days:]
        sarimax_mean_index = sarimax_mean_7d.index
    else:
        # It's a numpy array
        sarimax_mean_7d = sarimax_mean[-last_7_days:]
        sarimax_mean_index = test_data_7d.index
    
    if isinstance(sarimax_garch_mean, pd.Series):
        sarimax_garch_mean_7d = sarimax_garch_mean.iloc[-last_7_days:]
    else:
        # It's a numpy array
        sarimax_garch_mean_7d = sarimax_garch_mean[-last_7_days:]
    
    if sarimax_conf_int is not None:
        sarimax_conf_int_7d = sarimax_conf_int[-last_7_days:]
        # Convert to DataFrame if it's a numpy array
        if isinstance(sarimax_conf_int_7d, np.ndarray):
            sarimax_conf_int_7d = pd.DataFrame(sarimax_conf_int_7d, index=sarimax_mean_index, columns=['lower', 'upper'])
    else:
        sarimax_conf_int_7d = None
    
    if sarimax_garch_conf_int is not None:
        sarimax_garch_conf_int_7d = sarimax_garch_conf_int[-last_7_days:]
    else:
        sarimax_garch_conf_int_7d = None
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: SARIMAX confidence intervals
    ax1.plot(test_data_7d.index, test_data_7d, label='Actual', color='black', linewidth=2, alpha=0.8)
    ax1.plot(sarimax_mean_index, sarimax_mean_7d, label='SARIMAX Forecast', color='blue', linewidth=2)
    
    if sarimax_conf_int_7d is not None:
        if isinstance(sarimax_conf_int_7d, pd.DataFrame):
            ax1.fill_between(sarimax_conf_int_7d.index, 
                            sarimax_conf_int_7d.iloc[:, 0], 
                            sarimax_conf_int_7d.iloc[:, 1], 
                            color='blue', alpha=0.2, label='SARIMAX 95% CI')
        else:
            # It's a numpy array
            ax1.fill_between(sarimax_mean_index, 
                            sarimax_conf_int_7d[:, 0], 
                            sarimax_conf_int_7d[:, 1], 
                            color='blue', alpha=0.2, label='SARIMAX 95% CI')
    
    ax1.set_title('SARIMAX Forecast with Confidence Intervals (Last 7 Days)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: SARIMAX+GARCH confidence intervals
    ax2.plot(test_data_7d.index, test_data_7d, label='Actual', color='black', linewidth=2, alpha=0.8)
    ax2.plot(sarimax_mean_index, sarimax_garch_mean_7d, label='SARIMAX+GARCH Forecast', color='red', linewidth=2)
    
    if sarimax_garch_conf_int_7d is not None:
        if isinstance(sarimax_garch_conf_int_7d, pd.DataFrame):
            ax2.fill_between(sarimax_garch_conf_int_7d.index, 
                            sarimax_garch_conf_int_7d.iloc[:, 0], 
                            sarimax_garch_conf_int_7d.iloc[:, 1], 
                            color='red', alpha=0.5, label='SARIMAX+GARCH 95% CI')
        else:
            # It's a numpy array
            ax2.fill_between(sarimax_mean_index, 
                            sarimax_garch_conf_int_7d[:, 0], 
                            sarimax_garch_conf_int_7d[:, 1], 
                            color='red', alpha=0.5, label='SARIMAX+GARCH 95% CI')
    
    ax2.set_title('SARIMAX+GARCH Forecast with Volatility-Adjusted Confidence Intervals (Last 7 Days)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Value')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('visualizations/confidence_intervals_7days_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a combined plot showing both models side by side
    plt.figure(figsize=(16, 8))
    
    # Plot actual values
    plt.plot(test_data_7d.index, test_data_7d, label='Actual', color='black', linewidth=2, alpha=0.8)
    
    # Plot SARIMAX forecast and confidence intervals
    plt.plot(sarimax_mean_index, sarimax_mean_7d, label='SARIMAX Forecast', color='blue', linewidth=2, alpha=0.8)
    if sarimax_conf_int_7d is not None:
        if isinstance(sarimax_conf_int_7d, pd.DataFrame):
            plt.fill_between(sarimax_conf_int_7d.index, 
                            sarimax_conf_int_7d.iloc[:, 0], 
                            sarimax_conf_int_7d.iloc[:, 1], 
                            color='blue', alpha=0.15)
        else:
            # It's a numpy array
            plt.fill_between(sarimax_mean_index, 
                            sarimax_conf_int_7d[:, 0], 
                            sarimax_conf_int_7d[:, 1], 
                            color='blue', alpha=0.15)
    
    # Plot SARIMAX+GARCH forecast and confidence intervals
    plt.plot(sarimax_mean_index, sarimax_garch_mean_7d, label='SARIMAX+GARCH Forecast', color='red', linewidth=2, alpha=0.8)
    if sarimax_garch_conf_int_7d is not None:
        if isinstance(sarimax_garch_conf_int_7d, pd.DataFrame):
            plt.fill_between(sarimax_garch_conf_int_7d.index, 
                            sarimax_garch_conf_int_7d.iloc[:, 0], 
                            sarimax_garch_conf_int_7d.iloc[:, 1], 
                            color='red', alpha=0.15)
        else:
            # It's a numpy array
            plt.fill_between(sarimax_mean_index, 
                            sarimax_garch_conf_int_7d[:, 0], 
                            sarimax_garch_conf_int_7d[:, 1], 
                            color='red', alpha=0.15)
    
    plt.title('SARIMAX vs SARIMAX+GARCH: Confidence Intervals Comparison (Last 7 Days)', fontsize=16, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Add text annotations to highlight the difference
    if sarimax_conf_int_7d is not None and sarimax_garch_conf_int_7d is not None:
        # Calculate average confidence interval widths
        if isinstance(sarimax_conf_int_7d, pd.DataFrame):
            sarimax_ci_width = (sarimax_conf_int_7d.iloc[:, 1] - sarimax_conf_int_7d.iloc[:, 0]).mean()
        else:
            # It's a numpy array
            sarimax_ci_width = (sarimax_conf_int_7d[:, 1] - sarimax_conf_int_7d[:, 0]).mean()
        
        if isinstance(sarimax_garch_conf_int_7d, pd.DataFrame):
            garch_ci_width = (sarimax_garch_conf_int_7d.iloc[:, 1] - sarimax_garch_conf_int_7d.iloc[:, 0]).mean()
        else:
            # It's a numpy array
            garch_ci_width = (sarimax_garch_conf_int_7d[:, 1] - sarimax_garch_conf_int_7d[:, 0]).mean()
        
        plt.text(0.02, 0.98, f'Avg CI Width - SARIMAX: {sarimax_ci_width:.2f}', 
                transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='blue', alpha=0.1))
        plt.text(0.02, 0.93, f'Avg CI Width - SARIMAX+GARCH: {garch_ci_width:.2f}', 
                transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.1))
    
    plt.tight_layout()
    plt.savefig('visualizations/confidence_intervals_7days_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Confidence interval plots saved:")
    print("- visualizations/confidence_intervals_7days_comparison.png")
    print("- visualizations/confidence_intervals_7days_combined.png")

def plot_comparison(test_data, sarimax_preds, sarimax_garch_preds):
    """Plot comparison between SARIMAX and SARIMAX+GARCH predictions."""
    print("\nGenerating comparison plot...")
    
    plt.figure(figsize=(15, 8))
    
    # Plot actual values
    plt.plot(test_data.index, test_data, label='Actual', color='black', alpha=0.7)
    
    # Plot SARIMAX predictions
    plt.plot(test_data.index, sarimax_preds, label='SARIMAX', color='blue', alpha=0.8)
    
    # Plot SARIMAX+GARCH predictions
    plt.plot(test_data.index, sarimax_garch_preds, label='SARIMAX+GARCH', color='red', alpha=0.8)
    
    plt.title('SARIMAX vs SARIMAX+GARCH Predictions')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/sarimax_vs_garch_comparison.png')
    plt.close()

def plot_volatility(residuals, conditional_vol):
    """Plot the conditional volatility."""
    plt.figure(figsize=(15, 6))
    
    # Plot residuals
    plt.subplot(2, 1, 1)
    plt.plot(residuals.index, residuals, label='Residuals', alpha=0.7)
    plt.title('Residuals from SARIMA Model')
    plt.legend()
    
    # Plot conditional volatility
    plt.subplot(2, 1, 2)
    plt.plot(conditional_vol.index, conditional_vol, label='Conditional Volatility', color='red')
    plt.title('Conditional Volatility (GARCH)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('visualizations/arch_volatility.png')
    plt.close()

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    """Main function to run ARCH modeling."""
    print("Starting ARCH/GARCH modeling...")
    print(f"Process started at: {datetime.datetime.now()}")
    
    # Setup directories first
    setup_directories()
    
    # Setup logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs/arch_modeling_{timestamp}.log'
    print(f"Logging to: {log_file}")
    
    # Redirect stdout and stderr to log file while also displaying on console
    sys.stdout = Logger(log_file)
    sys.stderr = Logger(log_file)
    
    # 1. Load residuals from SARIMAX model
    print("\n=== Loading Residuals ===")
    
    sarimax_model, residuals, scale_factor = load_sarimax_model()
    if sarimax_model is None and residuals is None:
        print("Failed to load SARIMAX model and residuals")
        return
    
    # 2. Test for ARCH effects
    has_arch_effect, significant_ratio, lb_test = test_arch_effect(residuals)
    
    if not has_arch_effect:
        print("No significant ARCH effects detected. ARCH modeling may not be necessary.")
        return
    
    # 3. Fit ARCH/GARCH model iteratively (testing both)
    print("\n=== Fitting ARCH/GARCH Models ===")
    model, arch_results = fit_arch_model_iterative(residuals, max_attempts=10, model_type='both')
    
    if model is None:
        print("Failed to fit any ARCH model.")
        return
    
    # 4. Save the trained ARCH model
    arch_model_path = save_arch_model(model, p=arch_results['p'], q=arch_results.get('q', 0))
    
    # 5. Comprehensive evaluation of the best model
    print("\n=== Comprehensive Model Evaluation ===")
    fit_evaluation = evaluate_volatility_model(model, residuals)
    
    # 6. Performance testing on test set
    print("\n=== Performance Testing ===")
    
    if sarimax_model is None:
        print("No SARIMAX model available for performance testing")
        return
    
    # Load test data
    data_files = ['data/PJM_Load_hourly_SARIMA.csv', 'data/PJM_Load_hourly_preprocessed.csv']
    df = None
    for file_path in data_files:
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            break
        except FileNotFoundError:
            continue
    
    if df is None:
        print("No test data available")
        return
    
    # Split data for testing
    column_name = df.columns[0]
    series = df[column_name]
    test_size = 365*24
    test = series.iloc[-test_size:]
    
    # Load exogenous variables for test set
    try:
        # Try to load the original exogenous features file first (9 features)
        exog_test = pd.read_csv('data/exogenous_features_original.csv', index_col=0, parse_dates=True)
        exog_test = exog_test.iloc[-len(test):]
        print(f"Loaded exogenous variables with shape: {exog_test.shape}")
    except FileNotFoundError:
        try:
            # Fallback to PCA file (6 features)
            exog_test = pd.read_csv('data/exogenous_features_pca.csv', index_col=0, parse_dates=True)
            exog_test = exog_test.iloc[-len(test):]
            print(f"Loaded PCA exogenous variables with shape: {exog_test.shape}")
            # Pad with zeros to match expected 9 features if needed
            if exog_test.shape[1] < 9:
                padding = np.zeros((len(exog_test), 9 - exog_test.shape[1]))
                exog_test = pd.concat([exog_test, pd.DataFrame(padding, index=exog_test.index)], axis=1)
                print(f"Padded exogenous variables to shape: {exog_test.shape}")
        except FileNotFoundError:
            print("Warning: No exogenous variables file found. Using dummy exog.")
            exog_test = pd.DataFrame(index=test.index, data=np.ones((len(test), 9)))
    
    # Compare performance
    print("\n=== Performance Comparison ===")
    comparison_results, sarimax_preds, sarimax_garch_preds, sarimax_conf_int, sarimax_garch_conf_int = evaluate_sarimax_vs_sarimax_volatility(sarimax_model, model, test, exog_test, arch_results)
    
    # 6. Generate comparison plot
    plot_comparison(test, sarimax_preds, sarimax_garch_preds)
    
    # 7. Generate confidence interval plots for last 7 days
    plot_confidence_intervals_7days(test, sarimax_preds, sarimax_garch_preds, sarimax_conf_int, sarimax_garch_conf_int)
    
    # 7. Generate volatility plot
    conditional_vol = model.conditional_volatility
    plot_volatility(residuals, conditional_vol)
    
    # 8. Save results
    final_results = {
        'arch_model': arch_results,
        'fit_evaluation': fit_evaluation,
        'performance_comparison': comparison_results,
        'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    # Convert numpy types before JSON serialization
    final_results = convert_numpy_types(final_results)
    with open('models/arch_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    
    # 9. Print final summary
    print("\n=== Final Results ===")
    if arch_results['q'] > 0:
        print(f",{arch_results['q']}")
    print(")")
    print(f"Fitting Attempt: {arch_results['attempt']}")
    print(f"Differenced: {arch_results['differenced']}")
    print(f"SARIMAX MAPE: {comparison_results['sarimax_mape']:.2f}%")
    print(f"SARIMAX+{arch_results['model']} MAPE: {comparison_results['sarimax_garch_mape']:.2f}%")
    print(f"Improvement: {comparison_results['improvement_percent']:.2f}%")
    
    print("\n=== Output Files ===")
    print(f"- Results: models/arch_results.json")
    print(f"- Comparison plot: visualizations/sarimax_vs_garch_comparison.png")
    print(f"- 7-day confidence intervals: visualizations/confidence_intervals_7days_comparison.png")
    print(f"- 7-day combined confidence intervals: visualizations/confidence_intervals_7days_combined.png")
    print(f"- Volatility plot: visualizations/arch_volatility.png")
    print(f"- Log file: {log_file}")
    print(f"- ARCH model: {arch_model_path}")
    
    print(f"\nProcess completed at: {datetime.datetime.now()}")

if __name__ == "__main__":
    main()
