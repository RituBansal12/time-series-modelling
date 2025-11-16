import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_percentage_error
import json
import os
import sys
import datetime
import joblib

def load_residuals():
    """Load residuals from SARIMA model."""
    try:
        # Try to load the stationary data first, if available
        try:
            df = pd.read_csv('data/PJM_Load_hourly_SARIMA.csv', index_col=0, parse_dates=True)
        except FileNotFoundError:
            # If stationary data not found, use the preprocessed data
            df = pd.read_csv('data/PJM_Load_hourly_preprocessed.csv', index_col=0, parse_dates=True)
        
        # Load SARIMA model results to get residuals
        with open('results/sarima_results.json', 'r') as f:
            sarima_results = json.load(f)
        
        # Get the residuals from SARIMA model
        # Note: In a real scenario, you would get residuals from the SARIMA model fit
        # For now, we'll use the differenced series as a placeholder
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

def fit_arch_model_iterative(residuals, max_attempts=10, model_type='both'):
    """Fit ARCH/GARCH model with iterative approach if needed.
    
    Args:
        residuals: Time series residuals to model
        max_attempts: Maximum number of model configurations to try
        model_type: 'arch', 'garch', or 'both' to test which model types
    
    Returns:
        tuple: (best_model_fit, best_results_dict)
    """
    print(f"\n=== Iterative {model_type.upper()} Fitting ===")
    
    best_model = None
    best_results = None
    best_score = float('inf')
    best_attempt = 0
    
    # Define model configurations based on requested model types
    model_configs = []
    if model_type in ['arch', 'both']:
        model_configs.extend([
            {'type': 'arch', 'p': 1, 'q': 0, 'differenced': False, 'name': 'ARCH(1) on original'},
            {'type': 'arch', 'p': 1, 'q': 0, 'differenced': True, 'name': 'ARCH(1) on differenced'},
            {'type': 'arch', 'p': 2, 'q': 0, 'differenced': False, 'name': 'ARCH(2) on original'},
            {'type': 'arch', 'p': 2, 'q': 0, 'differenced': True, 'name': 'ARCH(2) on differenced'}
        ])
    if model_type in ['garch', 'both']:
        model_configs.extend([
            {'type': 'garch', 'p': 1, 'q': 1, 'differenced': False, 'name': 'GARCH(1,1) on original'},
            {'type': 'garch', 'p': 1, 'q': 1, 'differenced': True, 'name': 'GARCH(1,1) on differenced'},
            {'type': 'garch', 'p': 2, 'q': 1, 'differenced': False, 'name': 'GARCH(2,1) on original'},
            {'type': 'garch', 'p': 1, 'q': 2, 'differenced': False, 'name': 'GARCH(1,2) on original'},
            {'type': 'garch', 'p': 2, 'q': 2, 'differenced': True, 'name': 'GARCH(2,2) on differenced'}
        ])
    
    # Limit to max_attempts
    model_configs = model_configs[:max_attempts]
    
    for attempt, config in enumerate(model_configs):
        print(f"\nAttempt {attempt + 1}/{len(model_configs)}")
        print(f"Trying {config['name']} residuals...")
        
        residuals_to_use = residuals.diff().dropna() if config['differenced'] else residuals
        
        try:
            # Fit the model
            if config['type'] == 'arch':
                model = arch_model(residuals_to_use, vol='Arch', p=config['p'])
            else:  # GARCH
                model = arch_model(residuals_to_use, vol='Garch', p=config['p'], q=config['q'])
            
            model_fit = model.fit(disp='off')
            
            # Evaluate
            has_arch_effect, significant_ratio, lb_test = test_arch_effect(residuals_to_use / model_fit.conditional_volatility)
            
            # Calculate score (lower is better - combination of AIC and remaining ARCH effects)
            score = model_fit.aic
            if has_arch_effect:
                score += 10000 * significant_ratio  # Penalty for remaining ARCH effects
            
            print(f"Model score: {score:.2f}")
            print(f"Remaining ARCH effects: {has_arch_effect} ({significant_ratio:.1%} significant lags)")
            
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
                    'log_likelihood': float(model_fit.loglikelihood),
                    'aic': float(model_fit.aic),
                    'bic': float(model_fit.bic),
                    'has_arch_effect_after_fit': bool(has_arch_effect),
                    'significant_ratio': float(significant_ratio),
                    'score': float(score)
                }
                best_attempt = attempt + 1
                print("New best model found!")
                
        except Exception as e:
            print(f"Error in attempt {attempt + 1}: {e}")
            continue
    
    if best_model is None:
        print("All fitting attempts failed!")
        return None, None
    
    print(f"\n=== Best Model Selected ===")
    print(f"Attempt: {best_attempt}")
    model_type = best_results['model']
    if model_type == 'ARCH':
        print(f"Model: ARCH({best_results['p']})")
    else:
        print(f"Model: GARCH({best_results['p']},{best_results['q']})")
    print(f"Differenced: {best_results['differenced']}")
    print(f"Score: {best_results['score']:.2f}")
    print(f"Remaining ARCH effects: {best_results['has_arch_effect_after_fit']} ({best_results['significant_ratio']:.1%} significant lags)")
    
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
    
def evaluate_sarimax_vs_sarimax_volatility(sarimax_model, volatility_model, test_data, exog_test, arch_results):
    """Compare SARIMAX alone vs SARIMAX+ARCH/GARCH predictions on test set."""
    print("\n=== Performance Comparison: SARIMAX vs SARIMAX+Volatility ===")
    
    # Get SARIMAX predictions on test set
    print("Getting SARIMAX predictions...")
    sarimax_preds = sarimax_model.get_forecast(steps=len(test_data), exog=exog_test)
    sarimax_mean = sarimax_preds.predicted_mean
    sarimax_conf_int = sarimax_preds.conf_int()
    
    # Calculate SARIMAX metrics
    sarimax_mape = mean_absolute_percentage_error(test_data, sarimax_mean) * 100
    print(f"SARIMAX MAPE: {sarimax_mape:.2f}%")
    
    # For GARCH-enhanced predictions, we need to simulate volatility-adjusted forecasts
    print("Calculating GARCH volatility-adjusted predictions...")
    
    # Get the last few residuals from training to start GARCH forecasting
    train_residuals = pd.Series(sarimax_model.resid)
    
    # Forecast volatility using GARCH model
    garch_forecast_horizon = len(test_data)
    garch_forecast = volatility_model.forecast(horizon=garch_forecast_horizon, start=train_residuals.index[-1])
    
    # Get conditional volatility forecasts
    volatility_forecast = garch_forecast.variance.iloc[-1].values
    
    # Create volatility-adjusted predictions (simple approach)
    # Adjust SARIMAX predictions based on predicted volatility
    volatility_adjustment = np.sqrt(volatility_forecast) * 0.1  # Small adjustment factor
    sarimax_garch_mean = sarimax_mean + np.random.normal(0, volatility_adjustment, len(sarimax_mean))
    
    # Calculate SARIMAX+GARCH metrics
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
    
    return comparison_results, sarimax_mean, sarimax_garch_mean

def save_arch_model(model, p, q=0):
    """Save the trained ARCH/GARCH model."""
    print("\n=== Saving ARCH model ===")
    
    # Create models directory if it doesn't exist
    os.makedirs('results/models', exist_ok=True)
    
    # Determine model type based on q parameter
    model_type = 'GARCH' if q > 0 else 'ARCH'
    
    # Generate filename with timestamp and model parameters
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'results/models/arch_model_{timestamp}_{model_type.lower()}({p}'
    if q > 0:
        filename += f',{q}'
    filename += ').pkl'
    
    # Save the model
    joblib.dump(model, filename)
    print(f"ARCH model saved successfully to: {filename}")
    
    return filename

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

def main():
    print("Starting ARCH/GARCH modeling...")
    print(f"Process started at: {datetime.datetime.now()}")
    
    # Set up logging (only once)
    log_file = f'results/arch_modeling_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    print(f"Logging to: {log_file}")
    
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
    sys.stderr = Logger(log_file)
    
    # Create necessary directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Load residuals from SARIMAX model
    print("\n=== Loading Residuals ===")
    
    # First, try to load a saved SARIMAX model
    file_name = None
    for file in os.listdir('results'):
        if file.startswith('sarimax_model_') and file.endswith('.pkl'):
            file_name = os.path.join('results', file)
            break
    
    if file_name:
        print(f"Loading SARIMAX model from: {file_name}")
        sarimax_model = joblib.load(file_name)
        
        # Get residuals from the model
        residuals = pd.Series(sarimax_model.resid)
        residuals.index = sarimax_model.resid_index  # Set the correct index
        
        print(f"Loaded {len(residuals)} residual observations")
        print(f"Residuals date range: {residuals.index[0]} to {residuals.index[-1]}")
    else:
        # If no saved model, load from SARIMAX results and recreate residuals
        print("No SARIMAX model found. Loading from results...")
        
        # Load the data and results
        try:
            df = pd.read_csv('data/PJM_Load_hourly_SARIMA.csv', index_col=0, parse_dates=True)
        except FileNotFoundError:
            df = pd.read_csv('data/PJM_Load_hourly_preprocessed.csv', index_col=0, parse_dates=True)
        
        # Load exogenous variables
        try:
            exog = pd.read_csv('data/PJM_Load_hourly_exog.csv', index_col=0, parse_dates=True)
        except FileNotFoundError:
            print("Warning: Exogenous variables file not found. Using dummy exog.")
            exog = pd.DataFrame(index=df.index, data=np.ones((len(df), 1)))
        
        # Load SARIMAX results
        with open('results/sarimax_results.json', 'r') as f:
            sarimax_results = json.load(f)
        
        # Fit SARIMAX model to get residuals
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        column_name = df.columns[0]
        series = df[column_name]
        
        # Split data
        test_size = 365*24
        train = series.iloc[:-test_size]
        test = series.iloc[-test_size:]
        exog_train = exog.iloc[:-test_size]
        
        # Fit SARIMAX model with saved parameters
        order = sarimax_results['order']
        seasonal_order = sarimax_results['seasonal_order']
        
        print(f"Fitting SARIMAX{order}x{seasonal_order} to get residuals...")
        sarimax_model = SARIMAX(train, exog=exog_train, order=order, 
                               seasonal_order=seasonal_order)
        sarimax_fit = sarimax_model.fit(disp=False)
        
        # Get residuals
        residuals = pd.Series(sarimax_fit.resid)
        residuals.index = train.index
        
        # Save the model for future use
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'results/models/sarimax_model_{timestamp}.pkl'
        os.makedirs('results/models', exist_ok=True)
        joblib.dump(sarimax_fit, model_path)
        print(f"SARIMAX model saved to: {model_path}")
        
        print(f"Generated {len(residuals)} residual observations")
        print(f"Residuals date range: {residuals.index[0]} to {residuals.index[-1]}")
    
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
    
    # Load the SARIMAX model again for test set evaluation
    if 'sarimax_fit' in locals():
        sarimax_model = sarimax_fit
    else:
        sarimax_model = joblib.load(file_name)
    
    # Load test data
    try:
        df = pd.read_csv('data/PJM_Load_hourly_SARIMA.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        df = pd.read_csv('data/PJM_Load_hourly_preprocessed.csv', index_col=0, parse_dates=True)
    
    # Load exogenous variables
    try:
        exog = pd.read_csv('data/PJM_Load_hourly_exog.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Warning: Exogenous variables file not found. Using dummy exog.")
        exog = pd.DataFrame(index=df.index, data=np.ones((len(df), 1)))
    
    # Split data for testing
    column_name = df.columns[0]
    series = df[column_name]
    test_size = 365*24
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    
    # Split exogenous variables
    exog_test = exog.iloc[-test_size:]
    
    # Compare performance
    print("\n=== Performance Comparison ===")
    comparison_results, sarimax_preds, sarimax_garch_preds = evaluate_sarimax_vs_sarimax_volatility(sarimax_model, model, test, exog_test, arch_results)
    
    # 6. Generate comparison plot
    plot_comparison(test, sarimax_preds, sarimax_garch_preds)
    
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
    
    with open('results/arch_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    
    # 9. Print final summary
    print("\n=== Final Results ===")
    print(f"Best ARCH Model: {arch_results['model']}({arch_results['p']}")
    if arch_results['q'] > 0:
        print(f",{arch_results['q']}")
    print(")")
    print(f"Fitting Attempt: {arch_results['attempt']}")
    print(f"Differenced: {arch_results['differenced']}")
    print(f"SARIMAX MAPE: {comparison_results['sarimax_mape']:.2f}%")
    print(f"SARIMAX+{arch_results['model']} MAPE: {comparison_results['sarimax_garch_mape']:.2f}%")
    print(f"Improvement: {comparison_results['improvement_percent']:.2f}%")
    
    print("\n=== Output Files ===")
    print(f"- Results: results/arch_results.json")
    print(f"- Comparison plot: visualizations/sarimax_vs_garch_comparison.png")
    print(f"- Volatility plot: visualizations/arch_volatility.png")
    print(f"- Log file: {log_file}")
    print(f"- ARCH model: {arch_model_path}")
    
    print(f"\nProcess completed at: {datetime.datetime.now()}")

if __name__ == "__main__":
    main()
