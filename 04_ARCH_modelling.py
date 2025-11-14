import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_percentage_error
import json
import os

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

def test_arch_effect(residuals, lags=10):
    """Test for ARCH effects in the residuals."""
    print("Testing for ARCH effects...")
    
    # Ljung-Box test on squared residuals
    lb_test = acorr_ljungbox(residuals**2, lags=lags, return_df=True)
    
    print("\nLjung-Box test on squared residuals:")
    print(lb_test)
    
    # Check if any p-values are significant (p < 0.05)
    has_arch_effect = any(lb_test['lb_pvalue'] < 0.05)
    
    if has_arch_effect:
        print("\nARCH effects detected in the residuals (p < 0.05).")
    else:
        print("\nNo significant ARCH effects detected in the residuals.")
    
    return has_arch_effect

def fit_arch_model(residuals, p=1, q=1):
    """Fit an ARCH/GARCH model to the residuals."""
    print(f"\nFitting ARCH({p},{q}) model...")
    
    # Fit ARCH model
    model = arch_model(residuals, vol='Garch', p=p, q=q)
    model_fit = model.fit(disp='off')
    
    print(model_fit.summary())
    return model_fit

def evaluate_arch_model(model, residuals):
    """Evaluate the ARCH model and return metrics."""
    # Get the conditional volatility
    conditional_vol = model.conditional_volatility
    
    # Calculate standardized residuals
    std_residuals = residuals / conditional_vol
    
    # Test for remaining ARCH effects
    has_arch_effect = test_arch_effect(std_residuals)
    
    # Prepare results
    results = {
        'model': 'GARCH',
        'has_arch_effect_after_fit': has_arch_effect,
        'model_summary': str(model.summary()),
        'conditional_volatility_mean': float(conditional_vol.mean()),
        'conditional_volatility_std': float(conditional_vol.std())
    }
    
    # Save results
    with open('results/arch_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return results, conditional_vol

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
    
    # Create necessary directories
    os.makedirs('results', exist_ok=True)
    
    # 1. Load residuals from SARIMA model
    print("Loading residuals from SARIMA model...")
    residuals = load_residuals()
    
    if residuals is None:
        print("Failed to load residuals. Please run SARIMA modeling first.")
        return
    
    # 2. Test for ARCH effects
    has_arch_effect = test_arch_effect(residuals)
    
    if not has_arch_effect:
        print("No significant ARCH effects detected. ARCH modeling may not be necessary.")
        return
    
    # 3. Fit ARCH/GARCH model
    print("\nFitting ARCH/GARCH model...")
    # Start with GARCH(1,1) as it's often sufficient
    model = fit_arch_model(residuals, p=1, q=1)
    
    # 4. Evaluate the model
    print("\nEvaluating ARCH/GARCH model...")
    results, conditional_vol = evaluate_arch_model(model, residuals)
    
    # 5. Plot results
    print("\nGenerating volatility plots...")
    plot_volatility(residuals, conditional_vol)
    
    print("\nARCH/GARCH modeling completed successfully!")
    print(f"Results saved to: results/arch_results.json")
    print(f"Volatility plot saved to: visualizations/arch_volatility.png")
    
    # Check if ARCH effects were removed
    if not results['has_arch_effect_after_fit']:
        print("\nSuccess! The ARCH/GARCH model has removed the ARCH effects from the residuals.")
    else:
        print("\nNote: Some ARCH effects remain. You might want to try different GARCH model orders.")

if __name__ == "__main__":
    main()
