# Time Series Modelling Project

A comprehensive time series analysis and forecasting project implementing SARIMAX and ARCH/GARCH models for volatility prediction and time series forecasting.

## Table of Contents
1. [Overview](#overview)
2. [Project Workflow](#project-workflow)
3. [File Structure](#file-structure)
4. [Models Directory](#models-directory)
5. [Visualizations / Outputs](#visualizations--outputs)
6. [Key Concepts / Variables](#key-concepts--variables)
7. [Installation and Setup](#installation-and-setup)
8. [Usage](#usage)
9. [Results / Interpretation](#results--interpretation)
10. [Technical Details](#technical-details)
11. [Dependencies](#dependencies)
12. [Notes / Limitations](#notes--limitations)

---

## Overview
* **Goal**: Develop robust time series forecasting models using SARIMAX for trend/seasonality modeling and ARCH/GARCH for volatility prediction
* **Approach**: Statistical time series analysis with stationarity testing, seasonal decomposition, and advanced volatility modeling
* **Highlights**: 
  - SARIMAX parameter selection using auto_arima
  - ARCH/GARCH volatility modeling with residual analysis
  - Visualization and model evaluation
  - PCA-based dimensionality reduction for exogenous features
* **Article Link**: 

---

## Project Workflow
Typical steps in the workflow:

1. **Data Preprocessing**: Load and clean raw time series data, handle missing values, create datetime features
2. **Time Series Preparation**: Stationarity testing, seasonal decomposition, ACF/PACF analysis, exogenous feature engineering
3. **SARIMAX Modeling**: Automated parameter selection, model fitting, and forecasting with exogenous variables
4. **ARCH/GARCH Modeling**: Residual analysis, ARCH effect testing, volatility modeling
5. **Evaluation & Visualization**: Model performance metrics, comparative analysis, comprehensive plotting

---

## File Structure

### Core Scripts

#### `01_data_preprocessing.py`
* **Purpose**: Load and preprocess raw time series data
* **Input**: Raw data files (CSV/JSON format)
* **Output**: Cleaned and preprocessed time series data
* **Key Features**: Data loading, missing value handling, basic preprocessing

#### `02_time_series_prep.py`
* **Purpose**: Prepare time series for modeling with stationarity testing and feature engineering
* **Input**: Preprocessed data from step 1
* **Output**: Stationary series, exogenous features, diagnostic plots
* **Key Features**: 
  - Stationarity testing (ADF test)
  - Seasonal decomposition
  - ACF/PACF plotting
  - Exogenous feature creation with PCA dimensionality reduction

#### `03_SARIMAX_modelling.py`
* **Purpose**: Fit SARIMAX models with automated parameter selection
* **Input**: Stationary time series and exogenous features
* **Output**: Trained SARIMAX model, forecasts, model metadata
* **Key Features**: 
  - Auto_arima for parameter selection
  - Model evaluation with MAPE
  - Forecast generation with confidence intervals

#### `04_ARCH_modelling.py`
* **Purpose**: Model volatility using ARCH/GARCH on SARIMAX residuals and create confidence interval visualizations
* **Input**: SARIMAX model and residuals
* **Output**: ARCH/GARCH model, volatility forecasts, comparison plots with confidence intervals
* **Key Features**: 
  - ARCH effect testing (Ljung-Box)
  - Iterative ARCH/GARCH fitting
  - Combined SARIMAX+volatility forecasting with dynamic confidence intervals
  - 7-day confidence interval comparison plots (SARIMAX vs SARIMAX+GARCH)
  - Volatility-adjusted uncertainty estimation for risk-aware forecasting

---

## Models Directory
The `models/` folder contains:

* **Trained Models**: Serialized SARIMAX and ARCH model files
* **Model Metadata**: JSON files with model parameters, performance metrics, and training information
* **PCA Information**: Principal component analysis results for exogenous features

---

## Visualizations / Outputs
The `visualizations/` directory includes:

* **Time Series Plots**: Raw and preprocessed data visualization
* **ACF/PACF Plots**: Autocorrelation and partial autocorrelation analysis
* **Decomposition Plots**: Trend, seasonal, and residual components
* **ARCH Volatility Plots**: Conditional volatility and residual analysis
* **Model Comparison**: SARIMAX vs SARIMAX+GARCH performance comparisons
* **Confidence Interval Plots**: 
  - 7-day confidence interval comparison (SARIMAX vs SARIMAX+GARCH)
  - Dynamic confidence intervals showing volatility-adjusted uncertainty
  - Combined visualization of both models with confidence bands

---

## Key Concepts / Variables
* **Target Variable**: Time series values to be forecasted
* **Exogenous Features**: External variables including time-based features (hour, day, month) and PCA-reduced components
* **Residuals**: SARIMAX model errors used for ARCH/GARCH modeling
* **Conditional Volatility**: Time-varying variance modeled by ARCH/GARCH
* **Dynamic Confidence Intervals**: Volatility-adjusted uncertainty bounds around forecasts
* **Risk-Aware Forecasting**: Forecasting approach that quantifies uncertainty through confidence intervals

---

## Installation and Setup
1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd time-series-modelling
   ```

2. **Create and activate virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate 
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare data**:
   * Place raw time series data files in the project directory
   * Ensure data has datetime index and target variable column

---

## Usage

### Run Complete Pipeline
```bash
python 01_data_preprocessing.py
python 02_time_series_prep.py
python 03_SARIMAX_modelling.py
python 04_ARCH_modelling.py
```

---

## Results / Interpretation
* **Model Performance**: MAPE (Mean Absolute Percentage Error) for forecast accuracy
* **Key Findings**: 
  - SARIMAX captures trend and seasonality patterns
  - ARCH/GARCH models provide volatility estimates for uncertainty quantification
  - Dynamic confidence intervals adjust based on predicted volatility
  - Risk-aware forecasting improves decision-making under uncertainty
  - 7-day confidence interval comparisons highlight uncertainty differences between models
* **Confidence Interval Benefits**:
  - Identifies periods of higher/lower forecast uncertainty
  - Enables risk-aware planning and decision-making
  - Visual representation of forecast reliability
* **Limitations**: Model assumes stationarity after differencing, may not capture structural breaks

---

## Technical Details
* **Algorithms / Models**: 
  - SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables)
  - ARCH/GARCH (Autoregressive Conditional Heteroskedasticity / Generalized ARCH)
  - PCA (Principal Component Analysis) for dimensionality reduction
* **Frameworks / Tools**: 
  - statsmodels for time series modeling
  - pmdarima for automated ARIMA selection
  - arch library for volatility modeling
  - scikit-learn for PCA and metrics
* **Implementation Notes**: 
  - Automated parameter selection to reduce manual tuning
  - Iterative ARCH fitting for robust convergence
  - Comprehensive logging and error handling
  - Volatility scaling for confidence interval adjustment
  - Minimum scale factor ensures visibility of confidence bands
  - 7-day rolling window for confidence interval visualization

---

## Dependencies
Key dependencies (see `requirements.txt` for versions):

* `pandas >= 1.5.0` - Data manipulation and time series handling
* `numpy >= 1.21.0` - Numerical computations
* `matplotlib >= 3.5.0` - Plotting and visualization
* `statsmodels >= 0.13.0` - Statistical time series models
* `scikit-learn >= 1.1.0` - Machine learning utilities and PCA
* `pmdarima >= 2.0.0` - Automated ARIMA parameter selection
* `arch >= 5.0.0` - ARCH/GARCH volatility modeling
* `scipy >= 1.9.0` - Statistical functions and tests
* `joblib >= 1.1.0` - Model serialization

---

## Notes / Limitations
* **Data Requirements**: Requires regularly spaced time series with datetime index
* **Computational Complexity**: ARCH/GARCH fitting can be computationally intensive for large datasets
* **Model Assumptions**: Assumes linear relationships and may not capture non-linear patterns
* **Generalizability**: Models trained on specific time series may not transfer well to different domains

