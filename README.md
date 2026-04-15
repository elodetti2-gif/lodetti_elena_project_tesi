# Demand Forecasting for Retail Store Using Machine Learning

## Project Overview
This thesis project focuses on weekly demand forecasting for a retail store using machine learning techniques.  
The goal is to support inventory management and operational decision-making by building interpretable and predictive models based on historical sales data.

The analysis is based on the "Walmart Recruiting – Store Sales Forecasting" dataset and considers a single store–department combination to ensure interpretability and methodological clarity.

---

## Methodology
The project follows a complete machine learning pipeline structured into the following stages:

- Data preparation and filtering of raw datasets
- Data cleaning, including missing value treatment and outlier analysis
- Exploratory Data Analysis (EDA) with time series visualization and feature–target relationships
- Feature engineering, including:
  - calendar-based features (week, month, year)
  - lag features
  - rolling statistics
  - holiday indicators
- Model development and validation using time-aware strategies:
  - chronological train/test split
  - TimeSeriesSplit cross-validation
- Model selection and evaluation using:
  - Linear Regression (baseline model)
  - Random Forest Regressor (non-linear model)
  - metrics: MAE, RMSE, MAPE
- Forecast generation and post-analysis
- Distributional analysis of residuals (including Kolmogorov–Smirnov test)
- Model interpretation using SHAP values and feature importance

---

## Project Structure
```text
data/
├── features.csv
├── stores.csv
├── train.csv
├── test.csv

src/
├── data_preparation.py
├── data_cleaning.py
├── exploratory_data_analysis.py
├── feature_engineering.py
├── modelling_and_validation_strategy.py
├── model_selection.py
├── forecasting.py
├── final_model_evaluation.py
├── final_model_explanation.py
├── saving_output.py
├── __init__.py

output/
├── run_id (timestamped folder)
│   ├── dataset/
│   ├── graph/
│   ├── table/
│   ├── text/

main.py
requirements.txt
README.md
```

## Installation
Install the required dependencies using:

pip install -r requirements.txt

Main libraries include:
- pandas
- numpy
- scikit-learn
- matplotlib
- shap

To run the full pipeline:

python main.py

This will execute the complete workflow from data preprocessing to model evaluation and output generation.

## Outputs

Each execution generates a timestamped folder inside output/, containing four folders:

- dataset/: containing the processed dataset and predictions
- graph/: containing EDA plots, forecast visualizations, SHAP plots...
- table/: containing performance metrics, feature importances, results tables...
- text/: containing analytical summaries

### Author
Elena Lodetti
