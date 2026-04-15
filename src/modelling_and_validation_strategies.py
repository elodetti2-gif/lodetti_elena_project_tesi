import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import src.saving_output as so


def features_target_separation(df):
    """
    Split dataset into features (X) and target (y).
    """
    X = df.drop(columns=['Weekly_Sales','Date','IsOutlier'])
    y = df['Weekly_Sales']
    return X, y
    
def chronological_hold_out_validation(X, y):
    """
    Split dataset into training and validation sets using chronological order.
    """
    split_50 = len(X) // 2
    X_train_50, X_val_50 = X.iloc[:split_50], X.iloc[split_50:]
    y_train_50, y_val_50 = y.iloc[:split_50], y.iloc[split_50:]
    
    assert (
    X_train_50[['Year','Month','Week']].values[-1]
    <= X_val_50[['Year','Month','Week']].values[0]
).all()
    return X_train_50, X_val_50, y_train_50, y_val_50
  
def timeSeriesSplit_cross_validation():
    """
    Create a TimeSeriesSplit object for cross-validation.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    return tscv
    

#LINEAR REGRESSION ON 50/50 HOLD-OUT
def lr_chronological_hold_out(X_train_50, y_train_50, X_val_50, y_val_50):
    """
    Train a Linear Regression model using chronological hold-out validation.
    """
    lr = LinearRegression()
    lr.fit(X_train_50, y_train_50)
    y_pred_50 = lr.predict(X_val_50)
    return lr, y_pred_50
    
def mae_rmse_mape_calculation(y_val, y_pred):
    """
    Compute regression evaluation metrics: MAE, RMSE, MAPE.
    """
    lr_hold_out_values={}
    lr_hold_out_values['mae'] = mean_absolute_error(y_val, y_pred)
    lr_hold_out_values['rmse'] = np.sqrt(mean_squared_error(y_val, y_pred))
    lr_hold_out_values['mape'] = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
    
    print("Hold-Out 50/50 Validation:")
    print(f"MAE: {lr_hold_out_values['mae']:.2f}, RMSE: {lr_hold_out_values['rmse']:.2f}, MAPE: {lr_hold_out_values['mape']:.2f}%")
    return lr_hold_out_values
    
#TIME SERIES SPLIT
    
def output_strings_TimeSeriesSplit(name):
    """
    Initialize dictionary to store TimeSeriesSplit results.
    """
    model_values={'model_name' : name, 'mae_scores' : [], 'rmse_scores' : [], 'mape_scores' : [], 'residuals_per_fold' :[], 'year_week_per_fold' : [], 'y_val_per_fold' : [], 'y_pred_per_fold' : []}
    return model_values
    
def model_fit_TimeSeriesSplit(model, tscv, model_values, X, y, folders):
    """
    Train and evaluate a model using TimeSeriesSplit cross-validation.
    """
    file_name=f"info_folds_{model_values['model_name']}"
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):

        X_train_TSS, X_val_TSS = X.iloc[train_idx], X.iloc[val_idx]
        y_train_TSS, y_val_TSS = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train_TSS, y_train_TSS)
        y_pred_TSS = model.predict(X_val_TSS)

        model_values['y_val_per_fold'].append(y_val_TSS.values)
        model_values['y_pred_per_fold'].append(y_pred_TSS)

        # metriche
        mae_TSS = mean_absolute_error(y_val_TSS, y_pred_TSS)
        rmse_TSS = np.sqrt(mean_squared_error(y_val_TSS, y_pred_TSS))
        mape_TSS = np.mean(np.abs((y_val_TSS - y_pred_TSS) / y_val_TSS)) * 100

        model_values['mae_scores'].append(mae_TSS)
        model_values['rmse_scores'].append(rmse_TSS)
        model_values['mape_scores'].append(mape_TSS)

        # residui
        residuals = y_val_TSS - y_pred_TSS
        model_values['residuals_per_fold'].append(residuals.values)

        # Year–Week (SALVATI PER FOLD)
        year_week = list(zip(
            X_val_TSS['Year'].values,
            X_val_TSS['Week'].values
            ))
        model_values['year_week_per_fold'].append(year_week)

        # info fold
        print(f"Fold {fold+1}")
        print("Train period:",
          X_train_TSS[['Year','Month','Week']].head(1).values,
          "→",
          X_train_TSS[['Year','Month','Week']].tail(1).values)
        print("Validation period:",
          X_val_TSS[['Year','Month','Week']].head(1).values,
          "→",
          X_val_TSS[['Year','Month','Week']].tail(1).values)
        print(f"MAE: {mae_TSS:.2f}, RMSE: {rmse_TSS:.2f}, MAPE: {mape_TSS:.2f}")
        print("-" * 50)
        text=""
        text+=f"\n\nFold {fold+1}\n\n"
        text+=f"Train period: {X_train_TSS[['Year','Month','Week']].head(1).values} --> {X_train_TSS[['Year','Month','Week']].tail(1).values}\n"
        text+=f"Validation period: {X_val_TSS[['Year','Month','Week']].head(1).values} --> {X_val_TSS[['Year','Month','Week']].tail(1).values}\n"
        text+=f"MAE: {mae_TSS:.2f}, RMSE: {rmse_TSS:.2f}, MAPE: {mape_TSS:.2f}\n\n"
        text+="-" * 50
        so.save_text(text, file_name, folders)

    return model_values
        
def avg_mae_rmse_mape(model_values):
    """
    Compute average and standard deviation of evaluation metrics across folds.
    """
    model_values['mae_mean'] = np.mean(model_values['mae_scores'])
    model_values['mae_std']  = np.std(model_values['mae_scores'])

    model_values['rmse_mean'] = np.mean(model_values['rmse_scores'])
    model_values['rmse_std']  = np.std(model_values['rmse_scores'])

    model_values['mape_mean'] = np.mean(model_values['mape_scores'])
    model_values['mape_std']  = np.std(model_values['mape_scores'])
    return model_values
    
    
        
#RANDOM FOREST ON TIMESERIESSPLIT
def rf_definition():
    """
    Define Random Forest regression model.
    """
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    return rf
    

    

