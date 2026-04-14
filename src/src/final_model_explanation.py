import pandas as pd
import numpy as np
import shap
import src.saving_output as so
import matplotlib
matplotlib.use("Agg")  # forza backend interattivo per finestre in PyCharm
import matplotlib.pyplot as plt

def importances_calculation(rf, X, folders):
    """
    Compute feature importance from a trained Random Forest model
    and save the results for analysis.
    """
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(importances)
    so.save_feature_importances(importances, "rf_importances", folders)
    return importances
def shap_analysis(X_train_final, final_model):
    """
    Perform SHAP analysis on a sample of the training set to explain
    model predictions and compute feature contributions.
    """
    # Copy dataset to avoid modifying original data
    X_shap = X_train_final.copy()

    # Limit number of samples for SHAP to reduce computation time
    n_sample = 2000
    n_sample = min(n_sample, len(X_shap))

    # Randomly sample data for SHAP computation
    X_sample = X_shap.sample(n=n_sample, random_state=42)

    # Create SHAP explainer for the trained model
    explainer = shap.Explainer(final_model)

    # Compute SHAP values for sampled data
    exp=explainer(X_sample)

    # Compute mean absolute SHAP values for global feature importance
    mean_abs_shap = np.abs(exp.values).mean(axis=0)
    return X_shap, X_sample, explainer, exp, mean_abs_shap, n_sample
    
def shap_importance_display(X_sample,mean_abs_shap, n_sample, folders):
    """
    Create a table with SHAP feature importance values and save it
    for further analysis.
    """
    shap_importance = (
        pd.DataFrame({
            "feature": X_sample.columns,
            "mean_abs_shap": mean_abs_shap
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True))
    print(f"SHAP computed on sample of {n_sample} rows.")
    print(shap_importance.head(21))
    so.save_table(shap_importance, "shap_importance", folders)
    return shap_importance

def shap_summary_plot(exp, X_sample, folders):
    """
    Generate SHAP summary plots to visualize global feature importance
    and feature impact distribution.
    """
    # Plot global feature importance (bar chart)
    shap.summary_plot(exp.values, X_sample, plot_type="bar", show=False)
    so.save_plot("shap_summary_plot_bar", folders)
    plt.close()
    # Plot full SHAP distribution (beeswarm plot)
    shap.summary_plot(exp.values, X_sample, show=False)
    so.save_plot("shap_summary_plot", folders)
    plt.close()
def shap_summary_plot_waterfall(exp, folders, i=0):
    """
    Explain a single prediction using SHAP waterfall plot.
    """
    shap.plots.waterfall(exp[i], max_display=15, show=False)
    so.save_plot("shap_summary_plot_waterfall", folders)