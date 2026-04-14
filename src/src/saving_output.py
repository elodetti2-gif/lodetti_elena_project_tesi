import os
from datetime import datetime
import matplotlib
matplotlib.use("TkAgg")  # forza backend interattivo per finestre in PyCharm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def create_output_folders():
    """
    Create folders to save outputs for the current run.
    It creates a main folder with the current timestamp and subfolders
    for graphs, tables, datasets and text files.

    Returns:
        dict: paths of the created folders
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = f"output/{run_id}"

    folders = {
        "graph": os.path.join(base_output, "graph"),
        "table": os.path.join(base_output, "table"),
        "dataset": os.path.join(base_output, "dataset"),
        "text": os.path.join(base_output, "text")
    }

    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    return folders

def save_plot(fig_name, folders):
    """
    Save plot output into a png file in the graph folder.
    """
    path = f"{folders['graph']}/{fig_name}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def save_table(df, name, folders):
    """
    Save table output into a xlsx file in the table folder.
    """
    path = f"{folders['table']}/{name}.xlsx"
    df.to_excel(path, index=False)

def save_dataset(df, name, folders):
    """
    Save dataset output into a csv file in the dataset folder.
    """
    path = f"{folders['dataset']}/{name}.csv"
    df.to_csv(path, index=False)

def save_text(text, filename, folders):
    """
    Save text output into a txt file in the text folder.
    """
    path = f"{folders['text']}/{filename}.txt"
    with open(path, "a") as f:
        f.write(text + "\n")

def model_values_to_df(model_values, folders):
    """
    Convert cross-validation results into a structured dataframe.

    Each row contains actual values, predicted values, residuals,
    and the corresponding year-week for each fold.

    The resulting dataframe is saved for further analysis and comparison.
    """
    rows = []

    for i in range(len(model_values['mae_scores'])):

        y_val = model_values['y_val_per_fold'][i]
        y_pred = model_values['y_pred_per_fold'][i]
        residuals = model_values['residuals_per_fold'][i]
        year_week = model_values['year_week_per_fold'][i]

        for j in range(len(y_val)):

            rows.append({
                "model": model_values["model_name"],
                "fold": i + 1,
                "year": year_week[j][0],
                "week": year_week[j][1],
                "y_val": y_val[j],
                "y_pred": y_pred[j],
                "residual": residuals[j]
            })

    df = pd.DataFrame(rows)

    name = f"model_values_{model_values['model_name']}"
    save_table(df, name, folders)

    return df
def save_feature_importances(importances, name, folders):
    """
    Save feature importance output into a xlsx file in the table folder.
    """

    df = importances.reset_index()
    df.columns = ["feature", "importance"]

    save_table(df, name, folders)


