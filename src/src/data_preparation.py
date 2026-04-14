import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import src.saving_output as so

def loading_datasets():
    """
    Load all datasets from csv files.

    Returns:
        features, stores, test, train: loaded dataframes
    """
    features=pd.read_csv("data/features.csv")
    stores=pd.read_csv("data/stores.csv")
    test=pd.read_csv("data/test.csv")
    train=pd.read_csv("data/train.csv")
    return features, stores, test, train
    
def general_info(df, Store, folders):
    """
    Print and save basic information about a dataset.

    It shows shape, columns, data types, missing values,
    and basic statistics about the Store column.
    """
    df.info()
    print(f'\n')
    print(f'database shape: {df.shape}')
    print(f'\n')
    print(df.head())
    print(f'\ncolumns of the dataframe : \n{df.columns}\n')
    print(f'number of {Store} in the dataframe : \n{df[Store].unique()}\n')
    print(f'number of rows per each {Store} in the dataframe : \n{df[Store].value_counts()}\n')

    text = ""
    text += f"Shape: {df.shape}\n\n"
    text += f"Dtypes:\n{df.dtypes}\n\n"
    text += f"Columns of the dataframe:\n{list(df.columns)}\n\n"
    text += "Missing values per column:\n"
    text += df.isna().sum().to_string()
    text += f"\n\nNumber of {Store} in the dataframe:\n"
    text += str(df[Store].unique()) + "\n\n"
    text += f"Number of rows per each {Store} in the dataframe:\n"
    text += df[Store].value_counts().to_string() + "\n\n"
    text += f"Database head: \n"
    text += df.head().to_string()

    so.save_text(text, 'general_info', folders)

  
def best_store_by_missing(df, Store, folders):
    """
    Compute missing values per store and find the store with the least missing data.
    """
    df['missing_in_row'] =df.isna().sum(axis=1)
    missing_per_store = df.groupby(Store)['missing_in_row'].sum()
    missing_per_store
    #What is the store with less missing values? --> store number 20
    min_missing = missing_per_store.min()
    best_store=missing_per_store[missing_per_store == min_missing]
    print(f'missing values per {Store} : \n{missing_per_store}\n')
    print(f'{Store} with less missing values : \n{best_store}\n')

    text = ""
    text += f"Missing values per {Store}:\n"
    text += missing_per_store.to_string() + "\n\n"
    text += f"{Store} with less missing values:\n"
    text += best_store.to_string()
    so.save_text(text, "final_dataset_creation", folders)

    return missing_per_store, best_store

def new_database_20(df, Store, number):
    """
    Filter the dataset for a specific store.
    """
    df_store_20 = df[df[Store] == number]
    return df_store_20
  
def check_df_store_20(df_store_20, Store):
    """
    Check basic consistency of a filtered dataset.
    It prints shape, unique values of the store column and dataframe info.
    """
    print(f'shape: {df_store_20.shape}')
    print(f'different stores in the dataframe: {df_store_20[Store].unique()}')
    print(f'{df_store_20.info()}')
  
def best_dept_by_missing(df_store_20, Dept, folders):
    """
    Compute missing values per department and find the department with the least missing data.
    """
    df_store_20['missing_in_row'] =df_store_20.isna().sum(axis=1)
    missing_per_dept = df_store_20.groupby(Dept)['missing_in_row'].sum()
    missing_per_dept
    min_missing = missing_per_dept.min()
    best_dept=missing_per_dept[missing_per_dept == min_missing]
    print(f'missing values per {Dept} : \n{missing_per_dept}\n')
    print(f'{Dept} with less missing values : \n{best_dept}\n')

    text = ""
    text += f"Missing values per {Dept}:\n"
    text += missing_per_dept.to_string() + "\n\n"
    text += f"{Dept} with less missing values:\n"
    text += best_dept.to_string()
    so.save_text(text, "final_dataset_creation", folders)

    return missing_per_dept, best_dept

def best_dept_by_rows(df_store_20, Dept, folders):
    """
    Find the department with the highest number of rows in the dataset.
    """
    dept_counts = df_store_20[Dept].value_counts()
    print(f'number of rows per each {Dept} in the dataframe : \n{dept_counts}\n')
    best_dept = dept_counts.idxmax()
    print(f'the {Dept} with the highest number of lines is {best_dept}')

    text = ""
    text += f"Number of rows per each {Dept} in the dataframe:\n"
    text += dept_counts.to_string() + "\n\n"
    text += f"{Dept} with the highest number of lines:\n"
    text += str(best_dept)
    so.save_text(text, "final_dataset_creation", folders)

    return best_dept
  
def new_database_20_1(df_store_20, Store, Dept, number):
    """
    Filter the dataset for a specific department and check basic consistency.
    """
    df_store_20_1 = df_store_20[df_store_20[Dept] == number]
    print('I check that everything is alright: ')
    print(f'new database shape: {df_store_20_1.shape}')
    print(f'stores in the new database: {df_store_20_1[Store].unique()}')
    print(f'departments in the new database: {df_store_20_1[Dept].unique()}')
    return df_store_20_1
  
def dates_type_check(df):
    """
    Ensure the Date column is in datetime format.
    """
    if df['Date'].dtype != np.dtype('datetime64[ns]'):
        df['Date'] = pd.to_datetime(df['Date'])
    return df
    
def merging_datasets(df1, df2):
    """
    Merge two datasets on Store and Date.
    """
    df=df1.merge(df2, on=['Store', 'Date'], how='left')
    return df
    
