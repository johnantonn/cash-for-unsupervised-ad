""" This module contains auxiliary functions for importing the dataset(s) """

import sys
import os
from dotenv import load_dotenv
from scipy.io import arff
import pandas as pd
import numpy as np
from pathlib import Path
from os.path import dirname

def import_kddcup99(filename):
    """ Function that reads the KDDCup99 dataset and returns a dataframe.

    Args:
        filename (str): The name of the file

    Returns:
        (df): The dataframe with the data contents
    """

    # Build path to input directory
    filepath = Path(dirname(dirname(dirname(__file__)))+"/input/"+filename)

    # If file does not exist
    if not os.path.exists(filepath):
        raise FileNotFoundError("filepath %s does not exist" % filepath)

    # Load file to a df
    data = arff.loadarff(filepath)
    df = pd.DataFrame(data[0])
    df.outlier = df.outlier.str.decode("utf-8")
    df['outlier'] = df['outlier'].map({'yes':1,'no':0}) 

    return df

def split_df_kddcup99(df):
    """ Function that splits a dataframe and returns training, validation and test sets.

    Args:
        df (dataframe): The dataframe to split

    Returns:
        (X_train): The training set
        (y_train): The training set lables
        (X_validation): The validation set
        (y_validation): The validation set lables
        (X_test): The test set
        (y_test): The test set lables
    """
    
    # Load env
    load_dotenv()

     # Calculate sizes
    p_train=float(os.getenv('p_train'))
    p_validation=float(os.getenv('p_validation'))
    p_test=float(os.getenv('p_test'))
    n_train = round(p_train * len(df))
    n_validation = round(p_validation * len(df))
    n_test = round(p_test * len(df))

    # Split original df
    df_train = df.iloc[:n_train, :]
    df_validation = df.iloc[n_train:n_train + n_validation, :]
    df_test = df.iloc[n_train + n_validation:, :]

    # X and y np arrays
    y_train = df_train['outlier'].to_numpy()
    X_train = df_train.drop(columns=['id', 'outlier']).to_numpy()
    y_validation = df_validation['outlier'].to_numpy()
    X_validation = df_validation.drop(columns=['id', 'outlier']).to_numpy()
    y_test = df_test['outlier'].to_numpy()
    X_test = df_test.drop(columns=['id', 'outlier']).to_numpy()
    
    return y_train, X_train, y_validation, X_validation, y_test, X_test
