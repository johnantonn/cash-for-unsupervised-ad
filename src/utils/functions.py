""" This module contains auxiliary functions """

import sys
import os
import time
import itertools
import decimal
from dotenv import load_dotenv
from scipy.io import arff
import pandas as pd
import numpy as np
from pathlib import Path
from os.path import dirname
from sklearn.metrics import roc_auc_score, f1_score

def round_num(num):
    return round(num, 4)

def float_range(start, stop, step):
    while start < stop:
        yield float(start)
        start += decimal.Decimal(step)

def subsample(df, n):
    if(len(df) > 5000):
        df = df.sample(n=5000)
    return df

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def import_dataset(filename):
    """ Function that reads the KDDCup99 dataset and returns a dataframe.

    Args:
        filename (str): The name of the file

    Returns:
        (df): The dataframe with the data contents
    """

    # Build path to data directory
    filepath = Path(dirname(dirname(dirname(__file__)))+"/data/"+filename)

    # If file does not exist
    if not os.path.exists(filepath):
        raise FileNotFoundError("filepath %s does not exist" % filepath)

    # Load file to a df
    data = arff.loadarff(filepath)
    df = pd.DataFrame(data[0])
    df.outlier = df.outlier.str.decode("utf-8")
    df['outlier'] = df['outlier'].map({'yes':1,'no':0}) 
    if 'id' in df:
        del df['id']

    return df

def evaluate_model_instance(clf, hp, X_train, X_val, y_val):
    ''' Function that applies a model instance and returns scores.

    Args:
        clf (model): the anomaly detection model/algorithm
        hp (dict): the dict of hyperparameter instance values
        X_train: training set
        X_val: validation set
        y_val: validation set labels

    Returns:
        scores (dict): The resulting scores
        time (float): Elapsed time for fit method
    '''
    # set model params
    clf.set_params(**hp)

    # fit model and time it
    start_time = time.time()
    clf.fit(X_train)
    end_time = time.time()
    elapsed_time = round_num(end_time - start_time)

    # calculate scores
    y_pred = clf.predict(X_val) # predict()
    f1 = round_num(f1_score(y_val, y_pred)) # f1
    y_val_scores = clf.decision_function(X_val) # decision_function()
    auc = round_num(roc_auc_score(y_val, y_val_scores)) # auc

    scores={"f1": f1, "auc": auc} # dict

    return scores, elapsed_time
