""" This module contains auxiliary functions """

import sys
import os
import itertools
import decimal
from dotenv import load_dotenv
from scipy.io import arff
import pandas as pd
import numpy as np
from pathlib import Path
from os.path import dirname

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
