# general
import os
import time
import random
import numpy as np
from dotenv import load_dotenv
# sklearn
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
# custom
from utils.pyod_models import models
from utils.datasets import datasets
from utils.functions import *
# warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def find_best_model(dataset):
    ''' Function that finds the best model for a dataset according to a time budget for fit() function.

    Args:
        dataset (str): The name of the dataset file

    Returns:
        Prints results to console
    '''
    # Import dataset
    df = import_dataset(dataset)

    # Subsample to N=5000
    df = subsample(df, 5000)

    # Split dataset to train, validation and test sets
    X  = df.iloc[:, :-1]
    y = df['outlier']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=0)

    # Timeout for searching a single model
    timeout = float(os.getenv('dataset_timeout'))
    total_time = 0
    search_counter = 0

    # Best model
    best_model_name=''
    best_model_scores={
        'f1': 0.0,
        'auc': 0.0
    }
    best_model_params={}
    
    while(total_time < timeout):
        # Sample a model
        mdl = random.sample(list(models), 1)[0]

        # List of hyperparam combinations
        hyperparam_sets = list(product_dict(**models[mdl]['hyperparameters']))

        # Sample a hyperparam set instance
        hp_inst = random.sample(hyperparam_sets, 1)[0]

        # Evaluate model instance
        scores, fit_time = evaluate_model_instance(
            models[mdl]['instance'],
            hp_inst,
            X_train,
            X_val,
            y_val,
        )

        # Increase search counter
        search_counter += 1
        print(search_counter, mdl, hp_inst, scores, fit_time)

        # Compare scores
        if(scores['f1'] > best_model_scores['f1']):
            best_model_name = mdl
            best_model_scores = scores
            best_model_params = hp_inst
    
        # Increase timer
        total_time += fit_time
        total_time = round_num(total_time)

    print("Elapsed time:", total_time)
    print('Best model:', best_model_name)
    print('Best model scores:', best_model_scores)
    print('Best model params:', best_model_params)
    print('\n')

def main():
    # Load env
    load_dotenv()

    # Dataset loop
    for dataset in datasets:
        print(dataset) # print datset name
        find_best_model(dataset) # find best model for the dataset

if __name__=='__main__': 
    main()