# general
import os
import time
import random
import numpy as np
from dotenv import load_dotenv
# sklearn
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
    ''' Function that finds the best model for a dataset.

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
    X  = df.iloc[:, :-1] # drop labels from X
    y = df['outlier'] # keep labels for y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=0)

    # Best model
    best_model_name=''
    best_model_scores={
        'f1': 0.0,
        'auc': 0.0
    }
    best_model_params={}

    # Loop over models
    for key in models:
        print('\t', key)
        best_scores, best_params = evaluate_model_family(
            models[key]['instance'],
            models[key]['hyperparameters'],
            X_train,
            X_val,
            y_val,
            X_test,
            y_test
        )
        # Compare models
        if(best_scores['f1'] > best_model_scores['f1']):
            best_model_name = key
            best_model_scores = best_scores
            best_model_params = best_params
    
    print('Best model:', best_model_name)
    print('Best model scores:', best_model_scores)
    print('Best model params:', best_model_params)
    print('\n')

def evaluate_model_family(clf, hyperparams, X_train, X_val, y_val, X_test, y_test):
    ''' Function that randomly searches the hyperparameter space
    and finds the optimal hyperparameter set for a given model.

    Args:
        clf (model): the anomaly detection model/algorithm
        hyperparams (dict): the dict of all hyperparameter combinations
        X_train: training set
        X_val: validation set
        y_val: validation set labels
        X_test: test set
        y_test: test set labels

    Returns:
        best_score (number): The best score achieved during random search
        best_hp_set (dict): The set of hyperparameter values that achieved the best_score
    '''
    # Timeout for searching a single model
    timeout = float(os.getenv('model_timeout'))

    # List of hyperparam combinations
    hyperparam_sets = list(product_dict(**hyperparams))
    random.shuffle(hyperparam_sets) #shuffle

    total_time = 0 # total time
    total_searches = len(hyperparam_sets) # total searches
    search_count = 0 # searches count
    best_hp_set = {}
    best_scores = {
        'f1': 0.0,
        'auc': 0.0
    }
    best_score_auc = 0

    # Loop over hyperparam sets
    for hp_set in hyperparam_sets:
        #print(hp_set)
        search_count += 1 # inc search counter
        
        # Apply model instance and get scores
        scores, elapsed_time = evaluate_model_instance(clf, hp_set, X_train, X_val, y_val)

        # Determine best score and hyperparams
        if(scores['f1'] > best_scores['f1']):
            best_scores['f1'] = scores['f1']
            best_scores['auc'] = scores['auc']
            best_hp_set = hp_set

        total_time += elapsed_time # update time
        total_time = round_num(total_time)

        if(total_time >= timeout):
            break

    # total hyperparam space search rate
    search_percentage = search_count / total_searches
    search_percentage = round_num(search_percentage)

    # prints
    print('\t\tTotal elapsed time:', total_time)
    print('\t\tHyperparameter space search rate:', search_count, '/', total_searches, ' (', search_percentage, ')')
    print('\t\tBest hyperparam set: ', best_hp_set)
    print('\t\tBest scores achieved:', best_scores)

    return best_scores, best_hp_set

def main():
    # Load env
    load_dotenv()

    # Dataset loop
    for dataset in datasets:
        print(dataset) # print datset name
        find_best_model(dataset) # find best model for the dataset

if __name__=='__main__': 
    main()