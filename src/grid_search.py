# general
import os
import time
import random
import numpy as np
from dotenv import load_dotenv
# sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# pyod models
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
from pyod.models.abod import ABOD
from pyod.models.sos import SOS
# custom
from utils.functions import *
# warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def find_best_model(dataset):
    """ Function that finds the best model for a dataset.

    Args:
        dataset (str): The name of the dataset file

    Returns:
        Just prints to console
    """
    # Import dataset
    df = import_dataset(dataset)

    # Split dataset to train and test
    X  = df.iloc[:, :-1]
    y = df['outlier']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    # print size
    #_print_shape(X, y, X_train, X_test, y_test)
    # print("\tNumber of points in X_train:", len(X_train))

    # Models dict
    models = {
        "knn": {
            "instance": KNN(),
            "hyperparameters": {
                "contamination": np.arange(0.05, 0.45, 0.05),
                "n_neighbors": np.arange(1, 30, 1),
                "method": ['largest', 'mean', 'median'],
                "leaf_size": np.arange(1, 50, 1)
            }
        },
        "lof": {
            "instance": LOF(),
            "hyperparameters": {
                "contamination": np.arange(0.05, 0.45, 0.05),
                "n_neighbors": np.arange(1, 30, 1),
                "leaf_size": np.arange(1, 50, 1)
            }
        },
        "cblof": {
            "instance": CBLOF(),
            "hyperparameters": {
                "contamination": np.arange(0.05, 0.45, 0.05),
                "n_clusters": np.arange(2, 15, 1),
                "alpha": np.arange(0.05, 0.45, 0.05),
                "beta": np.arange(2, 20, 1)
            }
        },
        "sos": {
            "instance": SOS(),
            "hyperparameters": {
                "contamination": np.arange(0.05, 0.45, 0.05),
                "perplexity": np.arange(1, 100, 1)
            }
        }
    }

    # Define best model
    best_model=''
    best_model_score=0
    best_model_params={}

    # Loop over models
    for key in models:
        print("\t", key)
        best_score, best_params = evaluate_model(
            models[key]["instance"], 
            models[key]["hyperparameters"], 
            X_train, 
            X_test, 
            y_test
        )
        if(best_score > best_model_score):
            best_model = key
            best_model_score = best_score
            best_model_params = best_params

    print("Best model:", best_model)
    print("Best model score:", best_model_score)
    print("Best model params:", best_model_params)
    print("\n")

def evaluate_model(clf, hyperparams, X_train, X_test, y_test):
    """ Function that randomly searches the hyperparameter space
    and finds the optimal hyperparameter set for a given model.

    Args:
        clf (model): the anomaly detection model/algorithm
        hyperparams (dict): the dict of all hyperparameter combinations
        X_train: training set
        X_test: test set
        y_test: test set labels

    Returns:
        best_score (number): The best score achieved during random search
        best_hp_set (dict): The set of hyperparameter values that achieved the best_score
    """
    # Timeout for searching a single model
    timeout = float(os.getenv('timeout'))

    # List of hyperparam combinations
    hyperparam_sets = list(product_dict(**hyperparams))
    random.shuffle(hyperparam_sets) #shuffle

    total_time = 0 # total time
    total_searches = len(hyperparam_sets) # total searches
    search_count = 0 # searches count
    best_hp_set = {}
    best_score = 0

    # Loop over hyperparam sets
    for hp_set in hyperparam_sets:
        #print(hp_set)
        # define model
        clf.set_params(**hp_set)

        # fit model and time it
        #print("Fitting")
        start_time = time.time()
        clf.fit(X_train)
        end_time = time.time()
        elapsed_time = round_num(end_time - start_time)

        # get prediction on validation set
        #print("Predicting")
        y_true, y_pred = y_test, clf.predict(X_test)

        # calculate scores
        #print("Scoring")
        y_test_scores = clf.decision_function(X_test)  # outlier scores
        auc_score = round_num(roc_auc_score(y_test, y_test_scores))
        #print('\t\t', hp_set, " auc:", auc_score, "time =", elapsed_time)

        # Determine best score and hyperparams
        if(auc_score > best_score):
            best_score = auc_score
            best_hp_set = hp_set

        search_count += 1 # inc search counter
        total_time += elapsed_time # update time
        total_time = round_num(total_time)

        if(total_time >= timeout):
            break

    # total hyperparam space search rate
    search_percentage = search_count / total_searches
    search_percentage = round_num(search_percentage)

    # prints
    print("\t\tTotal elapsed time:", total_time)
    print("\t\tHyperparameter space search rate:", search_count, "/", total_searches, " (", search_percentage, ")")
    print("\t\tBest hyperparam set: ", best_hp_set)
    print("\t\tBest score achieved:", best_score)

    return best_score, best_hp_set

def main():
    # Load env
    load_dotenv()

    # Define datasets
    datasets = [
        'Waveform_withoutdupl_norm_v10.arff',
        'Cardiotocography_withoutdupl_norm_02_v10.arff'
    ]

    # Dataset loop
    for dataset in datasets:
        print(dataset) # print datset name
        find_best_model(dataset) # find best model for the dataset

if __name__=="__main__": 
    main()