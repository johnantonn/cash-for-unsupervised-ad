import os
import time
import random
import itertools
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from utils.functions import import_dataset

def _product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    # adapt this
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def _print_shape(*args):
    for arg in args:
        print(arg.shape)

def find_best_model(dataset):
    # Import dataset
    df = import_dataset(dataset)

    # Split dataset to train and test
    X  = df.iloc[:, :-1]
    y = df['outlier']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # print size
    #_print_shape(X, y, X_train, X_test, y_test)

    # Models dict
    models = {
        "knn": KNN(),
        "lof": LOF()
    }

    # Hyperparameters dict
    hyperparameters = {
        "knn": {
            "n_neighbors": [1, 5, 10, 40],
            "leaf_size": [1, 5, 10, 40],
            "contamination": [0.05, 0.1, 0.2]
        },
        "lof": {
            "n_neighbors": [1, 5, 10, 40],
            "leaf_size": [1, 5, 10, 40],
            "contamination": [0.05, 0.1, 0.2]
        }
    }

    best_model=''
    best_model_score=0
    best_model_params={}

    # model loop
    for key in models:
        print("\t", key)
        best_score, best_params = evaluate_model(models[key], hyperparameters[key], X_train, X_test, y_test)
        if(best_score > best_model_score):
            best_model = key
            best_model_score = best_score
            best_model_params = best_params

    print("Best model:", best_model)
    print("Best model score:", best_model_score)
    print("Best model params:", best_model_params)
    print("\n")

def evaluate_model(clf, hyperparams, X_train, X_test, y_test):
    # Timeout for searching a single model
    timeout = float(os.getenv('timeout'))

    # List of hyperparam combinations
    hyperparam_sets = list(_product_dict(**hyperparams))
    random.shuffle(hyperparam_sets) #shuffle

    total_time = 0 # total time
    total_searches = len(hyperparam_sets) # total searches
    search_count = 0 # searches count
    best_hp_set = {}
    best_score = 0

    # hyperparam search loop
    for hp_set in hyperparam_sets:

        # define model
        clf.set_params(**hp_set)

        # fit model and time it
        start_time = time.time()
        clf.fit(X_train)
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 4)

        # get prediction on validation set
        y_true, y_pred = y_test, clf.predict(X_test)

        # calculate scores
        y_test_scores = clf.decision_function(X_test)  # outlier scores
        auc_score = round(roc_auc_score(y_test, y_test_scores), 4)
        #print('\t\t', hp_set, " auc:", auc_score, "time =", elapsed_time)

        # Updates
        if(auc_score > best_score):
            best_score = auc_score
            best_hp_set = hp_set

        search_count += 1 # inc search counter
        total_time += elapsed_time # update time
        total_time = round(total_time, 4)

        if(total_time >= timeout):
            break

    # total hyperparam space search rate
    search_percentage = search_count / total_searches
    search_percentage = round(search_percentage, 4)

    # prints
    print("\t\tTotal elapsed time:", total_time)
    print("\t\tHyperparameter space search rate:", search_percentage)
    print("\t\tBest hyperparam set: ", best_hp_set)
    print("\t\tBest score achieved:", best_score)

    return best_score, best_hp_set

def main():
    # Load env
    load_dotenv()

    # Define datasets
    datasets = [
        'PenDigits_withoutdupl_norm_v10.arff',
        'Waveform_withoutdupl_norm_v10.arff',
        'Annthyroid_withoutdupl_norm_05_v10.arff',
        'Cardiotocography_withoutdupl_norm_20_v10.arff'
    ]

    # Dataset loop
    for dataset in datasets:
        print(dataset) # print datset name
        find_best_model(dataset) # find best model for the dataset

if __name__=="__main__": 
    main()