import os
import random
import time
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyod.models.lof import LOF
from pyod.models.copod import COPOD
from pyod.models.knn import KNN
from utils.functions import *

def main():
    # Load env
    load_dotenv()

    # Import the dataset
    df = import_kddcup99(os.getenv('dataset'))

    # Split dataset to train, validation and test
    X  = df.iloc[:, :-1]
    y = df['outlier']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    # print size
    print_flag=False
    if print_flag:
        print(X.shape)
        print(y.shape)
        print(X_train.shape)
        print(y_train.shape)
        print(X_test.shape)
        print(y_test.shape)

    # Set the parameters by cross-validation
    n_neighbors = [1, 2, 3, 4, 5, 10]
    method = ['largest', 'mean', 'median']
    leaf_size = [1, 2, 5]

    # Define estimator
    clf = KNN()
    total_time=time.time()
    for n in n_neighbors:
        clf.n_neighbors=n
        for m in method:
            clf.method=m
            for l in leaf_size:
                clf.leaf_size=l
                #clf = KNN(n_neighbors=n, method=m, leaf_size=l)
                start_time = time.time()
                clf.fit(X_train)
                end_time = time.time()
                # get prediction on validation set
                y_true, y_pred = y_test, clf.predict(X_test)
                y_test_scores = clf.decision_function(X_test)  # outlier scores
                print("n =", n, "m =", m, "l =", l, " auc:", roc_auc_score(y_test, y_test_scores), "time =",end_time-start_time)

    total_time = time.time() - total_time
    print("Total time:", total_time)

if __name__=="__main__": 
    main()