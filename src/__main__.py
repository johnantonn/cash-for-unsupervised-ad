import os
import random
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.utils.data import generate_data
from sklearn.metrics import roc_auc_score
from utils.functions import *
from models.bandit import Bandit

def main(): 
    # Load env
    load_dotenv()

    # Import the dataset
    df = import_kddcup99(os.getenv('dataset'))

    # Split dataset to train, validation and test
    y_train, X_train, y_validation, X_validation, y_test, X_test = split_df_kddcup99(df)

    # Define bandit
    od_bandit = Bandit(2)

    # Define algorithm parameters
    timeout = 180 # time budget

    # Begin bandit loop
    print('Bandit started...')
    while True:

        print('\tIteration:', len(od_bandit.policy)+1)

        # Select arm
        arm_index = od_bandit.select_arm()
        print('\t\tSelected arm:', arm_index)

        # Play arm
        od_bandit.play_arm(arm_index, X_train, X_validation, y_validation)

        # Print new state
        print('\t\tpolicy:', od_bandit.policy)
        print('\t\trewards:', od_bandit.rewards)

        # Update best arm
        od_bandit.update_best_arm()

        if od_bandit.total_time > timeout:
            break

    print('Bandit finished in', len(od_bandit.policy), 'iterations.')
    print('\tSelected actions:', od_bandit.policy)
    print('\tPolicy payoff:', od_bandit.get_policy_payoff())

    # Apply the best model to the test set
    print('Applying the best model to the test data...')
    print('\tBest model: ', od_bandit.best_arm.model)
    od_bandit.best_arm.model.fit(X_test)

    # Get the prediction on the test data
    y_test_pred = od_bandit.best_arm.model.predict(X_test)  # outlier labels (0 or 1)
    y_test_scores = od_bandit.best_arm.model.decision_function(X_test)  # outlier scores

    roc_auc = np.round(roc_auc_score(y_test, y_test_scores), decimals=4)
    print("\troc_auc on test set: ", roc_auc)

if __name__=="__main__": 
    main()