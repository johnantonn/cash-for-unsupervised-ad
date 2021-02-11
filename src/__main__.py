import os
import random
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.cblof import CBLOF
from pyod.models.copod import COPOD
from pyod.utils.data import generate_data
from sklearn.metrics import roc_auc_score
from utils.functions import *
from models.bandit import Bandit
from models.arm import Arm

def main():
    # Load env
    load_dotenv()

    # Import the dataset
    df = import_kddcup99(os.getenv('dataset'))

    # Split dataset to train, validation and test
    y_train, X_train, y_validation, X_validation, y_test, X_test = split_df_kddcup99(df)

    # Define arms
    arms = np.array([Arm(LOF()), Arm(KNN()), Arm(IForest()), Arm(CBLOF()), Arm(COPOD())])

    # Create bandit
    od_bandit = Bandit(K=len(arms), arms=arms, solver='egreedy')

    # Define time budget
    timeout = float(os.getenv('timeout'))
    timeout_ratio = float(os.getenv('timeout_ratio'))
    arm_timeout = timeout * timeout_ratio

    # Begin bandit loop
    print('Bandit started...')
    while True:

        # print loop counter
        print('\tIteration:', len(od_bandit.policy)+1)

        # select arm
        arm_index = od_bandit.select_arm()
        print('\t\tSelected arm:', arm_index)

        # play arm
        od_bandit.play_arm(arm_index, X_train, X_validation, y_validation, arm_timeout)

        # print policy
        print('\t\tpolicy:', od_bandit.policy)

        # Print rewards
        print('\t\trewards:', od_bandit.rewards)

        # update best arm
        od_bandit.update_best_arm()

        # check for timeout
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