import os
import random
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pyod.models.lof import LOF
from pyod.utils.data import generate_data
from sklearn.metrics import roc_auc_score
from util.util_functions import *

# Load env
load_dotenv()

# Import the dataset
df = import_kddcup99(os.getenv('dataset'))

# Split dataset to train, validation and test
y_train, X_train, y_validation, X_validation, y_test, X_test = split_df_kddcup99(df)

# Define algorithm parameters
timeout = 60.0 # time budget
total_time = 0.0 # total time
p_threshold = 0.8 # 1.0 for greedy, 0.0 for random
actions = np.array([1, 2, 5, 10, 15, 20, 30, 40, 50]) # actions (n_neighbors parameter)
action_values = np.full(len(actions), 0.0) # action values (expected rewards)
selected_actions = [] # sequence of selected actions
sum_of_rewards = 0 # sum of rewards
best_action = None # best action

# Begin bandit loop
print('Bandit started (ε-greedy)...')
while True:

  print('\tIteration:', len(selected_actions)+1)

  # Simulate random selection
  p = random.uniform(0, 1)

  # Select action
  max_index = action_values.argmax()
  if(p < p_threshold):
    k = actions[max_index]
    print('\t\tGreedy selection (exploitation), k =',k)
  else:
    k = random.choice(np.delete(actions, max_index))
    print('\t\tNon-greedy selection (exploration), k =',k)
  selected_actions.append(k)

  # Train LOF detector
  clf = LOF(n_neighbors=k)
  # Time this phase
  start_time = time.time()
  clf.fit(X_train)
  end_time = time.time()
  print("\t\tTraining time: %s seconds" % (end_time - start_time))
  total_time += end_time - start_time

  # Get the prediction on the vaLidation data
  y_validation_pred = clf.predict(X_validation)  # outlier labels (0 or 1)
  y_validation_scores = clf.decision_function(X_validation)  # outlier scores

  roc_auc = np.round(roc_auc_score(y_validation, y_validation_scores), decimals=4)
  sum_of_rewards += roc_auc
  idx = np.where(actions == k)
  # Q(A) <- Q(A) + 1/N(A)[R - Q(A)]
  if(np.sum(action_values == k) != 0.0):
    action_values[idx] = action_values[idx] + (1/np.sum(action_values == k))(roc_auc - action_values[idx])
  else:
    action_values[idx] = roc_auc

  # Print new state
  print('\t\tactions:', actions)
  print('\t\taction_values:', action_values)
  print('\t\tselected_actions:', selected_actions)

  # Update the best action (model)
  max_index = action_values.argmax()
  best_action = LOF(n_neighbors=actions[max_index])

  if total_time > timeout:
    break

print('Bandit finished in', len(selected_actions), 'iterations.')
print('\tTotal average reward:', np.round(sum_of_rewards/len(selected_actions), decimals=4))
print('\tSelected actions:', selected_actions)

# Apply the best model to the test set
print('Applying the best model to the test data...')
print('\tBest model: ', best_action.n_neighbors)
best_action.fit(X_test)

# Get the prediction on the test data
y_test_pred = best_action.predict(X_test)  # outlier labels (0 or 1)
y_test_scores = best_action.decision_function(X_test)  # outlier scores

roc_auc = np.round(roc_auc_score(y_test, y_test_scores), decimals=4)
print("\troc_auc on test set: ", roc_auc)