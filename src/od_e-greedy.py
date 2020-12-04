import os
import random
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pyod.models.lof import LOF
from pyod.models.knn import KNN
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
timeout = 5*60 # time budget
total_time = 0.0 # total time
p_threshold = 0.8 # 1.0 for greedy, 0.0 for random
actions = np.array([LOF(), KNN()]) # actions
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
    curr_action = actions[max_index]
    print('\t\tGreedy selection (exploitation), curr_action =',curr_action)
  else:
    curr_action = random.choice(np.delete(actions, max_index))
    print('\t\tNon-greedy selection (exploration), curr_action =',curr_action)
  selected_actions.append(curr_action)

  # Train selected model (time training)
  start_time = time.time()
  curr_action.fit(X_train)
  end_time = time.time()
  print("\t\tTraining time: %s seconds" % (end_time - start_time))
  total_time += end_time - start_time

  # Get the prediction on the vaLidation data
  y_validation_pred = curr_action.predict(X_validation)  # outlier labels (0 or 1)
  y_validation_scores = curr_action.decision_function(X_validation)  # outlier scores

  roc_auc = np.round(roc_auc_score(y_validation, y_validation_scores), decimals=4)
  sum_of_rewards += roc_auc
  idx = np.where(actions == curr_action)
  # Q(A) <- Q(A) + 1/N(A)[R - Q(A)]
  if(np.sum(action_values == curr_action) != 0):
    action_values[idx] = action_values[idx] + (1/np.sum(action_values == curr_action))(roc_auc - action_values[idx])
  else:
    action_values[idx] = roc_auc

  # Print new state
  print('\t\tactions:', actions)
  print('\t\taction_values:', action_values)
  print('\t\tselected_actions:', selected_actions)

  # Update the best action (model)
  max_index = action_values.argmax()
  best_action = actions[max_index]

  if total_time > timeout:
    break

print('Bandit finished in', len(selected_actions), 'iterations.')
print('\tTotal average reward:', np.round(sum_of_rewards/len(selected_actions), decimals=4))
print('\tSelected actions:', selected_actions)

# Apply the best model to the test set
print('Applying the best model to the test data...')
print('\tBest model: ', best_action)
best_action.fit(X_test)

# Get the prediction on the test data
y_test_pred = best_action.predict(X_test)  # outlier labels (0 or 1)
y_test_scores = best_action.decision_function(X_test)  # outlier scores

roc_auc = np.round(roc_auc_score(y_test, y_test_scores), decimals=4)
print("\troc_auc on test set: ", roc_auc)