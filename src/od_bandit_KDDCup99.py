from pyod.models.lof import LOF
from pyod.utils.data import generate_data
from sklearn.metrics import roc_auc_score
import numpy as np
import time
import random
from scipy.io import arff
import pandas as pd

# Import the dataset
data = arff.loadarff('datasets/KDDCup99_withoutdupl_norm_idf.arff')
df = pd.DataFrame(data[0])
df.outlier = df.outlier.str.decode("utf-8")
df['outlier'] = df['outlier'].map({'yes':1,'no':0}) 

# Split dataset to training and test sets
n_full = len(df)
n_train_full = round(0.9 * n_full)
n_test_full = n_full - n_train_full
df_train = df.iloc[:n_train_full, :]
df_test = df.iloc[n_train_full:, :]
y_train_full = df_train['outlier'].to_numpy()
X_train_full = df_train.drop(columns=['id', 'outlier']).to_numpy()
y_test_full = df_test['outlier'].to_numpy()
X_test_full = df_test.drop(columns=['id', 'outlier']).to_numpy()

# Define general parameters
timeout = time.time() + 2*60 # seconds from now
actions = np.array([1, 2, 5, 10, 15, 20, 30, 40, 50]) # actions (n_neighbors parameter)
action_values = np.full(len(actions), 0.0) # action values (expected rewards)
selected_actions = [] # sequence of selected actions
sum_of_rewards = 0 # sum of rewards
p_threshold = 0.8 # 1.0 for greedy, 0.0 for random
size_factor = 5 # factor reducing per iteration dataset size
n_train_sample = round(n_train_full / size_factor)
n_test_sample = round(n_test_full / size_factor)

# Begin bandit loop
print('Bandit started...')
while True:

  print('\tIteration:', len(selected_actions)+1)
  
  # Select random portion of the dataset
  X_train = X_train_full[np.random.choice(n_train_full, n_train_sample, replace=False), :]
  # y_train = y_train_full[np.random.choice(y_train_full.shape[0], n_train_sample, replace=False)]
  X_test = X_test_full[np.random.choice(n_test_full, n_test_sample, replace=False), :]
  y_test = y_test_full[np.random.choice(n_test_full, n_test_sample, replace=False)]

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

  # train LOF detector
  clf_name = 'LOF'
  clf = LOF(n_neighbors=k)
  clf.fit(X_train)

  # get the prediction on the test data
  y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
  y_test_scores = clf.decision_function(X_test)  # outlier scores

  roc_auc = np.round(roc_auc_score(y_test, y_test_scores), decimals=4)
  sum_of_rewards += roc_auc
  idx = np.where(actions == k)
  action_values[idx] = roc_auc
  
  # Print new state
  print('\t\tactions:', actions)
  print('\t\taction_values:', action_values)

  if time.time() > timeout:
    break

print('Bandit finished in', len(selected_actions), 'iterations.')
print('\tTotal average reward:', np.round(sum_of_rewards/len(selected_actions), decimals=4))
print('\tSelected actions:', selected_actions)