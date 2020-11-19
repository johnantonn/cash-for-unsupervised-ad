# Random Selection

# Importing the libraries
from pyod.models.lof import LOF
from scipy.io import arff
import numpy as np
import pandas as pd
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize
from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score

# Importing the dataset
data = arff.loadarff('datasets/KDDCup99_withoutdupl_norm_idf.arff')
df = pd.DataFrame(data[0])
df.outlier = df.outlier.str.decode("utf-8")
df['outlier'] = df['outlier'].map({'yes':1,'no':0}) 

# Define training set as subset of the original dataset
n_train = 10000
train_df = df.sample(n = n_train)
y_train = train_df['outlier'].to_numpy()
X_train = train_df.drop(columns=['id', 'outlier'])

# Define test set as subset of the original dataset
n_test = 1000
test_df = df.sample(n = n_test)
y_test = test_df['outlier'].to_numpy()
X_test = test_df.drop(columns=['id', 'outlier'])

# train LOF detector
clf_name = 'LOF'
clf = LOF(n_neighbors=50)
clf.fit(X_train)

# get the prediction labels and outlier scores of the training data
y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
y_train_scores = clf.decision_scores_  # raw outlier scores

# get the prediction on the test data
y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
y_test_scores = clf.decision_function(X_test)  # outlier scores

# evaluate and print the results
print("\nOn Training Data:")
evaluate_print(clf_name, y_train, y_train_scores)
print("\nOn Test Data:")
evaluate_print(clf_name, y_test, y_test_scores)

# print('Precision @ n:', precision_n_scores(y_test, y_test_scores))
# print('ROC AUC:', np.round(roc_auc_score(y_test, y_test_scores), decimals=4))