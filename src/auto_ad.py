# Imports
import sklearn.metrics
from sklearn.model_selection import train_test_split, PredefinedSplit
from autosklearn.pipeline.components.classification import add_classifier
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import roc_auc, average_precision
from utils import import_dataset, select_split_indices, add_pyod_models_to_pipeline, \
    get_metric_result, show_results

# Import DataFrame
df = import_dataset('../data/Annthyroid_withoutdupl_norm_02_v01.arff')
N = 2000 # number of samples to use
if(len(df) > N):
        df = df.sample(n=N)

# Extract X, y
X  = df.iloc[:, :-1]
y = df['outlier']
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.3,
    random_state = 82 # my current percentage grade
)

# Add PyOD models to Auto-Sklearn's pipeline
add_pyod_models_to_pipeline()

# Split data based on labels
split_indices = select_split_indices(y_train)

# Define custom resampling strategy
resampling_strategy = PredefinedSplit(test_fold=split_indices)

# Build and fit a automl classifier
automl = AutoSklearnClassifier(
    include = {
      'classifier': [
          'ABODClassifier',
          'CBLOFClassifier',
          #'COFClassifier', # currently not used
          'COPODClassifier',
          'ECODClassifier',
          'HBOSClassifier',
          'IForestClassifier',
          'KNNClassifier',
          'LMDDClassifier',
          'LOCIClassifier',
          'LOFClassifier',
          #'MADClassifier', # only for univariate data
          'MCDClassifier',
          'OCSVMClassifier',
          'PCAClassifier',
          'RODClassifier',
          #'SODClassifier', # currently not used
          'SOSClassifier'
      ],
      'feature_preprocessor': ["no_preprocessing"],
    },
    exclude = None,
    metric = roc_auc,
    scoring_functions = [ roc_auc, average_precision],
    time_left_for_this_task = 60,
    per_run_time_limit = 10,
    ensemble_size = 1,
    initial_configurations_via_metalearning = 0,
    delete_tmp_folder_after_terminate = False,
    resampling_strategy = resampling_strategy,
)

# Fit
print('Running fit.')
automl.fit(X_train, y_train, X_test, y_test, dataset_name='Shuttle')

# Refit due to PredefinedSplit
print('Running refit.')
automl.refit(X_train, y_train)

# Evaluate best model on test set
y_pred=automl.predict_proba(X_test)
score=sklearn.metrics.roc_auc_score(y_test,y_pred[:,1])
print("ROC AUC score on test set:", score)

# Print smac results
print(get_metric_result(automl.cv_results_).to_string(index=False))

# Visualize performance over time
show_results(automl)
