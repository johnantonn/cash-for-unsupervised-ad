import os
import pandas as pd
import numpy as np
from scipy.io import arff
from matplotlib import pyplot as plt
from autosklearn.pipeline.components.classification import add_classifier

def import_dataset(filepath):
    """ Function that reads the KDDCup99 dataset and returns a dataframe.

    Args:
        filename (str): The name of the file

    Returns:
        (df): The dataframe with the data contents
    """

    # If file does not exist
    if not os.path.exists(filepath):
        raise FileNotFoundError("filepath %s does not exist" % filepath)

    # Load file to a df
    data = arff.loadarff(filepath)
    df = pd.DataFrame(data[0])
    df.outlier = df.outlier.str.decode("utf-8")
    df['outlier'] = df['outlier'].map({'yes':1,'no':0}) 
    if 'id' in df:
        del df['id']

    return df

def add_pyod_models_to_pipeline():
    # Import Auto-Sklearn implemented PyOD classifiers
    from pyod_models.abod import ABODClassifier # probabilistic
    from pyod_models.cblof import CBLOFClassifier # proximity-based
    from pyod_models.cof import COFClassifier # proximity-based
    from pyod_models.copod import COPODClassifier # probabilistic
    from pyod_models.ecod import ECODClassifier # probabilistic
    from pyod_models.hbos import HBOSClassifier # proximity-based
    from pyod_models.iforest import IForestClassifier # outlier ensembles
    from pyod_models.knn import KNNClassifier # proximity-based
    from pyod_models.lmdd import LMDDClassifier # linear model
    from pyod_models.loci import LOCIClassifier # proximity-based
    from pyod_models.lof import LOFClassifier # proximity-based
    from pyod_models.mad import MADClassifier # probabilistic
    from pyod_models.mcd import MCDClassifier # linear model
    from pyod_models.ocsvm import OCSVMClassifier # linear model
    from pyod_models.pca import PCAClassifier # linear model
    from pyod_models.rod import RODClassifier # proximity-based
    from pyod_models.sod import SODClassifier # proximity-based
    from pyod_models.sos import SOSClassifier # probabilistic
    
    # Add algorithms to the pipeline components of Auto-Sklearn
    add_classifier(ABODClassifier)
    add_classifier(CBLOFClassifier)
    add_classifier(COFClassifier)
    add_classifier(COPODClassifier)
    add_classifier(ECODClassifier)
    add_classifier(HBOSClassifier)
    add_classifier(IForestClassifier)
    add_classifier(KNNClassifier)
    add_classifier(LMDDClassifier)
    add_classifier(LOCIClassifier)
    add_classifier(LOFClassifier)
    add_classifier(MADClassifier)
    add_classifier(MCDClassifier)
    add_classifier(OCSVMClassifier)
    add_classifier(PCAClassifier)
    add_classifier(RODClassifier)
    add_classifier(SODClassifier)
    add_classifier(SOSClassifier)

def select_split_indices(y_train):
    # Pre-defined split indices for train and validation
    selected_indices = []
    sacrificed_outliers = 0 # count of outliers sampled for the training set
    for v in y_train:
        if v==1: # outlier
            if np.random.rand()>0.05:
                selected_indices.append(1) # validation
            else:
                selected_indices.append(0) # training
                sacrificed_outliers += 1 # will not be used for evaluation
        else:
            if np.random.rand()>0.9:
                selected_indices.append(1) # validation
            else:
                selected_indices.append(0) # training

    # prints
    print('Length of selected indices:', len(selected_indices))
    print('Number of total samples:', len(y_train))
    print('Number of outliers:', sum(y_train))
    print('Number of sacrificed outliers:', sacrificed_outliers)
    print('Number of validation samples:', sum(selected_indices))
    print('Number of training samples:', len(y_train) - sum(selected_indices))

    return selected_indices


def get_metric_result(cv_results):
  # Get metric results function definition
    results = pd.DataFrame.from_dict(cv_results)
    cols = ['rank_test_scores', 'status', 'param_classifier:__choice__', 'mean_test_score', 'mean_fit_time']
    cols.extend([key for key in cv_results.keys() if key.startswith('metric_')]) # if there are additional metrics
    return results[cols].sort_values(['rank_test_scores'])

def show_results(automl):
    # auto-sklearn execution details
    print(automl.sprint_statistics())
    # Top ranked model
    print(automl.leaderboard(top_k=10))
    # Top ranked model configuration
    print()
    print(automl.show_models())

    # Call the function
    print(get_metric_result(automl.cv_results_).to_string(index=False))

    # Plot training performance over time
    automl.performance_over_time_.plot(
        x='Timestamp',
        kind='line',
        legend=True,
        title='Auto-sklearn ROC AUC score over time',
        grid=True,
    )

    plt.savefig('perf-time.png')