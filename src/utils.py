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
    """ Function that imports the external PyOD models and adds them to Aut-Sklearn.

    Args:
        None

    Returns:
        None
    """

    # Import Auto-Sklearn-compatible PyOD algorithms
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
    
    # Add to Auto-Sklearn pipeline
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

def select_split_indices(y, print_flag=True):
    """ Function that takes the target attribute values, y 
    and returns indices for training and validation sets.

    Args:
        y (list or np.array): The target attribute labels y

    Returns:
        (list): A list indicating whether the corresponding 
        index will be part of the training set (0) or the 
        validation set (1).
    """

    # init
    selected_indices = []
    sacrificed_outliers = 0 # count of outliers sampled for the training set
    prob_outlier_in_valid = 0.95 # probability of outlier included in the validation set
    prob_normal_in_valid = 0.1 # probability of normal point included in the validation set
    for v in y:
        if v==1: # outlier
            if np.random.rand() <= prob_outlier_in_valid:
                selected_indices.append(1) # validation
            else:
                selected_indices.append(-1) # training
                sacrificed_outliers += 1 # will not be used for evaluation
        else:
            if np.random.rand() <= prob_normal_in_valid:
                selected_indices.append(1) # validation
            else:
                selected_indices.append(-1) # training
    # prints
    if print_flag:
        print('Number of total samples:', len(y))
        print('Number of training samples:', selected_indices.count(-1))
        print('Number of validation samples:', selected_indices.count(1))
        print('Number of outliers:', sum(y))
        print('Number of outliers in the training split:', sacrificed_outliers)
        print('Number of outliers in the validation split:', sum(y) - sacrificed_outliers)

    return selected_indices

def get_metric_result(cv_results):
    """ Function that takes as input the cv_results attribute
    of an Auto-Sklearn classifier and returns a number of
    results w.r.t. the defined metrics.

    Args:
        cv_results (dict): The cv_results attribute

    Returns:
        (list): list of applied models and their corresponding
        performance metrics.
    """

  # Get metric results function definition
    results = pd.DataFrame.from_dict(cv_results)
    cols = ['rank_test_scores', 'status', 'param_classifier:__choice__', 'mean_test_score', 'mean_fit_time']
    cols.extend([key for key in cv_results.keys() if key.startswith('metric_')]) # if there are additional metrics
    return results[cols].sort_values(['rank_test_scores'])

def show_results(automl):
    """ Takes as input an Auto-Sklearn estimator instance and
    visualizes/prints performance metrics.

    Args:
        automl (dict): Auto-Sklearn estimator object

    Returns:
        None
    """

    print('Auto-Sklearn execution details')
    print(automl.sprint_statistics())
    print('Top ranked model')
    print(automl.leaderboard(top_k=10))
    print('Top ranked model configuration')
    print(automl.show_models())

    # Call get_metric_result
    print(get_metric_result(automl.cv_results_).to_string(index=False))

    # Plot of performance over time
    automl.performance_over_time_.plot(
        x = 'Timestamp',
        kind = 'line',
        legend = True,
        title = 'Auto-sklearn ROC AUC score over time',
        grid = True,
    )

    # Save figure
    plt.savefig('perf-time.png')