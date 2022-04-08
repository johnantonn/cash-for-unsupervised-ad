import os
import json
import time
import numpy as np
import pandas as pd
from scipy.io import arff
from autosklearn.pipeline.components.classification import add_classifier


def import_dataset(filepath):
    """
    Function that reads a arff-formatted dataset
    and returns a dataframe.

    Args:
        filename (str): The name of the file

    Returns:
        (df): The dataframe with the data contents
    """

    # File does not exist
    if not os.path.exists(filepath):
        raise FileNotFoundError("filepath {} does not exist".format(filepath))

    # Load file to dataframe
    data = arff.loadarff(filepath)
    df = pd.DataFrame(data[0])
    df.outlier = df.outlier.str.decode("utf-8")
    df['outlier'] = df['outlier'].map({'yes': 1, 'no': 0})
    if 'id' in df:
        del df['id']

    return df


def load_config_values():
    """
    Function that loads the environment variables
    in the main script.

    Args:
        None

    Returns:
        [...] the values of the environment variables
    """
    # Read json config file
    config_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), 'config.json')
    with open(config_path, "r") as jsonfile:
        config = json.load(jsonfile)
    # Create local params
    datasets = config['datasets']
    dataset_iter = config['dataset_iter']
    classifiers = config['classifiers']
    total_budget = config['total_budget']
    per_run_budget = config['per_run_budget']
    v_strategy_default_flag = bool(config['v_strategy_default_flag'])
    v_size_default_flag = bool(config['v_size_default_flag'])
    if config['output_dir']:
        output_dir = config['output_dir']
    else:
        output_dir = time.strftime("%Y%m%d_%H%M%S")
    # Check if directory already exists
    output_dir_path = os.path.join(os.path.dirname(
        __file__), 'output', output_dir)
    if os.path.exists(output_dir_path):
        raise ValueError(
            "Output directory `{}` already exists.".format(output_dir))
    # Return params
    return datasets, dataset_iter, \
        classifiers, total_budget, per_run_budget, \
        v_strategy_default_flag, v_size_default_flag, \
        output_dir


def add_to_autosklearn_pipeline(classifiers):
    """
    Function that imports the provided PyOD models
    and adds them to Aut-Sklearn.

    Args:
        classifiers (list): the list of classifiers
        to add

    Returns:
        None
    """
    for clf_name in classifiers:
        if clf_name == 'ABODClassifier':
            from pyod_models.abod import ABODClassifier
            add_classifier(ABODClassifier)
        if clf_name == 'CBLOFClassifier':
            from pyod_models.cblof import CBLOFClassifier
            add_classifier(CBLOFClassifier)
        if clf_name == 'COPODClassifier':
            from pyod_models.copod import COPODClassifier
            add_classifier(COPODClassifier)
        if clf_name == 'ECODClassifier':
            from pyod_models.ecod import ECODClassifier
            add_classifier(ECODClassifier)
        if clf_name == 'HBOSClassifier':
            from pyod_models.hbos import HBOSClassifier
            add_classifier(HBOSClassifier)
        if clf_name == 'IForestClassifier':
            from pyod_models.iforest import IForestClassifier
            add_classifier(IForestClassifier)
        if clf_name == 'KNNClassifier':
            from pyod_models.knn import KNNClassifier
            add_classifier(KNNClassifier)
        if clf_name == 'LMDDClassifier':
            from pyod_models.lmdd import LMDDClassifier
            add_classifier(LMDDClassifier)
        if clf_name == 'LOFClassifier':
            from pyod_models.lof import LOFClassifier
            add_classifier(LOFClassifier)
        if clf_name == 'MCDClassifier':
            from pyod_models.mcd import MCDClassifier
            add_classifier(MCDClassifier)
        if clf_name == 'OCSVMClassifier':
            from pyod_models.ocsvm import OCSVMClassifier
            add_classifier(OCSVMClassifier)
        if clf_name == 'PCAClassifier':
            from pyod_models.pca import PCAClassifier
            add_classifier(PCAClassifier)
        if clf_name == 'RODClassifier':
            from pyod_models.rod import RODClassifier
            add_classifier(RODClassifier)
        if clf_name == 'SOSClassifier':
            from pyod_models.sos import SOSClassifier
            add_classifier(SOSClassifier)


def get_search_space(clf_name):
    """
    Function that returns the hyperparameter search space
    of a classifier whose name is provided as an argument.

    Args:
        clf_name (str): the classifier name

    Returns:
        search_space: the classifier's search space
    """
    if clf_name == 'ABODClassifier':
        from pyod_models.abod import ABODClassifier
        return ABODClassifier.get_hyperparameter_search_space()
    if clf_name == 'CBLOFClassifier':
        from pyod_models.cblof import CBLOFClassifier
        return CBLOFClassifier.get_hyperparameter_search_space()
    if clf_name == 'COPODClassifier':
        from pyod_models.copod import COPODClassifier
        return COPODClassifier.get_hyperparameter_search_space()
    if clf_name == 'ECODClassifier':
        from pyod_models.ecod import ECODClassifier
        return ECODClassifier.get_hyperparameter_search_space()
    if clf_name == 'HBOSClassifier':
        from pyod_models.hbos import HBOSClassifier
        return HBOSClassifier.get_hyperparameter_search_space()
    if clf_name == 'IForestClassifier':
        from pyod_models.iforest import IForestClassifier
        return IForestClassifier.get_hyperparameter_search_space()
    if clf_name == 'KNNClassifier':
        from pyod_models.knn import KNNClassifier
        return KNNClassifier.get_hyperparameter_search_space()
    if clf_name == 'LMDDClassifier':
        from pyod_models.lmdd import LMDDClassifier
        return LMDDClassifier.get_hyperparameter_search_space()
    if clf_name == 'LOFClassifier':
        from pyod_models.lof import LOFClassifier
        return LOFClassifier.get_hyperparameter_search_space()
    if clf_name == 'MCDClassifier':
        from pyod_models.mcd import MCDClassifier
        return MCDClassifier.get_hyperparameter_search_space()
    if clf_name == 'OCSVMClassifier':
        from pyod_models.ocsvm import OCSVMClassifier
        return OCSVMClassifier.get_hyperparameter_search_space()
    if clf_name == 'PCAClassifier':
        from pyod_models.pca import PCAClassifier
        return PCAClassifier.get_hyperparameter_search_space()
    if clf_name == 'RODClassifier':
        from pyod_models.rod import RODClassifier
        return RODClassifier.get_hyperparameter_search_space()
    if clf_name == 'SOSClassifier':
        from pyod_models.sos import SOSClassifier
        return SOSClassifier.get_hyperparameter_search_space()


def get_search_space_size(clf_list):
    """
    Function that calculates and returns the estimated size
    of the hyperparameter search space defined by the provided
    list of algorithms.

    Args:
        clf_list (list): the list of classifiers

    Returns:
        size (int): the estimated size of the combined search space
    """
    size = 0
    for clf in clf_list:
        clf_size = get_search_space(clf).estimate_size()
        size += clf_size
    return int(size)


def get_validation_strategy_list(default=True):
    """
    Function that takes the hypothesis number and
    returns a list of values for the validation 
    strategy to be used in the search.

    Args:
        default (boolean): whether to use the default
        value 'stratified' or both values

    Returns:
        (list): A list of possible validation strategies

    """
    if default:
        return ['stratified']
    else:
        return ['stratified', 'balanced']


def get_validation_size_list(dataset, iter=1, default=True):
    """
    Function that takes the name of the dataset and the
    hypothesis number and returns a list of values for
    the validation set size to be used in the search.

    Args:
        dataset (str): The name of the dataset
        h_num (string): hypothesis number {0, 1, 2, 3}
        default (boolean): Whether to return the default
        value

    Returns:
        (list): A list of possible validation set sizes

    """
    if default:
        return [20, 50, 100, 200]
    else:
        dataset_dir = os.path.join(os.path.dirname(
            __file__), 'data/processed/' + dataset + '/iter'+str(iter))
        y_train = pd.read_csv(os.path.join(dataset_dir, 'y_train.csv'))
        size = round(0.3 * y_train.shape[0])
        return [size]


def train_valid_split(labels, validation_strategy='stratified',
                      validation_size=200, print_flag=True):
    """
    Function that takes a list of `labels` and returns
    indices for training and validation sets according
    to the provided `valid_method` parameter.

    Args:
        labels (np.array): The target attribute labels
        validation_strategy (string): 'stratified' or 'balanced'
        validation_size (int): size of the validation set
        print_flag (boolean): whether to print split stats

    Returns:
        idx_train_val (list): A list indicating whether the
        corresponding index will be part of the training set (0)
        or the validation set (1).
    """

    # Initialize
    idx_train_val = -np.ones(len(labels), dtype=int)  # all in training
    labels_0 = np.where(labels == 0)[0]  # normal points
    labels_1 = np.where(labels == 1)[0]  # outliers
    p_outlier = 0.  # outlier percentage

    if(validation_strategy == 'stratified'):
        # Equal to outlier percentage in training set
        p_outlier = float(len(labels_1)/len(labels))
    elif(validation_strategy == 'balanced'):
        # Equal to 50% no matter what
        p_outlier = 0.5
    else:
        raise ValueError(
            'The provided value of `validation_strategy`: {} is not supported!'.format(validation_strategy))

    # number of outliers in validation set
    n_outlier_val = round(p_outlier * validation_size)
    # number of normal points in validation set
    n_normal_val = validation_size - n_outlier_val
    # indices of outliers for validation
    idx_outlier_val = np.random.choice(
        labels_1, size=n_outlier_val, replace=False)
    # indices of normal points for validation
    idx_normal_val = np.random.choice(
        labels_0, size=n_normal_val, replace=False)
    # concatenate (should be of length validation_size)
    idx_val = np.concatenate(
        (idx_normal_val, idx_outlier_val))
    # construct the output list
    idx_train_val[idx_val] = 1

    # print details
    if print_flag:
        n_outlier_train = len(labels_1)-n_outlier_val
        n_train = len(labels) - validation_size
        print('Outliers in train_eval set:', len(labels_1))
        print('Outliers in training set:', n_outlier_train)
        print('Outliers in validation set:', n_outlier_val)
        print('Percentage of outliers in train_eval set:',
              len(labels_1)/len(labels))
        print('Percentage of outliers in training set:',
              n_outlier_train/n_train)
        print('Percentage of outliers in validation set:',
              n_outlier_val/validation_size)

    # Return indices
    return idx_train_val


def get_metric_result(cv_results):
    """
    Function that takes as input the cv_results attribute
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
    cols = [
        'rank_test_scores',
        'status',
        'param_classifier:__choice__',
        'mean_test_score',
        'mean_fit_time'
    ]
    cols.extend([key for key in cv_results.keys()
                 if key.startswith('metric_')])
    return results[cols].sort_values(['rank_test_scores'])
