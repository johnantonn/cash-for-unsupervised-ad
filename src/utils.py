import os
import json
import time
import numpy as np
from autosklearn.pipeline.components.classification import add_classifier


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
    iterations = config['iterations']
    classifiers = config['classifiers']
    search_space = config['search_space']
    validation_set_split_strategies = config['validation_set_split_strategies']
    validation_set_sizes = config['validation_set_sizes']
    total_budget = config['total_budget']
    per_run_budget = config['per_run_budget']
    output_dir = time.strftime("%Y%m%d_%H%M%S")
    # Check if directory already exists
    output_dir_path = os.path.join(os.path.dirname(
        __file__), 'output', output_dir)
    if os.path.exists(output_dir_path):
        raise ValueError(
            "Output directory `{}` already exists.".format(output_dir))
    # Return params
    return datasets, iterations, classifiers, search_space, \
        validation_set_split_strategies, validation_set_sizes, \
        total_budget, per_run_budget, output_dir


def add_to_autosklearn_pipeline(classifiers, search_space):
    """
    Function that imports the provided PyOD models
    and adds them to Aut-Sklearn.
    Args:
        classifiers (list): the list of classifiers
        to add
        search_space (string): sp1, sp2 or default
    Returns:
        None
    """
    for clf_name in classifiers:
        if clf_name == 'ABODClassifier':
            if search_space == "sp1":
                from pyod_models.sp1.abod import ABODClassifier
            elif search_space == "sp2":
                from pyod_models.sp2.abod import ABODClassifier
            else:
                from pyod_models.default.abod import ABODClassifier
            add_classifier(ABODClassifier)
        if clf_name == 'CBLOFClassifier':
            if search_space == "sp1":
                from pyod_models.sp1.cblof import CBLOFClassifier
            elif search_space == "sp2":
                from pyod_models.sp2.cblof import CBLOFClassifier
            else:
                from pyod_models.default.cblof import CBLOFClassifier
            add_classifier(CBLOFClassifier)
        if clf_name == 'COPODClassifier':
            if search_space == "sp1":
                from pyod_models.sp1.copod import COPODClassifier
            elif search_space == "sp2":
                from pyod_models.sp2.copod import COPODClassifier
            else:
                from pyod_models.default.copod import COPODClassifier
            add_classifier(COPODClassifier)
        if clf_name == 'ECODClassifier':
            if search_space == "sp1":
                from pyod_models.sp1.ecod import ECODClassifier
            elif search_space == "sp2":
                from pyod_models.sp2.ecod import ECODClassifier
            else:
                from pyod_models.default.ecod import ECODClassifier
            add_classifier(ECODClassifier)
        if clf_name == 'HBOSClassifier':
            if search_space == "sp1":
                from pyod_models.sp1.hbos import HBOSClassifier
            elif search_space == "sp2":
                from pyod_models.sp2.hbos import HBOSClassifier
            else:
                from pyod_models.default.hbos import HBOSClassifier
            add_classifier(HBOSClassifier)
        if clf_name == 'IForestClassifier':
            if search_space == "sp1":
                from pyod_models.sp1.iforest import IForestClassifier
            elif search_space == "sp2":
                from pyod_models.sp2.iforest import IForestClassifier
            else:
                from pyod_models.default.iforest import IForestClassifier
            add_classifier(IForestClassifier)
        if clf_name == 'KNNClassifier':
            if search_space == "sp1":
                from pyod_models.sp1.knn import KNNClassifier
            elif search_space == "sp2":
                from pyod_models.sp2.knn import KNNClassifier
            else:
                from pyod_models.default.knn import KNNClassifier
            add_classifier(KNNClassifier)
        if clf_name == 'LMDDClassifier':
            if search_space == "sp1":
                from pyod_models.sp1.lmdd import LMDDClassifier
            elif search_space == "sp2":
                from pyod_models.sp2.lmdd import LMDDClassifier
            else:
                from pyod_models.default.lmdd import LMDDClassifier
            add_classifier(LMDDClassifier)
        if clf_name == 'LOFClassifier':
            if search_space == "sp1":
                from pyod_models.sp1.lof import LOFClassifier
            elif search_space == "sp2":
                from pyod_models.sp2.lof import LOFClassifier
            else:
                from pyod_models.default.lof import LOFClassifier
            add_classifier(LOFClassifier)
        if clf_name == 'MCDClassifier':
            if search_space == "sp1":
                from pyod_models.sp1.mcd import MCDClassifier
            elif search_space == "sp2":
                from pyod_models.sp2.mcd import MCDClassifier
            else:
                from pyod_models.default.mcd import MCDClassifier
            add_classifier(MCDClassifier)
        if clf_name == 'OCSVMClassifier':
            if search_space == "sp1":
                from pyod_models.sp1.ocsvm import OCSVMClassifier
            elif search_space == "sp2":
                from pyod_models.sp2.ocsvm import OCSVMClassifier
            else:
                from pyod_models.default.ocsvm import OCSVMClassifier
            add_classifier(OCSVMClassifier)
        if clf_name == 'PCAClassifier':
            if search_space == "sp1":
                from pyod_models.sp1.pca import PCAClassifier
            elif search_space == "sp2":
                from pyod_models.sp2.pca import PCAClassifier
            else:
                from pyod_models.default.pca import PCAClassifier
            add_classifier(PCAClassifier)
        if clf_name == 'RODClassifier':
            if search_space == "sp1":
                from pyod_models.sp1.rod import RODClassifier
            elif search_space == "sp2":
                from pyod_models.sp2.rod import RODClassifier
            else:
                from pyod_models.default.rod import RODClassifier
            add_classifier(RODClassifier)
        if clf_name == 'SOSClassifier':
            if search_space == "sp1":
                from pyod_models.sp1.sos import SOSClassifier
            elif search_space == "sp2":
                from pyod_models.sp2.sos import SOSClassifier
            else:
                from pyod_models.default.sos import SOSClassifier
            add_classifier(SOSClassifier)


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
