import os
import pandas as pd
from scipy.io import arff
from datetime import timedelta as td
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


def preprocess_df(df, budget):
    '''
    Takes a DataFrame object with the raw performance results 
    from AutoSklearn and transform timestamps to seconds with
    the first run as a reference.

    Arguments:
        df(pd.DataFrame): the dataframe of the results
        budget(int): the total budget in seconds

    Returns:
        df(pd.DataFrame): the processed
    '''
    df.Timestamp = (df.Timestamp-df.Timestamp[0]).apply(td.total_seconds)
    n = df.shape[0]
    df.at[n, 'Timestamp'] = budget
    df = df.astype({"Timestamp": int})
    df.at[n, 'single_best_optimization_score'] = df.at[n -
                                                       1, 'single_best_optimization_score']
    df.at[n, 'single_best_test_score'] = df.at[n-1, 'single_best_test_score']
    df = df.drop_duplicates(
        subset=['Timestamp'], keep='last').reset_index(drop=True)
    return df


def fill_values(df, budget):
    '''
    Takes a DataFrame object with the raw performance results 
    from AutoSklearn and applies transformations and aggregations 
    to produce a final DataFrame object containing the average 
    performance values for every second based on `budget` parameter.

    Arguments:
        df(pd.DataFrame): the dataframe of the results
        budget(int): the total budget in seconds

    Returns:
        df(pd.DataFrame): the processed df with `budget` rows
    '''
    # Fill the missing values for `Timestamp` column
    ref_idx = 0  # the row index with the current max value
    for i in range(1, budget):
        if i not in df.Timestamp.values:
            n = df.shape[0]
            df.at[n, 'Timestamp'] = int(i)  # keep column name for consistency
            df.at[n, 'single_best_optimization_score'] = df.at[ref_idx,
                                                               'single_best_optimization_score']
            df.at[n, 'single_best_test_score'] = df.at[ref_idx,
                                                       'single_best_test_score']
        else:
            ref_idx = df.index[df['Timestamp'] == i][0]
            #print('Changing index at Timestamp =', i)
    df = df.iloc[1:, :]  # discard first row
    df = df[df['Timestamp'] > 0]
    df = df.sort_values(by='Timestamp').reset_index(
        drop=True)  # sort by timestamp
    # timestamp values (seconds) should be integer
    df = df.astype({"Timestamp": int})
    return df


def get_combinations(search_algorithm_list, validation_strategy_list, validation_size_list):
    '''
    Computes the combinations (as strings) of search algorithm,
    validation strategy and validation size.

    Arguments:
        search_algorithm_list(list): list of search algorithms
        validation_strategy_list(list): list of validation strategy values
        validation_size_list(list): list of validation size values

    Returns:
        cross_prod: the cross product list of combinations as strings
    '''
    cross_prod = []
    for algorithm in search_algorithm_list:
        for strategy in validation_strategy_list:
            for size in validation_size_list:
                cross_prod.append(
                    '{}_{}_{}.csv'.format(
                        algorithm,
                        strategy,
                        size
                    )
                )
    return cross_prod


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


def add_to_autosklearn_pipeline(classifiers):
    """
    Function that imports the provided PyOD models
    and adds them to AutoSklearn.

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
