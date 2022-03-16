import os
import pandas as pd
from scipy.io import arff
from matplotlib import pyplot as plt
from datetime import timedelta as td
from autosklearn.pipeline.components.classification import add_classifier


def import_dataset(filepath):
    """ 
    Function that reads a arff-formatted dataset and returns a dataframe.

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


def add_pyod_models_to_pipeline():
    """ 
    Function that imports the external PyOD models and adds them to Aut-Sklearn.

    Args:
        None

    Returns:
        None
    """
    # Import Auto-Sklearn-compatible PyOD algorithms
    from pyod_models.abod import ABODClassifier  # probabilistic
    from pyod_models.cblof import CBLOFClassifier  # proximity-based
    from pyod_models.cof import COFClassifier  # proximity-based
    from pyod_models.copod import COPODClassifier  # probabilistic
    from pyod_models.ecod import ECODClassifier  # probabilistic
    from pyod_models.hbos import HBOSClassifier  # proximity-based
    from pyod_models.iforest import IForestClassifier  # outlier ensembles
    from pyod_models.knn import KNNClassifier  # proximity-based
    from pyod_models.lmdd import LMDDClassifier  # linear model
    from pyod_models.loci import LOCIClassifier  # proximity-based
    from pyod_models.lof import LOFClassifier  # proximity-based
    from pyod_models.mad import MADClassifier  # probabilistic
    from pyod_models.mcd import MCDClassifier  # linear model
    from pyod_models.ocsvm import OCSVMClassifier  # linear model
    from pyod_models.pca import PCAClassifier  # linear model
    from pyod_models.rod import RODClassifier  # proximity-based
    from pyod_models.sod import SODClassifier  # proximity-based
    from pyod_models.sos import SOSClassifier  # probabilistic
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


def create_search_space(algos):
    """
    Create search space for random search.

    Args:
        algos: the list of algorithms to include

    Returns:
        models: the included models
        search_space: the hyperparameter search space
    """
    # Define PyOD algorithms
    models = {}  # default model instances
    search_space = {}  # the hyperparameter search space per algorithm
    # ABOD
    if 'ABODClassifier' in algos:
        from pyod_models.abod import ABODClassifier
        search_space['abod'] = ABODClassifier.get_hyperparameter_search_space()
        models['abod'] = ABODClassifier(
            **search_space['abod'].get_default_configuration())
    # CBLOF
    if 'CBLOFClassifier' in algos:
        from pyod_models.cblof import CBLOFClassifier
        search_space['cblof'] = CBLOFClassifier.get_hyperparameter_search_space()
        models['cblof'] = CBLOFClassifier(
            **search_space['cblof'].get_default_configuration())
    # COPOD
    if 'COPODClassifier' in algos:
        from pyod_models.copod import COPODClassifier
        search_space['copod'] = COPODClassifier.get_hyperparameter_search_space()
        models['copod'] = COPODClassifier(
            **search_space['copod'].get_default_configuration())
    # ECOD
    if 'ECODClassifier' in algos:
        from pyod_models.ecod import ECODClassifier
        search_space['ecod'] = ECODClassifier.get_hyperparameter_search_space()
        models['ecod'] = ECODClassifier(
            **search_space['ecod'].get_default_configuration())
    # HBOS
    if 'HBOSClassifier' in algos:
        from pyod_models.hbos import HBOSClassifier
        search_space['hbos'] = HBOSClassifier.get_hyperparameter_search_space()
        models['hbos'] = HBOSClassifier(
            **search_space['hbos'].get_default_configuration())
    # IForest
    if 'IForestClassifier' in algos:
        from pyod_models.iforest import IForestClassifier
        search_space['ifor'] = IForestClassifier.get_hyperparameter_search_space()
        models['ifor'] = IForestClassifier(
            **search_space['ifor'].get_default_configuration())
    # KNN
    if 'KNNClassifier' in algos:
        from pyod_models.knn import KNNClassifier
        search_space['knn'] = KNNClassifier.get_hyperparameter_search_space()
        models['knn'] = KNNClassifier(
            **search_space['knn'].get_default_configuration())
    # LMDD
    if 'LMDDClassifier' in algos:
        from pyod_models.lmdd import LMDDClassifier
        search_space['lmdd'] = LMDDClassifier.get_hyperparameter_search_space()
        models['lmdd'] = LMDDClassifier(
            **search_space['lmdd'].get_default_configuration())
    # LOF
    if 'LOFClassifier' in algos:
        from pyod_models.lof import LOFClassifier
        search_space['lof'] = LOFClassifier.get_hyperparameter_search_space()
        models['lof'] = LOFClassifier(
            **search_space['lof'].get_default_configuration())
    # MCD
    if 'MCDClassifier' in algos:
        from pyod_models.mcd import MCDClassifier
        search_space['mcd'] = MCDClassifier.get_hyperparameter_search_space()
        models['mcd'] = MCDClassifier(
            **search_space['mcd'].get_default_configuration())
    # OCSVM
    if 'OCSVMClassifier' in algos:
        from pyod_models.ocsvm import OCSVMClassifier
        search_space['ocsvm'] = OCSVMClassifier.get_hyperparameter_search_space()
        models['ocsvm'] = OCSVMClassifier(
            **search_space['ocsvm'].get_default_configuration())
    # PCA
    if 'PCAClassifier' in algos:
        from pyod_models.pca import PCAClassifier
        search_space['pca'] = PCAClassifier.get_hyperparameter_search_space()
        models['pca'] = PCAClassifier(
            **search_space['pca'].get_default_configuration())
    # ROD
    if 'RODClassifier' in algos:
        from pyod_models.rod import RODClassifier
        search_space['rod'] = RODClassifier.get_hyperparameter_search_space()
        models['rod'] = RODClassifier(
            **search_space['rod'].get_default_configuration())
    # SOS
    if 'SOSClassifier' in algos:
        from pyod_models.sos import SOSClassifier
        search_space['sos'] = SOSClassifier.get_hyperparameter_search_space()
        models['sos'] = SOSClassifier(
            **search_space['sos'].get_default_configuration())
    # return statement
    return models, search_space


def balanced_split(y, print_flag=False):
    """ 
    Function that takes the target attribute values, y 
    and returns indices for training and validation, with
    equal ratio of inliers/outliers for the validation set.

    Args:
        y (list or np.array): The target attribute labels y

    Returns:
        selected_indices (list): A list indicating whether the 
        corresponding index will be part of the training set (0) 
        or the validation set (1).
    """
    # Initialize
    selected_indices = []  # initially all in training
    norm_train = 0
    norm_test = 0
    out_train = 0
    out_test = 0
    for v in y:
        if v == 1:  # outlier
            if out_train > 0:  # one will have to go to train
                selected_indices.append(1)  # test
                out_test += 1
            else:
                selected_indices.append(-1)  # training
                out_train += 1
        else:  # normal
            if out_test > norm_test:
                selected_indices.append(1)  # test
                norm_test += 1
            else:
                selected_indices.append(-1)  # training
                norm_train += 1
    # Prints
    if print_flag:
        print('Number of total samples to split:', len(y))
        print('Number of outliers in training:', out_train)
        print('Number of outliers in test:', out_test)
        print('Number of normal points in training:', norm_train)
        print('Number of normal points in test:', norm_test)
    # Return indices
    return selected_indices


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


def plot_performance(out_dirname, total_budget):
    '''
    Function that plots the combined search performance
    over time.
    '''
    # Import csv files
    path = os.path.join(os.path.dirname(__file__),
                        'output', out_dirname, 'performance')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for file in os.listdir(path):
        df = pd.read_csv(os.path.join(path, file), parse_dates=['Timestamp'])
        # Plot validation set performance
        x1 = (df.Timestamp-df.Timestamp[0]).apply(td.total_seconds)
        y1 = df.single_best_optimization_score
        x1.at[x1.shape[0]] = total_budget
        y1.at[y1.shape[0]] = y1.at[y1.shape[0]-1]
        ax1.plot(x1, y1, label=file.split('.')[0])
        ax1.set_ylim([0.5, 1.])
        ax1.set_xlabel('seconds')
        ax1.set_ylabel('score')
        ax1.set_title('Performance on validation set')
        ax1.grid()
        # Plot test set performance
        x2 = (df.Timestamp-df.Timestamp[0]).apply(td.total_seconds)
        y2 = df.single_best_test_score
        x2.at[x2.shape[0]] = total_budget
        y2.at[y2.shape[0]] = y2.at[y2.shape[0]-1]
        ax2.plot(x2, y2, label=file.split('.')[0])
        ax2.set_ylim([0.5, 1.])
        ax2.set_xlabel('seconds')
        ax2.set_title('Performance on test set')
        ax2.grid()
    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')
    plt.savefig(os.path.join(path, 'all.png'))
