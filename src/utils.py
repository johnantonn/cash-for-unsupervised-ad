import os
import pandas as pd
from scipy.io import arff
from matplotlib import pyplot as plt
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


def add_pyod_models_to_pipeline():
    """ 
    Function that imports the external PyOD models
    and adds them to Aut-Sklearn.

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


def clf_lookup(clf_name):
    """
    Function that returns a classifier's object instance
    given its name.

    Args:
        clf_name (str): the classifier name

    Returns:
        clf: the classifier's object instance
    """
    # ABOD
    if clf_name == 'ABODClassifier':
        from pyod_models.abod import ABODClassifier
        return ABODClassifier(
            **ABODClassifier.get_hyperparameter_search_space().get_default_configuration())
    # CBLOF
    if clf_name == 'CBLOFClassifier':
        from pyod_models.cblof import CBLOFClassifier
        return CBLOFClassifier(
            **CBLOFClassifier.get_hyperparameter_search_space().get_default_configuration())
    # COPOD
    if clf_name == 'COPODClassifier':
        from pyod_models.copod import COPODClassifier
        return COPODClassifier(
            **COPODClassifier.get_hyperparameter_search_space().get_default_configuration())
    # ECOD
    if clf_name == 'ECODClassifier':
        from pyod_models.ecod import ECODClassifier
        return ECODClassifier(
            **ECODClassifier.get_hyperparameter_search_space().get_default_configuration())
    # HBOS
    if clf_name == 'HBOSClassifier':
        from pyod_models.hbos import HBOSClassifier
        return HBOSClassifier(
            **HBOSClassifier.get_hyperparameter_search_space().get_default_configuration())
    # IForest
    if clf_name == 'IForestClassifier':
        from pyod_models.iforest import IForestClassifier
        return IForestClassifier(
            **IForestClassifier.get_hyperparameter_search_space().get_default_configuration())
    # KNN
    if clf_name == 'KNNClassifier':
        from pyod_models.knn import KNNClassifier
        return KNNClassifier(
            **KNNClassifier.get_hyperparameter_search_space().get_default_configuration())
    # LMDD
    if clf_name == 'LMDDClassifier':
        from pyod_models.lmdd import LMDDClassifier
        return LMDDClassifier(
            **LMDDClassifier.get_hyperparameter_search_space().get_default_configuration())
    # LOF
    if clf_name == 'LOFClassifier':
        from pyod_models.lof import LOFClassifier
        return LOFClassifier(
            **LOFClassifier.get_hyperparameter_search_space().get_default_configuration())
    # MCD
    if clf_name == 'MCDClassifier':
        from pyod_models.mcd import MCDClassifier
        return MCDClassifier(
            **MCDClassifier.get_hyperparameter_search_space().get_default_configuration())
    # OCSVM
    if clf_name == 'OCSVMClassifier':
        from pyod_models.ocsvm import OCSVMClassifier
        return OCSVMClassifier(
            **OCSVMClassifier.get_hyperparameter_search_space().get_default_configuration())
    # PCA
    if clf_name == 'PCAClassifier':
        from pyod_models.pca import PCAClassifier
        return PCAClassifier(
            **PCAClassifier.get_hyperparameter_search_space().get_default_configuration())
    # ROD
    if clf_name == 'RODClassifier':
        from pyod_models.rod import RODClassifier
        return RODClassifier(
            **RODClassifier.get_hyperparameter_search_space().get_default_configuration())
    # SOS
    if clf_name == 'SOSClassifier':
        from pyod_models.sos import SOSClassifier
        return SOSClassifier(
            **SOSClassifier.get_hyperparameter_search_space().get_default_configuration())


def get_search_space(clf_name):
    """
    Function that returns the hyperparameter search space
    of a classifier whose name is provided as an argument.

    Args:
        clf_name (str): the classifier name

    Returns:
        search_space: the classifier's search space
    """
    # ABOD
    if clf_name == 'ABODClassifier':
        from pyod_models.abod import ABODClassifier
        return ABODClassifier.get_hyperparameter_search_space()
    # CBLOF
    if clf_name == 'CBLOFClassifier':
        from pyod_models.cblof import CBLOFClassifier
        return CBLOFClassifier.get_hyperparameter_search_space()
    # COPOD
    if clf_name == 'COPODClassifier':
        from pyod_models.copod import COPODClassifier
        return COPODClassifier.get_hyperparameter_search_space()
    # ECOD
    if clf_name == 'ECODClassifier':
        from pyod_models.ecod import ECODClassifier
        return ECODClassifier.get_hyperparameter_search_space()
    # HBOS
    if clf_name == 'HBOSClassifier':
        from pyod_models.hbos import HBOSClassifier
        return HBOSClassifier.get_hyperparameter_search_space()
    # IForest
    if clf_name == 'IForestClassifier':
        from pyod_models.iforest import IForestClassifier
        return IForestClassifier.get_hyperparameter_search_space()
    # KNN
    if clf_name == 'KNNClassifier':
        from pyod_models.knn import KNNClassifier
        return KNNClassifier.get_hyperparameter_search_space()
    # LMDD
    if clf_name == 'LMDDClassifier':
        from pyod_models.lmdd import LMDDClassifier
        return LMDDClassifier.get_hyperparameter_search_space()
    # LOF
    if clf_name == 'LOFClassifier':
        from pyod_models.lof import LOFClassifier
        return LOFClassifier.get_hyperparameter_search_space()
    # MCD
    if clf_name == 'MCDClassifier':
        from pyod_models.mcd import MCDClassifier
        return MCDClassifier.get_hyperparameter_search_space()
    # OCSVM
    if clf_name == 'OCSVMClassifier':
        from pyod_models.ocsvm import OCSVMClassifier
        return OCSVMClassifier.get_hyperparameter_search_space()
    # PCA
    if clf_name == 'PCAClassifier':
        from pyod_models.pca import PCAClassifier
        return PCAClassifier.get_hyperparameter_search_space()
    # ROD
    if clf_name == 'RODClassifier':
        from pyod_models.rod import RODClassifier
        return RODClassifier.get_hyperparameter_search_space()
    # SOS
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
        size: the estimated size of the combined search space
    """
    size = 0
    for clf in clf_list:
        clf_size = get_search_space(clf).estimate_size()
        size += clf_size
    return size


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
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    for file in os.listdir(path):
        df = pd.read_csv(os.path.join(path, file), parse_dates=['Timestamp'])
        # validation score
        x = (df.Timestamp-df.Timestamp[0]).apply(td.total_seconds)
        x.at[x.shape[0]] = total_budget
        yval = df.single_best_optimization_score
        yval.at[yval.shape[0]] = yval.at[yval.shape[0]-1]
        # test score
        ytest = df.single_best_test_score
        ytest.at[ytest.shape[0]] = ytest.at[ytest.shape[0]-1]
        if 'balanced' in file:
            ax1.plot(x, yval, label=file.split('.')[0])
            ax1.set_ylim([0.5, 1.])
            ax1.set_xlabel('seconds')
            ax1.set_ylabel('score')
            ax1.set_title('balanced-validation')
            ax1.grid()
            ax2.plot(x, ytest, label=file.split('.')[0])
            ax2.set_ylim([0.5, 1.])
            ax2.set_xlabel('seconds')
            ax2.set_title('balanced-test')
            ax2.grid()
            for ax in [ax1, ax2]:
                handles, labels = ax.get_legend_handles_labels()
                labels, handles = zip(
                    *sorted(zip(labels, handles), key=lambda t: t[0]))
                ax.legend(handles, labels, loc='lower right')
        elif 'stratified' in file:
            ax3.plot(x, yval, label=file.split('.')[0])
            ax3.set_ylim([0.5, 1.])
            ax3.set_xlabel('seconds')
            ax3.set_ylabel('score')
            ax3.set_title('stratified-validation')
            ax3.grid()
            ax4.plot(x, ytest, label=file.split('.')[0])
            ax4.set_ylim([0.5, 1.])
            ax4.set_xlabel('seconds')
            ax4.set_title('stratified-test')
            ax4.grid()
            for ax in [ax3, ax4]:
                handles, labels = ax.get_legend_handles_labels()
                labels, handles = zip(
                    *sorted(zip(labels, handles), key=lambda t: t[0]))
                ax.legend(handles, labels, loc='lower right')
    fig_title = 'all_' + str(total_budget) + '.png'
    plt.savefig(os.path.join(path, fig_title))
