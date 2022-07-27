import os
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
from datetime import timedelta as td


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
            # print('Changing index at Timestamp =', i)
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


def get_search_space(clf_name, search_space):
    """
    Function that returns the hyperparameter search space
    of a classifier whose name is provided as an argument.

    Args:
        clf_name (str): the classifier name
        search_space (str): the type of search space

    Returns:
        the classifier's search space
    """
    if clf_name == 'ABODClassifier':
        if search_space == "sp1":
            from pyod_models.sp1.abod import ABODClassifier
        elif search_space == "sp2":
            from pyod_models.sp2.abod import ABODClassifier
        else:
            from pyod_models.default.abod import ABODClassifier
        return ABODClassifier.get_hyperparameter_search_space()
    if clf_name == 'CBLOFClassifier':
        if search_space == "sp1":
            from pyod_models.sp1.cblof import CBLOFClassifier
        elif search_space == "sp2":
            from pyod_models.sp2.cblof import CBLOFClassifier
        else:
            from pyod_models.default.cblof import CBLOFClassifier
        return CBLOFClassifier.get_hyperparameter_search_space()
    if clf_name == 'COPODClassifier':
        if search_space == "sp1":
            from pyod_models.sp1.copod import COPODClassifier
        elif search_space == "sp2":
            from pyod_models.sp2.copod import COPODClassifier
        else:
            from pyod_models.default.copod import COPODClassifier
        return COPODClassifier.get_hyperparameter_search_space()
    if clf_name == 'ECODClassifier':
        if search_space == "sp1":
            from pyod_models.sp1.ecod import ECODClassifier
        elif search_space == "sp2":
            from pyod_models.sp2.ecod import ECODClassifier
        else:
            from pyod_models.default.ecod import ECODClassifier
        return ECODClassifier.get_hyperparameter_search_space()
    if clf_name == 'HBOSClassifier':
        if search_space == "sp1":
            from pyod_models.sp1.hbos import HBOSClassifier
        elif search_space == "sp2":
            from pyod_models.sp2.hbos import HBOSClassifier
        else:
            from pyod_models.default.hbos import HBOSClassifier
        return HBOSClassifier.get_hyperparameter_search_space()
    if clf_name == 'IForestClassifier':
        if search_space == "sp1":
            from pyod_models.sp1.iforest import IForestClassifier
        elif search_space == "sp2":
            from pyod_models.sp2.iforest import IForestClassifier
        else:
            from pyod_models.default.iforest import IForestClassifier
        return IForestClassifier.get_hyperparameter_search_space()
    if clf_name == 'KNNClassifier':
        if search_space == "sp1":
            from pyod_models.sp1.knn import KNNClassifier
        elif search_space == "sp2":
            from pyod_models.sp2.knn import KNNClassifier
        else:
            from pyod_models.default.knn import KNNClassifier
        return KNNClassifier.get_hyperparameter_search_space()
    if clf_name == 'LMDDClassifier':
        if search_space == "sp1":
            from pyod_models.sp1.lmdd import LMDDClassifier
        elif search_space == "sp2":
            from pyod_models.sp2.lmdd import LMDDClassifier
        else:
            from pyod_models.default.lmdd import LMDDClassifier
        return LMDDClassifier.get_hyperparameter_search_space()
    if clf_name == 'LOFClassifier':
        if search_space == "sp1":
            from pyod_models.sp1.lof import LOFClassifier
        elif search_space == "sp2":
            from pyod_models.sp2.lof import LOFClassifier
        else:
            from pyod_models.default.lof import LOFClassifier
        return LOFClassifier.get_hyperparameter_search_space()
    if clf_name == 'MCDClassifier':
        if search_space == "sp1":
            from pyod_models.sp1.mcd import MCDClassifier
        elif search_space == "sp2":
            from pyod_models.sp2.mcd import MCDClassifier
        else:
            from pyod_models.default.mcd import MCDClassifier
        return MCDClassifier.get_hyperparameter_search_space()
    if clf_name == 'OCSVMClassifier':
        if search_space == "sp1":
            from pyod_models.sp1.ocsvm import OCSVMClassifier
        elif search_space == "sp2":
            from pyod_models.sp2.ocsvm import OCSVMClassifier
        else:
            from pyod_models.default.ocsvm import OCSVMClassifier
        return OCSVMClassifier.get_hyperparameter_search_space()
    if clf_name == 'PCAClassifier':
        if search_space == "sp1":
            from pyod_models.sp1.pca import PCAClassifier
        elif search_space == "sp2":
            from pyod_models.sp2.pca import PCAClassifier
        else:
            from pyod_models.default.pca import PCAClassifier
        return PCAClassifier.get_hyperparameter_search_space()
    if clf_name == 'RODClassifier':
        if search_space == "sp1":
            from pyod_models.sp1.rod import RODClassifier
        elif search_space == "sp2":
            from pyod_models.sp2.rod import RODClassifier
        else:
            from pyod_models.default.rod import RODClassifier
        return RODClassifier.get_hyperparameter_search_space()
    if clf_name == 'SOSClassifier':
        if search_space == "sp1":
            from pyod_models.sp1.sos import SOSClassifier
        elif search_space == "sp2":
            from pyod_models.sp2.sos import SOSClassifier
        else:
            from pyod_models.default.sos import SOSClassifier
        return SOSClassifier.get_hyperparameter_search_space()


def get_search_space_size(clf_list, search_space):
    """
    Function that calculates and returns the estimated size
    of the hyperparameter search space defined by the provided
    list of algorithms.

    Args:
        clf_list (list): the list of classifiers
        search_space: the type of search space
    Returns:
        size (int): the estimated size of the combined search space
    """
    size = 0
    for clf in clf_list:
        clf_size = get_search_space(clf, search_space).estimate_size()
        size += clf_size
    return int(size)


# Plotting function for H1
def plot_h1_results(
    w0,
    h0,
    save_figures,
    figures_dir,
    results_processed_path,  # str
    dataset_list,  # list(str)
    total_budget,  # int
    validation_strategy,  # int
    validation_size,  # int
    eval_type,  # str
    color_list,  # list(str)
    y_range=[0.5, 1.0]  # y range
):
    # Figure
    width = w0 * len(dataset_list)
    height = h0 * len(dataset_list)
    fig = plt.figure(figsize=(width, height))  # grid dimensions
    fig.subplots_adjust(wspace=0.4, hspace=0.3)  # space between plots
    # fig.suptitle('Performance on {} set for {} validation set of size {}'\
    # .format(eval_type, validation_strategy, validation_size), y=0.93)
    # Plots
    for i, dataset in enumerate(dataset_list):
        for filename in os.listdir(results_processed_path):
            if dataset in filename \
                    and validation_strategy in filename \
                    and str(validation_size)+'.' in filename:
                # import performance data as DataFrame
                df = pd.read_csv(os.path.join(
                    results_processed_path, filename))
                # x-axis
                x = df.Timestamp
                # score based on eval_type
                if eval_type == 'validation':
                    y = df.single_best_optimization_score
                elif eval_type == 'test':
                    y = df.single_best_test_score
                # plot
                lbl = filename.split('_')[1]
                if lbl == 'ue':
                    label = 'uniform exploration'
                elif lbl == 'random':
                    label = 'random search'
                elif lbl == 'default':
                    label = 'default'
                elif lbl == 'smac':
                    label = 'SMAC'
                else:
                    print('No valid label')
                ax = plt.subplot(3, 3, i + 1)
                ax.set_ylim(y_range)
                ax.set_xlabel('seconds', fontsize=18)
                ax.set_ylabel('score', fontsize=18)
                if 'default' in filename:
                    ax.plot(x, y, label=label,
                            color=color_list[label], linestyle='--', linewidth=3)
                else:
                    ax.plot(x, y, label=label, color=color_list[label])
                ax.grid()
                handles, labels = ax.get_legend_handles_labels()
                labels, handles = zip(
                    *sorted(zip(labels, handles), key=lambda t: t[0]))
                plt.title(dataset)
    # savefig puts it in the right place
    fig.legend(handles, labels, loc=(0.74, 0.58), fontsize='medium')
    # save
    if save_figures:
        plt.savefig(os.path.join(figures_dir, '{}_{}_{}_{}.pdf'.format(
            eval_type, total_budget, validation_strategy, validation_size)), bbox_inches="tight")


# Plotting function for H2
def plot_h2_results(
    w0,
    h0,
    save_figures,
    figures_dir,
    results_processed_path,  # str
    dataset_list,  # list(str)
    search_strategy,  # str
    total_budget,  # int
    validation_size,  # int
    eval_type,  # str
    color_list,  # list(str)
    y_range=[0.5, 1.0]  # y range
):
    # Figure
    width = w0 * len(dataset_list)
    height = h0 * len(dataset_list)
    fig = plt.figure(figsize=(width, height))  # grid dimensions
    fig.subplots_adjust(wspace=0.4, hspace=0.3)  # space between plots
    fig.suptitle('Performance on {} set for {} search and validation set of size {}'
                 .format(eval_type, search_strategy, validation_size), y=0.93)
    # Plots
    for i, dataset in enumerate(dataset_list):
        for filename in os.listdir(results_processed_path):
            if dataset in filename and \
                    search_strategy in filename and \
                    str(validation_size)+'.' in filename:
                df = pd.read_csv(os.path.join(
                    results_processed_path, filename))
                # x-axis (time)
                x = df.Timestamp
                # score based on eval_type
                if eval_type == 'validation':
                    y = df.single_best_optimization_score
                elif eval_type == 'test':
                    y = df.single_best_test_score
                # plot
                label = filename.split('_')[2]
                ax = plt.subplot(3, 3, i + 1)
                ax.set_ylim(y_range)
                ax.set_xlabel('seconds', fontsize=18)
                ax.set_ylabel('score', fontsize=18)
                ax.plot(x, y, label=label, color=color_list[label])
                ax.grid()
                handles, labels = ax.get_legend_handles_labels()
                labels, handles = zip(
                    *sorted(zip(labels, handles), key=lambda t: t[0]))
                # ax.legend(handles, labels, loc='lower right')
                plt.title(dataset)
                if save_figures:
                    plt.savefig(os.path.join(figures_dir, '{}_{}_{}_{}.png'.format(
                        eval_type, total_budget, search_strategy, validation_size)))
    fig.legend(handles, labels, loc=(0.797, 0.237), fontsize='large')


# Plotting function for H3
def plot_h3_results(
    w0,
    h0,
    save_figures,
    figures_dir,
    results_processed_path,  # str
    dataset_list,  # list(str)
    search_strategy,  # str
    total_budget,  # int
    validation_strategy,  # str
    validation_size_list,  # list(int)
    eval_type,  # str
    color_list,  # list(str)
    y_range=[0.5, 1.0]  # y range
):
    # Figure
    width = w0 * len(dataset_list)
    height = h0 * len(dataset_list)
    fig = plt.figure(figsize=(width, height))  # grid dimensions
    fig.subplots_adjust(wspace=0.4, hspace=0.3)  # space between plots
    # Plots
    for i, dataset in enumerate(dataset_list):
        for filename in os.listdir(results_processed_path):
            if (dataset in filename and
                search_strategy in filename and
                validation_strategy in filename
                    and any(str(size) in filename for size in validation_size_list)):
                # import performance data as DataFrame
                df = pd.read_csv(os.path.join(
                    results_processed_path, filename))
                # x-axis (seconds)
                x = df.Timestamp
                # score based on eval_type
                if eval_type == 'validation':
                    y = df.single_best_optimization_score
                elif eval_type == 'test':
                    y = df.single_best_test_score
                # plot
                label = int(filename.split('_')[3].split('.')[0])
                ax = plt.subplot(3, 3, i + 1)
                ax.set_ylim(y_range)
                ax.set_xlabel('seconds', fontsize=18)
                ax.set_ylabel('score', fontsize=18)
                ax.plot(x, y, label=label, color=color_list[label])
                if dataset != 'Waveform':
                    handles, labels = ax.get_legend_handles_labels()
                    # convert to integer for sorting
                    labels = [int(l) for l in labels]
                    labels, handles = zip(
                        *sorted(zip(labels, handles), key=lambda t: t[0]))
                plt.title(dataset)
    # fig.legend(handles, labels, loc=(0.075, 0.615), fontsize='medium')
    if save_figures:
        plt.savefig(os.path.join(figures_dir, '{}_{}_{}_{}.pdf'.format(
            eval_type, total_budget, search_strategy, validation_strategy)), bbox_inches="tight")

# Plotting function for H3 mixed


def plot_h3_results_mixed(
    w0,
    h0,
    save_figures,
    figures_dir,
    results_processed_path,  # str
    dataset_list,  # list(str)
    search_strategy,  # str
    total_budget,  # int
    validation_strategy,  # str
    validation_size_list,  # list(int)
    color_list,  # list(str)
    y_range=[0.5, 1.0]  # y range
):
    # Figure
    width = w0 * 2 * len(dataset_list)
    height = h0 * 2 * len(dataset_list)
    fig = plt.figure(figsize=(width, height))  # grid dimensions
    fig.subplots_adjust(wspace=0.4, hspace=0.3)  # space between plots
    # Plots
    for j, eval_type in enumerate(['validation', 'test']):
        for i, dataset in enumerate(dataset_list):
            for filename in os.listdir(results_processed_path):
                if (dataset in filename and
                    search_strategy in filename and
                    validation_strategy in filename
                        and any(str(size) in filename for size in validation_size_list)):
                    # import performance data as DataFrame
                    df = pd.read_csv(os.path.join(
                        results_processed_path, filename))
                    # x-axis (seconds)
                    x = df.Timestamp
                    # score based on eval_type
                    if eval_type == 'validation':
                        y = df.single_best_optimization_score
                    elif eval_type == 'test':
                        y = df.single_best_test_score
                    # plot
                    label = int(filename.split('_')[3].split('.')[0])
                    ax = plt.subplot(3, 3, j*3 + i+1)
                    ax.set_ylim(y_range)
                    ax.set_xlabel('seconds', fontsize=18)
                    ax.set_ylabel('score', fontsize=18)
                    ax.plot(x, y, label=label, color=color_list[label])
                    handles, labels = ax.get_legend_handles_labels()
                    # convert to integer for sorting
                    labels = [int(l) for l in labels]
                    labels, handles = zip(
                        *sorted(zip(labels, handles), key=lambda t: t[0]))
                    plt.title(dataset)
        fig.legend(handles, labels, loc=(0.86, 0.58), fontsize='medium')
        if save_figures:
            plt.savefig(os.path.join(figures_dir, '{}_{}_{}_{}.pdf'.format(
                eval_type, total_budget, search_strategy, validation_strategy)), bbox_inches="tight")


# Plotting function with std
def plot_results_with_std_1(
    w0,
    h0,
    save_figures,
    figures_dir,
    results_processed_path,  # str
    dataset_list,  # list(str)
    total_budget,  # int
    search_strategy,  # str
    validation_strategy,  # int
    validation_size,  # int
    eval_type  # str
):
    # Figure
    width = w0 * len(dataset_list)
    height = h0 * len(dataset_list)
    fig = plt.figure(figsize=(width, height))  # grid dimensions
    fig.subplots_adjust(wspace=0.4, hspace=0.3)  # space between plots
    fig.suptitle('{} performance on {} set for {} validation set of size {}'
                 .format(search_strategy, eval_type, validation_strategy, validation_size), y=0.93)
    # Plots
    for i, dataset in enumerate(dataset_list):
        for filename in os.listdir(results_processed_path):
            if dataset in filename \
                    and search_strategy in filename \
                    and validation_strategy in filename \
                    and str(validation_size)+'.' in filename:
                # import performance data as DataFrame
                df = pd.read_csv(os.path.join(
                    results_processed_path, filename),)
                x = df.Timestamp
                # score based on eval_type
                if eval_type == 'validation':
                    y = df.single_best_optimization_score
                    dy = df.single_best_optimization_score_std
                elif eval_type == 'test':
                    y = df.single_best_test_score
                    dy = df.single_best_test_score_std
                # plot
                label = filename.split('_')[1]
                ax = plt.subplot(3, 3, i + 1)
                ax.set_ylim([0.5, 1.])
                ax.set_xlabel('seconds', fontsize=18)
                ax.set_ylabel('score', fontsize=18)
                ax.plot(x, y, label=label)
                ax.fill_between(x, y - dy, y + dy, alpha=0.2)
                ax.grid()
                handles, labels = ax.get_legend_handles_labels()
                labels, handles = zip(
                    *sorted(zip(labels, handles), key=lambda t: t[0]))
                plt.title(dataset)
    if save_figures:
        plt.savefig(os.path.join(figures_dir, '{}_{}_{}_{}_std.png'.format(
            search_strategy, eval_type, total_budget, validation_strategy, validation_size)))


# Plotting function with std
def plot_results_with_std_2(
    w0,
    h0,
    save_figures,
    figures_dir,
    results_processed_path,  # str
    dataset_list,  # list (str)
    dataset,  # str
    total_budget,  # int
    search_strategy_list,  # str
    validation_strategy,  # int
    validation_size,  # int
    eval_type  # str
):
    # Figure
    width = w0 * len(dataset_list)
    height = h0 * len(dataset_list)
    fig = plt.figure(figsize=(width, height))  # grid dimensions
    fig.subplots_adjust(wspace=0.4, hspace=0.3)  # space between plots
    fig.suptitle('Performance on {} set for {} validation set of size {}'
                 .format(eval_type, validation_strategy, validation_size), y=0.93)
    # Plots
    for i, search_strategy in enumerate(search_strategy_list):
        for filename in os.listdir(results_processed_path):
            if search_strategy in filename \
                    and dataset in filename \
                    and validation_strategy in filename \
                    and str(validation_size)+'.' in filename:
                # import performance data as DataFrame
                df = pd.read_csv(os.path.join(
                    results_processed_path, filename),)
                # x-axis (time)
                x = df.Timestamp
                # score based on eval_type
                if eval_type == 'validation':
                    y = df.single_best_optimization_score
                    dy = df.single_best_optimization_score_std
                elif eval_type == 'test':
                    y = df.single_best_test_score
                    dy = df.single_best_test_score_std
                # plot
                label = filename.split('_')[1]
                ax = plt.subplot(3, 3, i + 1)
                ax.set_ylim([0.5, 1.])
                ax.set_xlabel('seconds', fontsize=18)
                ax.set_ylabel('score', fontsize=18)
                ax.plot(x, y, label=label)
                ax.fill_between(x, y - dy, y + dy, alpha=0.2)
                ax.grid()
                handles, labels = ax.get_legend_handles_labels()
                labels, handles = zip(
                    *sorted(zip(labels, handles), key=lambda t: t[0]))
                ax.legend(handles, labels, loc='lower right',
                          fontsize='medium')
                plt.title(dataset)
    if save_figures:
        plt.savefig(os.path.join(figures_dir, '{}_{}_{}_{}_std.png'.format(
            dataset, eval_type, total_budget, validation_strategy, validation_size)))
