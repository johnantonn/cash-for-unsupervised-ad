import os
import random
import pandas as pd
from matplotlib import pyplot as plt
from datetime import timedelta as td
from smac.facade.roar_facade import ROAR
from smac.scenario.scenario import Scenario
from ConfigSpace.configuration_space import ConfigurationSpace
from autosklearn.pipeline.components import data_preprocessing
from autosklearn.pipeline.constants import SPARSE, DENSE, UNSIGNED_DATA, INPUT
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import roc_auc, average_precision
from sklearn.model_selection import PredefinedSplit
from utils import train_valid_split


class NoPreprocessing(AutoSklearnPreprocessingAlgorithm):

    def __init__(self, **kwargs):
        """ This preprocessors does not change the data """
        # Some internal checks makes sure parameters are set
        for key, val in kwargs.items():
            setattr(self, key, val)

    def fit(self, X, Y=None):
        return self

    def transform(self, X):
        return X

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            'shortname': 'NoPreprocessing',
            'name': 'NoPreprocessing',
            'handles_regression': True,
            'handles_classification': True,
            'handles_multiclass': True,
            'handles_multilabel': True,
            'handles_multioutput': True,
            'is_deterministic': True,
            'input': (SPARSE, DENSE, UNSIGNED_DATA),
            'output': (INPUT,)
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        return ConfigurationSpace()  # Return an empty configuration as there is None


class Search:

    def __init__(self, dataset_name, dataset_iter, classifiers, validation_strategy, validation_size, total_budget,
                 per_run_budget, output_dir, random_state):
        self.dataset_name = dataset_name  # dataset name
        self.dataset_iter = dataset_iter  # dataset iteration number
        self.dataset_dir = os.path.join(os.path.dirname(
            __file__), 'data/processed/' + self.dataset_name + '/iter' + str(self.dataset_iter))
        self.classifiers = classifiers  # PyOD algorithms to use
        self.validation_strategy = validation_strategy  # split strategy
        self.validation_size = validation_size  # validation set size
        self.total_budget = total_budget  # total budget in seconds
        self.per_run_budget = per_run_budget  # per run budget in seconds
        self.random_state = random_state  # random state for reproducibility
        self.resampling_strategy = None  # resampling strategy
        self.automl = None  # autosklearn classifier
        self.cv_results = None  # as DataFrame
        self.performance_over_time = None  # as DataFrame
        self.output_dir = os.path.join(os.path.dirname(
            __file__), 'output', output_dir)  # output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.save_metadata()  # save metadata info

    def build_automl(self):
        return AutoSklearnClassifier(
            include={
                'classifier': self.classifiers,
                'feature_preprocessor': ['no_preprocessing'],
                'data_preprocessor': ['NoPreprocessing']
            },
            exclude=None,
            metric=roc_auc,
            scoring_functions=[roc_auc, average_precision],
            time_left_for_this_task=self.total_budget,
            per_run_time_limit=self.per_run_budget,
            ensemble_size=0,
            initial_configurations_via_metalearning=0,
            resampling_strategy=self.resampling_strategy,
            get_smac_object_callback=self.smac_object_callback
        )

    def print_search_details(self):
        print('Running Search:')
        print('  Dataset:\t', self.dataset_name + ' ' + str(self.dataset_iter))
        print('  Type:\t\t', self.search_type)
        print('  Budget:\t', self.total_budget)
        print('  Validation:\t ({}, {})'.format(
            self.validation_strategy, self.validation_size))
        print('  Classifiers:\t', str(self.classifiers))

    def run(self):
        # Print serach details
        self.print_search_details()
        # Import train/test data
        X_train = pd.read_csv(os.path.join(self.dataset_dir, 'X_train.csv'))
        y_train = pd.read_csv(os.path.join(self.dataset_dir, 'y_train.csv'))
        X_test = pd.read_csv(os.path.join(self.dataset_dir, 'X_test.csv'))
        y_test = pd.read_csv(os.path.join(self.dataset_dir, 'y_test.csv'))
        # Resampling strategy
        try:
            train_valid_indices = train_valid_split(
                labels=y_train,
                validation_strategy=self.validation_strategy,
                validation_size=self.validation_size
            )
            self.resampling_strategy = PredefinedSplit(
                test_fold=train_valid_indices)
        except:
            raise RuntimeError('Failed to create validation set for ({}, {})!'.format(
                self.validation_strategy, self.validation_size))
        # Add NoPreprocessing component to auto-sklearn
        data_preprocessing.add_preprocessor(
            NoPreprocessing)
        # Build automl classifier
        self.automl = self.build_automl()
        self.automl.fit(X_train, y_train, X_test,
                        y_test, dataset_name=self.dataset_name+str(self.dataset_iter))
        # Save results
        self.cv_results = pd.DataFrame.from_dict(self.automl.cv_results_)
        self.performance_over_time = self.automl.performance_over_time_

    def print_summary(self):
        print(self.automl.sprint_statistics())

    def print_rankings(self):
        # Columns to include
        cols = [
            'rank_test_scores',
            'status',
            'param_classifier:__choice__',
            'mean_test_score',
            'mean_fit_time'
        ]
        cols.extend([key for key in self.cv_results.keys()
                    if key.startswith('metric_')])
        print(self.cv_results[cols].sort_values(
            ['rank_test_scores']).to_string(index=False))

    def plot_scores(self):
        # Filename and directory
        fname = '{}_{}_{}_{}_{}'.format(
            self.dataset_name,
            str(self.dataset_iter),
            self.search_type,
            self.validation_strategy,
            self.validation_size
        )
        plots_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        # Extract scores
        val_scores = self.performance_over_time.loc[:,
                                                    ('Timestamp', 'single_best_optimization_score')]
        test_scores = self.performance_over_time.loc[:,
                                                     ('Timestamp', 'single_best_test_score')]
        xval = (val_scores.Timestamp -
                val_scores.Timestamp[0]).apply(td.total_seconds)
        yval = val_scores.single_best_optimization_score
        xtest = (test_scores.Timestamp -
                 test_scores.Timestamp[0]).apply(td.total_seconds)
        ytest = test_scores.single_best_test_score
        # Modify scores for better plotting
        xval.at[xval.shape[0]] = self.total_budget
        yval.at[yval.shape[0]] = yval.at[yval.shape[0]-1]
        xtest.at[xtest.shape[0]] = self.total_budget
        ytest.at[ytest.shape[0]] = ytest.at[ytest.shape[0]-1]
        # Plot
        plt.figure()
        plt.plot(xval, yval)
        plt.plot(xtest, ytest)
        plt.legend(['validation', 'test'])
        plt.ylim([0.5, 1.])
        plt.xlabel('seconds')
        plt.ylabel('score')
        plt.title(fname)
        plt.grid()
        plt.show()
        plt.savefig(os.path.join(plots_dir, fname+'.png'))

    def save_metadata(self):
        import csv
        # filename
        fname = 'metadata.csv'
        fpath = os.path.join(self.output_dir, fname)
        fexists = os.path.isfile(fpath)
        # data
        data = [
            self.dataset_name,
            self.dataset_iter,
            self.search_type,
            len(self.classifiers),
            self.validation_strategy,
            self.validation_size,
            self.total_budget,
            self.per_run_budget
        ]
        # open file for write
        with open(fpath, 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            if not fexists:
                # define header
                header = [
                    'dataset_name',
                    'dataset_iter',
                    'search_type',
                    'num_classifiers',
                    'validation_strategy',
                    'validation_size',
                    'total_budget',
                    'per_run_budget'
                ]
                # write header
                writer.writerow(header)
            # write data
            writer.writerow(data)

    def save_results(self):
        fname = '{}_{}_{}_{}_{}'.format(
            self.dataset_name,
            str(self.dataset_iter),
            self.search_type,
            self.validation_strategy,
            self.validation_size
        )
        cv_results_dir = os.path.join(self.output_dir, 'cv_results')
        if not os.path.exists(cv_results_dir):
            os.makedirs(cv_results_dir)
        performance_dir = os.path.join(self.output_dir, 'performance')
        if not os.path.exists(performance_dir):
            os.makedirs(performance_dir)
        # cv_results
        self.cv_results.to_csv(os.path.join(
            cv_results_dir, fname+'.csv'), index=False)
        # performance_over_time
        self.performance_over_time.to_csv(os.path.join(
            performance_dir, fname+'.csv'), index=False)


class SMACSearch(Search):

    def __init__(self, dataset_name, dataset_iter, classifiers, validation_strategy, validation_size=200,
                 total_budget=600, per_run_budget=30, output_dir='output', random_state=123):
        # Search type
        self.search_type = 'smac'
        # SMAC object callback is None
        self.smac_object_callback = None
        # Call parent constructor
        super().__init__(dataset_name, dataset_iter, classifiers, validation_strategy, validation_size,
                         total_budget, per_run_budget, output_dir, random_state)


class RandomSearch(Search):

    def __init__(self, dataset_name, dataset_iter, classifiers, validation_strategy, validation_size=200,
                 total_budget=600, per_run_budget=30, output_dir='output', random_state=123):
        # Search type
        self.search_type = 'random'
        # SMAC object callbak for random search
        self.smac_object_callback = get_random_search_object_callback
        # Call parent constructor
        super().__init__(dataset_name, dataset_iter, classifiers, validation_strategy, validation_size,
                         total_budget, per_run_budget, output_dir, random_state)


class UniformExplorationSearch(Search):
    def __init__(self, dataset_name, dataset_iter, classifiers, validation_strategy, validation_size=200,
                 total_budget=600, per_run_budget=30, output_dir='output', random_state=123):
        # Search type
        self.search_type = 'ue'
        # Random permutation of classifiers
        random.shuffle(classifiers)
        # Call parent constructor
        super().__init__(dataset_name, dataset_iter, classifiers, validation_strategy, validation_size,
                         total_budget, per_run_budget, output_dir, random_state)

    def run(self):
        # init
        cv_results_list = []
        performance_over_time_list = []
        # define budget per classifier
        budget = max(self.per_run_budget, int(
            self.total_budget / len(self.classifiers)))
        # run individual searches
        for clf in self.classifiers:
            # define random search object
            rs = RandomSearch(
                self.dataset_name,
                self.dataset_iter,
                [clf],
                self.validation_strategy,
                self.validation_size,
                budget,
                self.per_run_budget,
                self.output_dir,
                self.random_state
            )
            # run
            rs.run()
            # individual results
            cv_results_list.append(
                pd.DataFrame.from_dict(rs.automl.cv_results_))
            performance_over_time_list.append(rs.automl.performance_over_time_)
        # concatenate and save results
        self.cv_results = pd.concat(
            cv_results_list, axis=0, ignore_index=True)
        perf_df = pd.concat(
            performance_over_time_list, axis=0, ignore_index=True)
        # transform to monotonically increasing
        self.performance_over_time = self.transform_monotonic(
            perf_df, ['single_best_optimization_score', 'single_best_test_score'])

    def transform_monotonic(self, df, cols):
        for col in cols:
            for index, row in df.iterrows():
                if index > 0:
                    cur = df.at[index, col]
                    prev = df.at[index-1, col]
                    if cur < prev:
                        df.at[index, col] = df.at[index-1, col]
        return df


def get_random_search_object_callback(
    scenario_dict,
    seed,
    ta,
    ta_kwargs,
    metalearning_configurations,
    n_jobs,
    dask_client
):
    """ Random search """
    if n_jobs > 1 or (dask_client and len(dask_client.nthreads()) > 1):
        raise ValueError("Please make sure to guard the code invoking Auto-sklearn by "
                         "`if __name__ == '__main__'` and remove this exception.")
    # Setup
    scenario_dict['minR'] = len(scenario_dict['instances'])
    scenario_dict['initial_incumbent'] = 'RANDOM'
    scenario = Scenario(scenario_dict)
    return ROAR(
        scenario=scenario,
        rng=seed,
        tae_runner=ta,
        tae_runner_kwargs=ta_kwargs,
        run_id=seed,
        dask_client=dask_client,
        n_jobs=n_jobs,
    )
