import os
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
from sklearn.model_selection import train_test_split, PredefinedSplit, \
    StratifiedShuffleSplit
from utils import balanced_split


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

    def __init__(self, d_name, df, algos, max_samples, validation_strategy, total_budget,
                 per_run_budget, out_dir, random_state):
        self.d_name = d_name  # dataset name
        self.df = df  # dataframe
        self.algos = algos  # PyOD algorithms to use
        self.max_samples = max_samples  # max samples
        self.validation_strategy = validation_strategy  # split strategy
        self.total_budget = total_budget  # total budget in seconds
        self.per_run_budget = per_run_budget  # per run budget in seconds
        self.random_state = random_state  # random state for reproducibility
        self.resampling_strategy = StratifiedShuffleSplit(
            n_splits=5, test_size=0.3)  # resampling strategy for optimization
        self.automl = None  # autosklearn classifier
        self.output_dir = os.path.join(os.path.dirname(
            __file__), 'output', out_dir)  # output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def build_automl(self):
        return AutoSklearnClassifier(
            include={
                'classifier': self.algos,
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

    def run(self):
        print('Running search for {}, strategy:{}'.format(
            self.d_name, self.validation_strategy))
        # Subsample if too large
        if(len(self.df) > self.max_samples):
            self.df = self.df.sample(n=self.max_samples)
        # Extract X, y
        X = self.df.iloc[:, :-1]
        y = self.df['outlier']
        # Split to train, test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=self.random_state)
        # resampling strategy
        if self.validation_strategy == 'stratified':
            self.resampling_strategy = StratifiedShuffleSplit(
                n_splits=5, test_size=0.3)
        elif self.validation_strategy == 'balanced':  # based on y_train
            selected_indices = balanced_split(y_train)
            self.resampling_strategy = PredefinedSplit(
                test_fold=selected_indices)
        else:
            raise ValueError('Invalid value `{}` for argument `resampling_strategy`'.format(
                self.validation_strategy))
        # Add NoPreprocessing component to auto-sklearn.
        data_preprocessing.add_preprocessor(
            NoPreprocessing)
        # build automl classifier
        self.automl = self.build_automl()
        # fit
        self.automl.fit(X_train, y_train, X_test,
                        y_test, dataset_name=self.d_name)

    def print_summary(self):
        print(self.automl.sprint_statistics())

    def print_rankings(self):
        results = pd.DataFrame.from_dict(self.automl.cv_results_)
        cols = [
            'rank_test_scores',
            'status',
            'param_classifier:__choice__',
            'mean_test_score',
            'mean_fit_time'
        ]
        cols.extend([key for key in self.automl.cv_results_.keys()
                    if key.startswith('metric_')])
        print(results[cols].sort_values(
            ['rank_test_scores']).to_string(index=False))

    def plot_scores(self):
        title = '{}_{}'.format(self.d_name, self.validation_strategy)
        val_scores = self.automl.performance_over_time_[
            ['Timestamp', 'single_best_optimization_score']]
        test_scores = self.automl.performance_over_time_[
            ['Timestamp', 'single_best_test_score']]
        xval = (val_scores.Timestamp -
                val_scores.Timestamp[0]).apply(td.total_seconds)
        yval = val_scores.single_best_optimization_score
        xtest = (test_scores.Timestamp -
                 test_scores.Timestamp[0]).apply(td.total_seconds)
        ytest = test_scores.single_best_test_score
        # modify for plotting
        xval.at[xval.shape[0]] = self.total_budget
        yval.at[yval.shape[0]] = yval.at[yval.shape[0]-1]
        xtest.at[xtest.shape[0]] = self.total_budget
        ytest.at[ytest.shape[0]] = ytest.at[ytest.shape[0]-1]
        plt.figure()
        plt.plot(xval, yval)
        plt.plot(xtest, ytest)
        plt.legend(['validation', 'test'])
        plt.ylim([0.5, 1.])
        plt.xlabel('seconds')
        plt.ylabel('score')
        plt.title(title)
        plt.grid()
        plt.show()
        plt.savefig(os.path.join(self.output_dir, title+'.png'))

    def store_results(self):
        # filename
        title = '{}_{}'.format(self.d_name, self.validation_strategy)
        # automl.cv_results_
        cv_results_df = pd.DataFrame.from_dict(self.automl.cv_results_)
        cv_results_df.to_csv(os.path.join(
            self.output_dir, title+'_cv_results.csv'), index=False)
        # automl.performance_over_time_
        performance_over_time_df = pd.DataFrame.from_dict(
            self.automl.performance_over_time_)
        performance_over_time_df.to_csv(os.path.join(
            self.output_dir, title+'_performance_over_time.csv'), index=False)


class SMACSearch(Search):

    def __init__(self, d_name, df, algos, validation_strategy, max_samples=5000,
                 total_budget=300, per_run_budget=30, out_dir='output', random_state=123):
        self.smac_object_callback = None
        super().__init__(d_name, df, algos, max_samples, validation_strategy,
                         total_budget, per_run_budget, out_dir, random_state)


class RandomSearch(Search):

    def __init__(self, d_name, df, algos, validation_strategy, max_samples=5000,
                 total_budget=300, per_run_budget=30, out_dir='output', random_state=123):
        self.smac_object_callback = get_random_search_object_callback
        super().__init__(d_name, df, algos, max_samples, validation_strategy,
                         total_budget, per_run_budget, out_dir, random_state)


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
