from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    UniformFloatHyperparameter, Constant

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS


class HBOSClassifier(AutoSklearnClassificationAlgorithm):

    def __init__(self, n_bins, alpha, tol, contamination, random_state=None):
        self.n_bins = n_bins
        self.alpha = alpha
        self.tol = tol
        self.contamination = contamination
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        from pyod.models.hbos import HBOS

        self.estimator = HBOS(
            n_bins=self.n_bins,
            alpha=self.alpha,
            tol=self.tol,
            contamination=self.contamination
        )
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    def decision_function(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.decision_function(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            'shortname': 'HBOS',
            'name': 'Histogram-based Outlier Score',
            'handles_regression': False,
            'handles_classification': True,
            'handles_multiclass': False,
            'handles_multilabel': False,
            'handles_multioutput': False,
            'is_deterministic': True,
            'input': (DENSE, SPARSE, UNSIGNED_DATA),
            'output': (PREDICTIONS,)
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        n_bins = Constant(
            name="n_bins",
            value="auto"
        )
        alpha = UniformFloatHyperparameter(
            name="alpha",
            lower=0.1,
            upper=1.0,
            q=0.1,
            default_value=0.1
        )
        tol = UniformFloatHyperparameter(
            name="tol",
            lower=0.1,
            upper=1.0,
            q=0.1,
            default_value=0.5
        )
        contamination = UniformFloatHyperparameter(
            name="contamination",
            lower=0.01,
            upper=0.5,
            q=0.01,
            default_value=0.1
        )
        cs.add_hyperparameters([n_bins, alpha, tol, contamination])

        return cs
