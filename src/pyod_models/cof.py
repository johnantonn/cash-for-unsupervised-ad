from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    UniformFloatHyperparameter, CategoricalHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS

class COFClassifier(AutoSklearnClassificationAlgorithm):

    def __init__(self, contamination, n_neighbors, method, random_state=None):
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.method = method
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        from pyod.models.cof import COF

        self.estimator = COF(
            contamination = self.contamination,
            n_neighbors = self.n_neighbors,
            method = self.method
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
            'shortname': 'COF',
            'name': 'Connectivity-Based Outlier Factor',
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

        contamination = UniformFloatHyperparameter(
            name = "contamination",
            lower = 0.0,
            upper = 0.5,
            default_value = 0.1
        )
        n_neighbors = UniformIntegerHyperparameter(
            name = "n_neighbors",
            lower = 1,
            upper = 100, # ad-hoc
            default_value = 20
        )
        method = CategoricalHyperparameter(
            name = "method", 
            choices = ["fast", "memory"],
            default_value = "fast"
        )
        cs.add_hyperparameters([contamination, n_neighbors, method])

        return cs
