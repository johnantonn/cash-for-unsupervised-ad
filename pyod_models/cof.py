from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, CategoricalHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, PREDICTIONS, SPARSE

class COFClassifier(AutoSklearnClassificationAlgorithm):

    def __init__(self, n_neighbors, method, random_state=None):
        self.n_neighbors = n_neighbors
        self.method = method
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        from pyod.models.cof import COF

        self.estimator = COF(n_neighbors=self.n_neighbors, method=self.method)
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

        n_neighbors = UniformIntegerHyperparameter(
            name="n_neighbors", lower=1, upper=200, default_value=1)
        method = CategoricalHyperparameter(
            name="method", 
            choices=["fast", "memory"],
            default_value="fast"
        )
        cs.add_hyperparameters([n_neighbors])

        return cs
