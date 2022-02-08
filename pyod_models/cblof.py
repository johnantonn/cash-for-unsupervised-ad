from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, PREDICTIONS, SPARSE

class CBLOFClassifier(AutoSklearnClassificationAlgorithm):

    def __init__(self, n_clusters, alpha, beta, random_state=None):
        self.n_neighbors = n_clusters
        self.alpha = alpha
        self.beta = beta
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        from pyod.models.cblof import CBLOF

        self.estimator = CBLOF(n_clusters=self.n_clusters, alpha=self.alpha, beta=self.beta)
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
            'shortname': 'CBLOF',
            'name': 'Clustering-Based Local Outlier Factor',
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

        n_clusters  = UniformIntegerHyperparameter(
            name="n_clusters", lower=2, upper=20, default_value=8)
        alpha  = UniformFloatHyperparameter(
            name="alpha", lower=0.5, upper=1, default_value=0.5)
        beta = UniformIntegerHyperparameter(
            name="beta", lower=1, upper=20, default_value=5)
        cs.add_hyperparameters([n_clusters, alpha, beta])

        return cs
