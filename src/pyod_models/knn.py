from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    UniformFloatHyperparameter, Constant

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS


class KNNClassifier(AutoSklearnClassificationAlgorithm):

    def __init__(self, n_neighbors, method, p, contamination, random_state=None):
        self.n_neighbors = n_neighbors
        self.method = method
        self.p = p
        self.contamination = contamination
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        from pyod.models.knn import KNN

        self.estimator = KNN(
            n_neighbors=self.n_neighbors,
            method=self.method,
            p=self.p,
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
            'shortname': 'KNN',
            'name': 'K Nearest Neighbors',
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
            name="n_neighbors",
            lower=1,
            upper=100,  # ad-hoc
            default_value=5
        )
        method = Constant(
            name="method",
            value='largest'
        )
        # order of minkowski distance metric (used by default)
        p = Constant(
            name="p",
            value=2  # euclidean
        )
        contamination = UniformFloatHyperparameter(
            name="contamination",
            lower=0.01,
            upper=0.5,
            q=0.01,
            default_value=0.1
        )
        cs.add_hyperparameters([n_neighbors, method, p, contamination])

        return cs
