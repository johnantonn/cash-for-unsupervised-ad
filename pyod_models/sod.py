from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    UniformFloatHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS

class SODClassifier(AutoSklearnClassificationAlgorithm):

    def __init__(self, n_neighbors, alpha, contamination, 
                 random_state=None):
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.contamination = contamination
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        from pyod.models.sod import SOD

        self.estimator = SOD(
            n_neighbors = self.n_neighbors,
            alpha = self.alpha,
            contamination = self.contamination
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

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            'shortname': 'SOD',
            'name': 'Subspace Outlier Detection',
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
            name = "n_neighbors",
            lower = 1,
            upper = 200,
            default_value = 20
        )
        alpha = UniformFloatHyperparameter(
            name = "alpha",
            lower = 0.0,
            upper = 1.0,
            default_value = 0.8
        )
        contamination = UniformFloatHyperparameter(
            name = "contamination", 
            lower = 0.0,
            upper = 0.5,
            default_value = 0.1
        )
        cs.add_hyperparameters([n_neighbors, alpha, contamination])

        return cs
