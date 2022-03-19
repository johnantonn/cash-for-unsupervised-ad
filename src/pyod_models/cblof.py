from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    UniformFloatHyperparameter, Constant

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS

class CBLOFClassifier(AutoSklearnClassificationAlgorithm):

    def __init__(self, n_clusters, contamination, alpha, beta, 
                 use_weights, random_state=None):
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.alpha = alpha
        self.beta = beta
        self.use_weights = use_weights
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        from pyod.models.cblof import CBLOF

        self.estimator = CBLOF(
            n_clusters = self.n_clusters,
            contamination = self.contamination,
            alpha = self.alpha,
            beta = self.beta,
            use_weights = self.use_weights
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
            name = "n_clusters",
            lower = 2,
            upper = 16, # ad-hoc
            q = 2,
            default_value = 8
        )
        contamination = UniformFloatHyperparameter(
            name = "contamination",
            lower = 0.01,
            upper = 0.5,
            q = 0.01,
            default_value = 0.1
        )
        alpha = UniformFloatHyperparameter(
            name = "alpha",
            lower = 0.5,
            upper = 1.0,
            q = 0.1,
            default_value = 0.9
        )
        beta = UniformIntegerHyperparameter(
            name = "beta",
            lower = 2,
            upper = 10, # ad-hoc
            default_value = 5
        )
        use_weights = Constant(
            name = "use_weights",
            value = "False"
        )
        cs.add_hyperparameters([n_clusters, contamination, alpha, beta, use_weights])

        return cs
