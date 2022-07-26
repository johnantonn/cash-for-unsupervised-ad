from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS


class SOSClassifier(AutoSklearnClassificationAlgorithm):

    def __init__(self, contamination, perplexity, eps, random_state=None):
        self.contamination = contamination
        self.perplexity = perplexity
        self.eps = eps
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        from pyod.models.sos import SOS

        self.estimator = SOS(
            contamination=self.contamination,
            perplexity=self.perplexity,
            eps=self.eps
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
            'shortname': 'SOS',
            'name': 'Stochastic Outlier Selection',
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
            name="contamination",
            lower=0.05,
            upper=0.5,
            q=0.05,
            default_value=0.1
        )
        perplexity = UniformFloatHyperparameter(
            name="perplexity",
            lower=1.0,
            upper=100.0,
            q=1.0,
            default_value=4.5
        )
        eps = UniformFloatHyperparameter(
            name="eps",
            lower=1e-7,
            upper=1e-2,
            default_value=1e-5,
            log=True
        )
        cs.add_hyperparameters([contamination, perplexity, eps])

        return cs
