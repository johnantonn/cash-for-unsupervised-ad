from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import Constant

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS


class COPODClassifier(AutoSklearnClassificationAlgorithm):

    def __init__(self, contamination, random_state=None):
        self.contamination = contamination
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        from pyod.models.copod import COPOD

        self.estimator = COPOD(
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
            'shortname': 'COPOD',
            'name': 'Copula Based Outlier Detector',
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

        contamination = Constant(
            name="contamination",
            value=0.1
        )
        cs.add_hyperparameters([contamination])

        return cs
