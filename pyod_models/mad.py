from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS

class MADClassifier(AutoSklearnClassificationAlgorithm):

    def __init__(self, threshold, random_state=None):
        self.threshold = threshold
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        from pyod.models.mad import MAD

        self.estimator = MAD(
            threshold = self.threshold
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
            'shortname': 'MAD',
            'name': 'Median Absolute Deviation (MAD)',
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

        threshold = UniformFloatHyperparameter(
            name = "threshold", 
            lower = 0.0, # ad-hoc
            upper = 100.0, # ad-hoc
            default_value = 3.5
        )
        cs.add_hyperparameters([threshold])
        return cs
