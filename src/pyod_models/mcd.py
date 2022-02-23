from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS

class MCDClassifier(AutoSklearnClassificationAlgorithm):

    def __init__(self, contamination, assume_centered, support_fraction, random_state=None):
        self.contamination = contamination
        self.assume_centered = assume_centered
        self.support_fraction = support_fraction
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        from pyod.models.mcd import MCD

        self.estimator = MCD(
            contamination = self.contamination,
            assume_centered = self.assume_centered,
            support_fraction = self.support_fraction
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
            'shortname': 'MCD',
            'name': 'Outlier Detection with Minimum Covariance Determinant (MCD)',
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
        assume_centered = CategoricalHyperparameter(
            name = "assume_centered", 
            choices = [True, False],
            default_value = False
        )
        support_fraction = UniformFloatHyperparameter(
            name = "support_fraction", 
            lower = 0.0,
            upper = 1.0,
            default_value = 0.5 # ad-hoc
        )
        cs.add_hyperparameters([contamination, assume_centered, support_fraction])
        
        return cs
