from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, PREDICTIONS, SPARSE

class PCAClassifier(AutoSklearnClassificationAlgorithm):

    def __init__(self, n_components, random_state=None):
        self.random_state = random_state
        self.n_components = n_components
        self.estimator = None

    def fit(self, X, Y):
        from pyod.models.pca import PCA

        self.estimator = PCA(n_components = self.n_components)
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
            'shortname': 'PCA',
            'name': 'Principal Component Analysis',
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
        
        # upper ?
        n_components = UniformIntegerHyperparameter(
            name="n_components", lower=1, upper=10, default_value=2)
        cs.add_hyperparameters([n_components])

        return cs
