from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, PREDICTIONS, SPARSE

class LOCIClassifier(AutoSklearnClassificationAlgorithm):

    def __init__(self, k, alpha, random_state=None):
        self.k = k
        self.alpha = alpha
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        from pyod.models.loci import LOCI

        self.estimator = LOCI(k=self.k, alpha=self.k)
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
            'shortname': 'LOCI',
            'name': 'Local Correlation Integral (LOCI)',
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

        alpha = UniformFloatHyperparameter(
            name="alpha", lower=0.0, upper=1.0, default_value=0.5)
        k = UniformIntegerHyperparameter(
            name="k", lower=1, upper=20, default_value=3)
        cs.add_hyperparameters([k, alpha])

        return cs
