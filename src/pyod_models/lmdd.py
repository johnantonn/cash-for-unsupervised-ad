from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    UniformFloatHyperparameter, CategoricalHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS

class LMDDClassifier(AutoSklearnClassificationAlgorithm):

    def __init__(self, contamination, n_iter, dis_measure, random_state=None):
        self.contamination = contamination
        self.n_iter = n_iter
        self.dis_measure = dis_measure
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        from pyod.models.lmdd import LMDD

        self.estimator = LMDD(
            contamination = self.contamination,
            n_iter = self.n_iter,
            dis_measure = self.dis_measure, 
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
            'shortname': 'LMDD',
            'name': 'Deviation-based Outlier Detection (LMDD)',
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
            lower = 0.01,
            upper = 0.5,
            q = 0.01,
            default_value = 0.1
        )
        n_iter = UniformIntegerHyperparameter(
            name = "n_iter",
            lower = 5,
            upper = 200, # ad-hoc
            q = 5,
            default_value = 50
        )
        dis_measure = CategoricalHyperparameter(
            name = "dis_measure",
            choices = ["aad", "var", "iqr"],
            default_value = "aad"
        )
          
        cs.add_hyperparameters([contamination, n_iter, dis_measure])

        return cs
