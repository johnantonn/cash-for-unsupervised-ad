from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, PREDICTIONS, SPARSE

class LMDDClassifier(AutoSklearnClassificationAlgorithm):

    def __init__(self, dis_measure, n_iter, random_state=None):
        self.dis_measure = dis_measure
        self.n_iter = n_iter 
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        from pyod.models.lmdd import LMDD

        self.estimator = LMDD(dis_measure = self.dis_measure, n_iter = self.n_iter )
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

        dis_measure = CategoricalHyperparameter(
            name="dis_measure",
            choices=["aad", "var", "iqr"],
            default_value="aad"
        )
        n_iter = UniformIntegerHyperparameter(
            name="n_iter",
            lower=10,
            upper=200,
            default_value=50
        )
                                     
        cs.add_hyperparameters([dis_measure, n_iter])

        return cs
