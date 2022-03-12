from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    UniformFloatHyperparameter, CategoricalHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS

class PCAClassifier(AutoSklearnClassificationAlgorithm):

    def __init__(self, n_components, contamination, whiten, svd_solver, 
                 weighted, random_state=None):
        self.n_components = n_components
        self.contamination = contamination
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.weighted = weighted
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        from pyod.models.pca import PCA

        self.estimator = PCA(
            n_components = self.n_components,
            contamination = self.contamination,
            whiten = self.whiten,
            svd_solver = self.svd_solver,
            weighted = self.weighted
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
        
        n_components = UniformIntegerHyperparameter(
            name = "n_components",
            lower = 1,
            upper = 20, # ad-hoc
            default_value = 5 # ad-hoc
        )
        contamination = UniformFloatHyperparameter(
            name = "contamination", 
            lower = 0.01,
            upper = 0.5,
            q = 0.01,
            default_value = 0.1
        )
        whiten = CategoricalHyperparameter(
            name = "whiten", 
            choices = [True, False],
            default_value = False
        )
        svd_solver = CategoricalHyperparameter(
            name = "svd_solver", 
            choices = ['auto', 'full', 'arpack', 'randomized'],
            default_value = 'auto'
        )
        weighted = CategoricalHyperparameter(
            name = "weighted", 
            choices = [True, False],
            default_value = True
        )
        cs.add_hyperparameters([n_components, contamination, whiten, svd_solver, weighted])
        
        return cs
