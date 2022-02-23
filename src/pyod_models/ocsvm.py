from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    UniformFloatHyperparameter, CategoricalHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS

class OCSVMClassifier(AutoSklearnClassificationAlgorithm):

    def __init__(self, kernel, nu, gamma, shrinking, tol, contamination, 
                 degree=3, coef0=1, random_state=None):
        self.kernel = kernel
        self.nu = nu
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.shrinking = shrinking
        self.tol = tol
        self.contamination = contamination
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        from pyod.models.ocsvm import OCSVM

        self.estimator = OCSVM(
            kernel = self.kernel,
            nu = self.nu,
            degree = self.degree,
            coef0 = self.coef0,
            gamma = self.gamma,
            shrinking = self.shrinking,
            tol = self.tol,
            contamination = self.contamination
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
            'shortname': 'OCSVM',
            'name': 'One-Class Support Vector Machines',
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

        kernel = CategoricalHyperparameter(
            name = "kernel",
            choices = ["rbf", "poly", "sigmoid"],
            default_value= "rbf"
        )
        nu = UniformFloatHyperparameter(
            name = "nu",
            lower = 0.01,
            upper = 1.0,
            default_value = 0.5
        )
        degree = UniformIntegerHyperparameter(
            name = "degree",
            lower = 2,
            upper = 5,
            default_value = 3
        )
        coef0 = UniformFloatHyperparameter(
            name = 'coef0',
            lower = 1e-2,
            upper = 1e2,
            default_value = 1,
            log = True
        )
        gamma = UniformFloatHyperparameter(
            name= "gamma",
            lower = 3.0517578125e-05,
            upper = 8,
            default_value = 0.1,
            log = True
        )
        shrinking = CategoricalHyperparameter(
            name = "shrinking",
            choices = [True, False],
            default_value = True
        )
        tol = UniformFloatHyperparameter(
            name = "tol",
            lower = 1e-5,
            upper = 1e-1,
            default_value = 1e-3,
            log = True
        )
        contamination = UniformFloatHyperparameter(
            name = "contamination", 
            lower = 0.0,
            upper = 0.5,
            default_value = 0.1
        )
        cs.add_hyperparameters([nu, kernel, degree, coef0, gamma, shrinking, tol, 
                                contamination])
        degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
        coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
        cs.add_condition(degree_depends_on_poly)
        cs.add_condition(coef0_condition)

        return cs
