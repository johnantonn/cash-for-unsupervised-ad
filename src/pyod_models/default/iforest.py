from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import Constant

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS


class IForestClassifier(AutoSklearnClassificationAlgorithm):

    def __init__(self, n_estimators, max_samples, max_features, contamination,
                 bootstrap, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.contamination = contamination
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        from pyod.models.iforest import IForest

        self.estimator = IForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            contamination=self.contamination,
            bootstrap=self.bootstrap
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
            'shortname': 'IForest',
            'name': 'Isolation Forest',
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

        n_estimators = Constant(
            name="n_estimators",
            value=100
        )
        max_samples = Constant(
            name="max_samples",
            value=1.0
        )
        max_features = Constant(
            name="max_features",
            value=1.0,
        )
        contamination = Constant(
            name="contamination",
            value=0.1
        )
        bootstrap = Constant(
            name="bootstrap",
            value="False"
        )
        cs.add_hyperparameters(
            [n_estimators, max_samples, max_features, contamination, bootstrap])

        return cs
