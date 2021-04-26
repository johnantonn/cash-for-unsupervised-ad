import numpy as np
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
from pyod.models.abod import ABOD
from pyod.models.sos import SOS

# Models dictionary
models = {
    "knn": {
        "instance": KNN(),
        "hyperparameters": {
            "contamination": np.arange(0.01, 0.3, 0.01),
            "n_neighbors": np.arange(1, 100, 1),
            "method": ['largest', 'mean', 'median'],
        }
    },
    "lof": {
        "instance": LOF(),
        "hyperparameters": {
            "contamination": np.arange(0.01, 0.3, 0.01),
            "n_neighbors": np.arange(1, 50, 1),
        }
    },
    "cblof": {
        "instance": CBLOF(),
        "hyperparameters": {
            "contamination": np.arange(0.01, 0.3, 0.01),
            "n_clusters": np.arange(2, 15, 1),
            "alpha": np.arange(0.05, 0.45, 0.05),
            "beta": np.arange(2, 20, 1)
        }
    },
    "abod": {
        "instance": ABOD(),
        "hyperparameters":{
            "contamination": np.arange(0.01, 0.3, 0.01),
            "n_neighbors": np.arange(1, 30, 1),
            "method": ["fast"]
        }
    },
    "sos": {
        "instance": SOS(),
        "hyperparameters": {
            "contamination": np.arange(0.01, 0.3, 0.01),
            "perplexity": np.arange(1, 100, 1)
        }
    }
}