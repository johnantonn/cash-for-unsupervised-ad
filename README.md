# Gamblers galore; a multi-armed bandit strategy for anomaly detection model selection
This is the repository for my thesis in the Master of Artificial Intelligence programme at KU Leuven.

![alt text](https://miro.medium.com/max/875/0*jFV1aQ88ZsXajRgO.png)

## Problem description
Given a dataset that contains anomalies, a set of anomaly detection models and a computational budget, the task is to converge as quickly as possible to the best possible model for the dataset.

### Dataset
The dataset used by this implementation is taken from [here](https://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/).

### Anomaly detection algorithms
The implementation makes use of the [PyOD](https://pyod.readthedocs.io/en/latest/index.html) package.
