# CASH optimization for anomaly detection using AutoML
This is the repository for my thesis in the Master of Artificial Intelligence programme at KU Leuven.

## Description
The aim of the thesis is to solve the CASH optimization problem for anomaly detection by experimenting on constrained and biased anomaly detection datasets and compare the performance for several state-of-the-art hyper-parameter optimization frameworks such as:
* Grid Search
* Random Search
* Bayesian Optimization
* Bandit-based algorithms

Several packages exist for solving CASH optimization, however the anomaly detection setting has not been explored thoroughly yet. The goal of this thesis is to extend **Auto-Sklearn** and incorporate anomaly detection algorithms from **PyOD** and evaluate its performance on a number of datasets with varying characteristics, e.g. number of available training points, number of available labels etc.

### Auto-Sklearn
The current implementation can be found [here](https://github.com/automl/auto-sklearn).

### PyOD
The current implementation can be found [here](https://pyod.readthedocs.io/en/latest/index.html).

### Datasets
A commonly used set of datasets in the literature can be found [here](https://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/).

## License
Copyright © 2022 Ioannis Antoniadis
