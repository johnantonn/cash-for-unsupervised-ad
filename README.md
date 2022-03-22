# CASH optimization for anomaly detection using AutoML
This is the repository for my thesis in the Master of Artificial Intelligence programme at KU Leuven.

## Description
The aim of the thesis is to solve the CASH optimization problem for anomaly detection using AutoML by experimenting on constrained and biased validation sets and compare the performance for several state-of-the-art frameworks such as:
* Random Search
* Random Proportional Search
* Bayesian Optimization (SMAC)
* Bandit-based algorithms e.g. Successive Halving, Hyperband

Several packages exist for solving CASH optimization, however the anomaly detection setting has not been explored thoroughly. An additional goal of this thesis is to extend **Auto-Sklearn** by incorporating anomaly detection algorithms from **PyOD** and evaluate their performance on a number of datasets with varying characteristics, e.g. number of available training points, number of available labels for validation/optimization etc.

### Auto-Sklearn
The current implementation can be found [here](https://github.com/automl/auto-sklearn).

### PyOD
The current implementation can be found [here](https://pyod.readthedocs.io/en/latest/index.html).

### Datasets
A commonly used set of datasets in the literature can be found [here](https://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/).

## License
Copyright © 2022 Ioannis Antoniadis
