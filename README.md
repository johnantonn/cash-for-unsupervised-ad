# Systematic evaluation of CASH optimization for unsupervised anomaly detection
**Thesis** repository for the **[Master of Artificial Intelligence](https://wms.cs.kuleuven.be/cs/studeren/master-artificial-intelligence)** programme at [KU Leuven](https://www.kuleuven.be/english/kuleuven).

![image](https://user-images.githubusercontent.com/8168416/160713379-cc39a1a2-85de-434f-9ee7-4d8ff51838e1.png)

## Description
The aim of the thesis is to evaluate the CASH optimization problem for anomaly detection by systematically experimenting on different types and sizes of the validation set and compare the performance for the below search strategies:
* Random Search
* Uniform Exploration
* Bayesian Optimization (SMAC)


## Contributions
Several packages exist for solving CASH optimization, however the anomaly detection setting has not been explored thoroughly yet. More specifically, there have been a number of research efforts that focus on specific characteristics of the anomaly detection setting, i.e. target time-series data, purely supvervised or purely unsupervised approaches.

This thesis instead assumes the presence of a limited amount of labels for generating a validation set that can be used for optimization and evaluation purposes, while anomaly detection model training is unsupervised. This thesis contributes to the current state-of-the-art as follows:
 - Systematic experimental evaluation of CASH algorithms in the context of anomaly detection by careful testing a number of hypotheses surrounding the structure of the validation set.
 - Extending the core implementation of **Auto-Sklearn** by incorporating anomaly detection algorithms from **PyOD** and evaluate their performance.

## External links

### Auto-Sklearn
The current implementation can be found [here](https://github.com/automl/auto-sklearn).

### PyOD
The current implementation can be found [here](https://pyod.readthedocs.io/en/latest/index.html).

### Datasets
A commonly used set of datasets in the literature can be found [here](https://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/).

## License
Copyright © 2022 Ioannis Antoniadis
