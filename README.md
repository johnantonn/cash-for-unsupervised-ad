# Systematic evaluation of CASH optimization for unsupervised anomaly detection
**Thesis** repository for the **[Master of Artificial Intelligence](https://wms.cs.kuleuven.be/cs/studeren/master-artificial-intelligence)** programme at [KU Leuven](https://www.kuleuven.be/english/kuleuven).

![image](https://user-images.githubusercontent.com/8168416/160713379-cc39a1a2-85de-434f-9ee7-4d8ff51838e1.png)

## Description
The aim of the thesis is to evaluate the CASH optimization problem for unsupervised anomaly detection by systematically experimenting on different types and sizes of the validation set and compare the performance for the below search strategies:
* Random Search
* Uniform Exploration
* Bayesian Optimization (SMAC)


## Contributions
Several packages exist for solving CASH optimization, however the unsupervised anomaly detection setting has not been explored thoroughly to date. More specifically, there have been a number of research efforts that focus on specific characteristics of the anomaly detection setting, e.g. time-series data, purely supvervised approaches.

This work instead assumes the presence of only a small labeled validation set that is used for optimization and evaluation purposes, while anomaly detection model training is unsupervised. This thesis contributes to the current state-of-the-art as follows:
 - Systematic experimental evaluation of CASH algorithms in the context of unsupervised anomaly detection by careful testing a number of hypotheses surrounding the structure of the validation set.
 - Extending the core implementation of **Auto-Sklearn** by incorporating unsupervised anomaly detection algorithms from **PyOD** and evaluate their performance.

## External links

### Auto-Sklearn
The current implementation can be found [here](https://github.com/automl/auto-sklearn).

### PyOD
The current implementation can be found [here](https://pyod.readthedocs.io/en/latest/index.html).

### Datasets
A commonly used set of datasets in the literature can be found [here](https://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/).

## License
Copyright © 2022 Ioannis Antoniadis
