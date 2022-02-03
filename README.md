# CASH optimization for anomaly detection using AutoML
This is the repository for my thesis in the Master of Artificial Intelligence programme at KU Leuven.

## Description
The aim of the thesis is to experiment on constrained and biased anomaly detection datasets and compare the performance for several state-of-the-art hyper-parameter optimization frameworks such as:
* Grid Search
* Random Search
* Genetic Algorithms
* Bayesian Optimization
* Bandit-based algorithms

for model and hyper-parameter selection. Several packages exist that provide subset or all of the aforementioned functionality:
* Auto-sklearn
* Hyperopt-Sklearn
* Auto-WEKA
* TPOT
* Automatic Statistician

An additional goal of this thesis is the experimentation with (some of) these packages on the stated problem and the proper visualization/report of the results and conclusions.

### Datasets
A well-known set of datasets for initial experimentation can be found [here](https://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/).

### Anomaly detection algorithms
The implementation makes use of the [PyOD](https://pyod.readthedocs.io/en/latest/index.html) package.

## License
Copyright © 2021 Ioannis Antoniadis
