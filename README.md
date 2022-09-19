# Systematic Evaluation of CASH Search Strategies for Unsupervised Anomaly Detection
Repository for the corresponding full-paper accepted at the [LIDTA-2022 workshop](https://lidta.dcc.fc.up.pt/) of the [ECML/PKDD 2022](https://2022.ecmlpkdd.org/).

**Note**: This repository initially served as the code repo for my thesis in **[Master of Artificial Intelligence](https://wms.cs.kuleuven.be/cs/studeren/master-artificial-intelligence)** programme at [KU Leuven](https://www.kuleuven.be/english/kuleuven) but it was later modified/extended to accommodate the relevant content of the LIDTA-2022 full-paper submission.


## Description
The code provides an experimental evaluation of how the structure of the validation set, i.e., its size and label bias, impacts the performance of different CASH search strategies within the context of anomaly detection.

## Contents
- `data` directory contains a sub-directory of the `original` datasets used in the experiments, while the `processed` sub-directory is created by `src/notebooks/dataset_preprocessor.ipynb` notebook.
- `src` directory contains the core implemntation code comprised of python scripts and notebooks. It also contains the `auto-sklearn` package which is modified to accommodate unsupervised anomaly detection tasks.
- `results` directory contains the raw results of the paper for the different CASH search spaces.

## How to run the code
Provide the experiment parameters in `src/config.json`:
- `datasets`: list of datasets
- `iterations`: list of iterations, i.e. different versions of the train/test splits (1, 2, ..., 10)
- `classifiers`: list of anomaly detectors
- `search_space`: version of search space (sp1, sp2 or default)
- `validation_set_split_strategies`: list of strategies to split the validation set (stratified, balanced)
- `validation_set_sizes`: list of sizes for the validation set (20, 50, 100, 200)
- `total_budget`: total duration of a single search
- `per_run_budget`: minimum duration of a single run
and run `auto_ad_main.py`.

## External links

| Name | Description | Link |
|:------------- |:------------- |:-------------:|
| Auto-Sklearn | Automated machine learning toolkit | [:link:](https://www.coursera.org/account/accomplishments/certificate/RMLFKH4CJZM4) |
| PyOD | Python library for anomaly detection | [:link:](https://pyod.readthedocs.io/en/latest/index.html) |
| Datasets | Anomaly detection datasets | [:link:](https://pyod.readthedocs.io/en/latest/index.html) |

## License
Copyright © 2022 Ioannis Antoniadis
