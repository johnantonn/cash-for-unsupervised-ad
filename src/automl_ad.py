import time
from search import RandomSearch, SMACSearch, EquallyDistributedBudgetSearch
from utils import add_pyod_models_to_pipeline, get_validation_set_size, \
    get_validation_strategy

if __name__ == "__main__":

    # Add models to Auto-Sklearn
    add_pyod_models_to_pipeline()

    # List of datasets
    datasets = [
        'ALOI',
        'Annthyroid',
        'Cardiotocography',
        'PageBlocks',
        'SpamBase',
        'Waveform'
    ]

    # PyOD algorithms to use
    classifiers = [
        'CBLOFClassifier',
        'COPODClassifier',
        'IForestClassifier',
        'KNNClassifier',
        'LOFClassifier',
    ]

    # Budget constraints
    # TODO: should be based estimated budget
    total_budget = 600
    per_run_budget = 30

    # Dataset iteration number
    iter = 1

    # Current timestamp string
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Loop over datsets
    for dataset in datasets:
        # Loop over validation strategy
        for validation_strategy in get_validation_strategy():
            # Loop over validation set size values
            for validation_size in get_validation_set_size(dataset, iter):

                # Equally distributed budget search
                edb_search = EquallyDistributedBudgetSearch(
                    dataset_name=dataset,
                    iter=iter,
                    classifiers=classifiers,
                    validation_strategy=validation_strategy,
                    validation_size=validation_size,
                    total_budget=total_budget,
                    per_run_budget=per_run_budget,
                    output_dir=timestamp
                )
                try:
                    edb_search.run()
                except RuntimeError as err:
                    print(err)
                    continue
                edb_search.plot_scores()
                edb_search.save_results()

                # Random search
                random_search = RandomSearch(
                    dataset_name=dataset,
                    iter=iter,
                    classifiers=classifiers,
                    validation_strategy=validation_strategy,
                    total_budget=total_budget,
                    validation_size=validation_size,
                    per_run_budget=per_run_budget,
                    output_dir=timestamp
                )
                try:
                    random_search.run()
                except RuntimeError as err:
                    print(err)
                    continue
                random_search.plot_scores()
                random_search.print_summary()
                # random_search.print_rankings()
                random_search.save_results()

                # SMAC search
                smac_search = SMACSearch(
                    dataset_name=dataset,
                    iter=iter,
                    classifiers=classifiers,
                    validation_strategy=validation_strategy,
                    validation_size=validation_size,
                    total_budget=total_budget,
                    per_run_budget=per_run_budget,
                    output_dir=timestamp
                )
                try:
                    smac_search.run()
                except RuntimeError as err:
                    print(err)
                    continue
                smac_search.plot_scores()
                smac_search.print_summary()
                # smac_search.print_rankings()
                smac_search.save_results()
