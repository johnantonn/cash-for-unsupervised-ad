import time
from search import RandomSearch, SMACSearch, EquallyDistributedBudgetSearch
from utils import add_pyod_models_to_pipeline


if __name__ == "__main__":

    # Add models to Auto-Sklearn
    add_pyod_models_to_pipeline()

    # List of datasets
    datasets = [
        'Annthyroid',
        'Cardiotocography'
    ]

    # PyOD algorithms to use
    classifiers = [
        'CBLOFClassifier',
        'COPODClassifier',
        'IForestClassifier',
        'KNNClassifier',
        'LOFClassifier',
    ]

    # Budget estimation
    # TODO

    # Budget constraints
    # TODO: should be based estimated budget
    total_budget = 300
    per_run_budget = 30

    # Validation set size
    validation_size = 400

    # Timestamp string
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Loop
    for dataset in datasets:

        # Resampling strategy
        for validation_strategy in ['stratified', 'balanced']:

            # Equally distributed budget search
            edb_search = EquallyDistributedBudgetSearch(
                dataset_name=dataset,
                classifiers=classifiers,
                validation_strategy=validation_strategy,
                validation_size=validation_size,
                total_budget=total_budget,
                per_run_budget=per_run_budget,
                output_dir=timestamp
            )
            edb_search.run()
            edb_search.plot_scores()
            edb_search.save_results()

            # Random search
            random_search = RandomSearch(
                dataset_name=dataset,
                classifiers=classifiers,
                validation_strategy=validation_strategy,
                total_budget=total_budget,
                validation_size=validation_size,
                per_run_budget=per_run_budget,
                output_dir=timestamp
            )
            random_search.run()
            random_search.plot_scores()
            random_search.print_summary()
            random_search.print_rankings()
            random_search.save_results()

            # SMAC search
            smac_search = SMACSearch(
                dataset_name=dataset,
                classifiers=classifiers,
                validation_strategy=validation_strategy,
                validation_size=validation_size,
                total_budget=total_budget,
                per_run_budget=per_run_budget,
                output_dir=timestamp
            )
            smac_search.run()
            smac_search.plot_scores()
            smac_search.print_summary()
            smac_search.print_rankings()
            smac_search.save_results()
