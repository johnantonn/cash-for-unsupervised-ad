from search import RandomSearch, SMACSearch, EquallyDistributedBudgetSearch
from utils import add_to_autosklearn_pipeline, get_validation_set_size, \
    get_validation_strategy, load_config_values

if __name__ == "__main__":

    # Load experiment parameters
    datasets, dataset_iter, \
        classifiers, total_budget, per_run_budget, \
        v_strategy_param, v_size_param, \
        output_dir = load_config_values()

    # Add classifiers to Auto-Sklearn
    add_to_autosklearn_pipeline(classifiers)

    # Loop over datsets
    for dataset in datasets:
        # Loop over validation strategy
        for validation_strategy in get_validation_strategy(v_strategy_param):
            # Loop over validation set size values
            for validation_size in get_validation_set_size(dataset, dataset_iter, v_size_param):

                # Equally distributed budget search
                edb_search = EquallyDistributedBudgetSearch(
                    dataset_name=dataset,
                    dataset_iter=dataset_iter,
                    classifiers=classifiers,
                    validation_strategy=validation_strategy,
                    validation_size=validation_size,
                    total_budget=total_budget,
                    per_run_budget=per_run_budget,
                    output_dir=output_dir
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
                    dataset_iter=dataset_iter,
                    classifiers=classifiers,
                    validation_strategy=validation_strategy,
                    total_budget=total_budget,
                    validation_size=validation_size,
                    per_run_budget=per_run_budget,
                    output_dir=output_dir
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
                    dataset_iter=dataset_iter,
                    classifiers=classifiers,
                    validation_strategy=validation_strategy,
                    validation_size=validation_size,
                    total_budget=total_budget,
                    per_run_budget=per_run_budget,
                    output_dir=output_dir
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
