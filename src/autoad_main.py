from search import RandomSearch, SMACSearch, UniformExplorationSearch
from utils import add_to_autosklearn_pipeline, get_validation_size_list, \
    get_validation_strategy_list, load_config_values

if __name__ == "__main__":

    # Load experiment parameters
    datasets, dataset_iter, \
        classifiers, total_budget, per_run_budget, \
        v_strategy_default_flag, v_size_default_flag, \
        output_dir = load_config_values()

    # Add classifiers to Auto-Sklearn
    add_to_autosklearn_pipeline(classifiers)

    # Loop over datsets
    for dataset in datasets:
        # Loop over validation strategy
        for validation_strategy in get_validation_strategy_list(v_strategy_default_flag):
            # Loop over validation set size values
            for validation_size in get_validation_size_list(dataset, dataset_iter, v_size_default_flag):

                # Equally distributed budget search
                ue_search = UniformExplorationSearch(
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
                    ue_search.run()
                except RuntimeError as err:
                    print(err)
                    continue
                ue_search.plot_scores()
                ue_search.save_results()

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
