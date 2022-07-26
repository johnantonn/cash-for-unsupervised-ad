from search import RandomSearch, SMACSearch, UniformExplorationSearch
from utils import add_to_autosklearn_pipeline, load_config_values

if __name__ == "__main__":

    # Load experiment parameters
    datasets, iterations, classifiers, search_space, \
        validation_set_split_strategies, validation_set_sizes, \
        total_budget, per_run_budget, output_dir = load_config_values()

    # Add classifiers to Auto-Sklearn
    add_to_autosklearn_pipeline(classifiers, search_space)

    # Loop over datsets
    for dataset in datasets:
        # Loop over iterations
        for dataset_iter in iterations:
            # Loop over validation strategy
            for validation_strategy in validation_set_split_strategies:
                # Loop over validation set size values
                for validation_size in validation_set_sizes:

                    # Uniform exploration
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

                    # SMAC
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
