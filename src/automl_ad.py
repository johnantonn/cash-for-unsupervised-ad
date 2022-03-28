import os
import time
from search import RandomSearch, SMACSearch, EqualBudgetSearch, \
    BOSHSearch, BOHBSearch
from utils import import_dataset, add_pyod_models_to_pipeline, \
    plot_performance


if __name__ == "__main__":

    # Add models to Auto-Sklearn
    add_pyod_models_to_pipeline()

    # List of datasets
    datasets = {
        # 'cardio_02': '../data/Cardiotocography_withoutdupl_norm_02_v10.arff',
        'cardio_05': '../data/Cardiotocography_withoutdupl_norm_05_v10.arff',
        'cardio_10': '../data/Cardiotocography_withoutdupl_norm_10_v10.arff',
        'cardio_20': '../data/Cardiotocography_withoutdupl_norm_20_v10.arff',
        # 'cardio_22': '../data/Cardiotocography_withoutdupl_norm_22.arff'
    }

    # Max samples
    max_samples = 5000

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

    # Output directory (based on timestamp)
    out_dir = time.strftime("%Y%m%d_%H%M%S")

    # Loop
    for name, filename in datasets.items():

        # Import dataset
        full_path = os.path.join(os.path.dirname(__file__), filename)
        df = import_dataset(full_path)

        # Resampling strategy
        for validation_strategy in ['stratified', 'balanced']:

            # Random proportional search
            d_name = name + '_equal'
            equal = EqualBudgetSearch(
                d_name=d_name,
                df=df,
                classifiers=classifiers,
                validation_strategy=validation_strategy,
                max_samples=max_samples,
                total_budget=total_budget,
                per_run_budget=per_run_budget,
                output_dir=out_dir
            )
            equal.run()
            equal.plot_scores()
            equal.save_results()

            # Random search
            d_name = name + '_random'
            random = RandomSearch(
                d_name=d_name,
                df=df,
                classifiers=classifiers,
                validation_strategy=validation_strategy,
                max_samples=max_samples,
                total_budget=total_budget,
                per_run_budget=per_run_budget,
                output_dir=out_dir
            )
            random.run()
            random.plot_scores()
            random.print_summary()
            random.print_rankings()
            random.save_results()

            # SMAC search
            d_name = name + '_smac'
            smac = SMACSearch(
                d_name=d_name,
                df=df,
                classifiers=classifiers,
                validation_strategy=validation_strategy,
                max_samples=max_samples,
                total_budget=total_budget,
                per_run_budget=per_run_budget,
                output_dir=out_dir
            )
            smac.run()
            smac.plot_scores()
            smac.print_summary()
            smac.print_rankings()
            smac.save_results()

        # Successive Halving (stratified)
        # only stratified applicable
        d_name = name + '_bosh'
        bosh = BOSHSearch(
            d_name=d_name,
            df=df,
            classifiers=classifiers,
            validation_strategy='stratified',
            max_samples=max_samples,
            total_budget=total_budget,
            per_run_budget=per_run_budget,
            output_dir=out_dir
        )
        bosh.run()
        bosh.plot_scores()
        bosh.print_summary()
        bosh.print_rankings()
        bosh.save_results()

        # Hyberband (stratified)
        d_name = name + '_bohb'
        bohb = BOHBSearch(
            d_name=d_name,
            df=df,
            classifiers=classifiers,
            validation_strategy='stratified',
            max_samples=max_samples,
            total_budget=total_budget,
            per_run_budget=per_run_budget,
            output_dir=out_dir
        )
        bohb.run()
        bohb.plot_scores()
        bohb.print_summary()
        bohb.print_rankings()
        bohb.save_results()

    # Plot multi-line performance graph
    plot_performance(out_dir, total_budget)
