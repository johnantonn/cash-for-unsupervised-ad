import os
from search import SMACSearch, RandomSearch
from utils import import_dataset, add_pyod_models_to_pipeline


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

    # Algorithms to use
    algos = [
        'CBLOFClassifier',
        'COPODClassifier',
        'IForestClassifier',
        'KNNClassifier',
        'LOFClassifier',
    ]

    # Loop
    for d_name, filename in datasets.items():

        # Import dataset
        full_path = os.path.join(os.path.dirname(__file__), filename)
        df = import_dataset(full_path)

        # Resampling strategy
        for validation_strategy in ['stratified', 'balanced']:

            # Random search
            random = RandomSearch(d_name+'_random', df, algos,
                                  validation_strategy, total_budget=300)
            random.run()
            random.plot_scores()
            random.print_summary()
            random.print_rankings()
            random.store_results()

            # SMAC search
            smac = SMACSearch(d_name+'_smac', df, algos,
                              validation_strategy, total_budget=300)
            smac.run()
            smac.plot_scores()
            smac.print_summary()
            smac.print_rankings()
            smac.store_results()
