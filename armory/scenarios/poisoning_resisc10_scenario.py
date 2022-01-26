"""
Classifier evaluation within ARMORY

Scenario Contributor: MITRE Corporation
"""
import logging

from armory.scenarios.poison import Poison
from armory.utils import new_poisoning_metrics

logger = logging.getLogger(__name__)

class RESISC10(Poison):
    """
    Dirty label poisoning on resisc10 dataset

    NOTE: "validation" is a split for resisc10 that is currently unused
    """

    def load_explanatory_model(self):
        from importlib import import_module
        from armory.data.utils import maybe_download_weights_from_s3

        model_config = self.config["explanatory_model"]
        model_module = import_module(model_config["module"])
        model_fn = getattr(model_module, model_config["name"])
        weights_file = model_config.get("weights_file", None)
        if isinstance(weights_file, str):
            weights_path = maybe_download_weights_from_s3(
                weights_file, auto_expand_tars=True
            )
        elif isinstance(weights_file, list):
            weights_path = [
                maybe_download_weights_from_s3(w, auto_expand_tars=True)
                for w in weights_file
            ]
        elif isinstance(weights_file, dict):
            weights_path = {
                k: maybe_download_weights_from_s3(v) for k, v in weights_file.items()
            }
        else:
            weights_path = None

        model = model_fn(
            weights_path, model_config["model_kwargs"]
        )
        if not weights_file and not model_config["fit"]:
            logger.warning(
                "No weights file was provided and the model is not configured to train. "
                "Are you loading model weights from an online repository?"
            )

        self.explanatory_model = model

    def get_typicality_stats_dataset(self):
        # Get the dataset level statistics      
        train_data = []
        for xt, yt in zip(self.x_clean, self.y_clean):
            train_data.append((xt,yt))

        (
            self.train_typicality_dist,
            self.class_typicality_match_stats,
            self.class_typicality_mismatch_stats,
            self.mean_activations,
            self.std_activations,
        ) = new_poisoning_metrics.get_data_level_stats(self.explanatory_model, train_data, resize=224)
        logger.info("Calculated dataset-level typicality statistics")

    def get_typicality_stats_examples(self, x, y):
        # Calculate typicality scores and majority/minority labels for 
        # x and associated class id y
        scored_data = []
        for xt, yt in zip(x,y):
            scored_data.append((xt,yt))

        # Get statistics, including majority/minority, of the data to be scored
        typicality_output, majority_minority_output = new_poisoning_metrics.get_per_example_stats(
            self.explanatory_model,
            scored_data,
            self.mean_activations,
            self.std_activations,
            self.class_typicality_match_stats,
            self.class_typicality_mismatch_stats,
            resize=224
        )
        logger.info("Calculated example-level typicality statistics")

        return typicality_output, majority_minority_output

    def load(self):
        self.set_random_seed()
        self.set_dataset_kwargs()
        self.load_model()
        self.load_explanatory_model()
        self.load_train_dataset()
        self.get_typicality_stats_dataset()
        self.load_poisoner()
        self.load_metrics()
        self.poison_dataset()
        self.filter_dataset()
        #self.get_typicality_stats_examples(x,y) # get stats on filtered data x and labels y
        self.fit()
        self.load_dataset()

    def run_attack(self):
        x, y = self.x, self.y
        source = self.source

        x_adv, _ = self.test_poisoner.poison_dataset(x, y, fraction=1.0)
        x_adv.flags.writeable = False
        y_pred_adv = self.model.predict(x_adv, **self.predict_kwargs)

        self.poisoned_test_metric.add_results(y, y_pred_adv)
        # NOTE: uses source->target trigger
        if source.any():
            self.poisoned_targeted_test_metric.add_results(
                [self.target_class] * source.sum(), y_pred_adv[source]
            )

            ##################
            # Test for typicality and sihouette metrics. Remove when done.            
            # Test typicality
            logger.info('Test for typicality metric')
            typicality_output_clean, majority_minority_output_clean = self.get_typicality_stats_examples(
                x[source],
                y[source]
            )
            typicality_output_mislabeled, majority_minority_output_mislabeled = self.get_typicality_stats_examples(
                x[source],
                [self.target_class] * source.sum()
            )
            typicality_output_poisoned, majority_minority_output_poisoned = self.get_typicality_stats_examples(
                x_adv[source],
                [self.target_class] * source.sum()
            )

            for i in range(source.sum()):
                print('Typicality clean/mislabeled/poisoned: {:2.4}/{:2.4}/{:2.4}'.format(
                    typicality_output_clean[i],
                    typicality_output_mislabeled[i],
                    typicality_output_poisoned[i]
                    )
                )

            LABELS = [
                "airplane",
                "airport",
                "harbor",
                "industrial_area",
                "railway",
                "railway_station",
                "runway",
                "ship",
                "storage_tank",
                "thermal_power_station",
            ]
            import matplotlib.pyplot as plt
            plt.figure(1)
            plt.hist(typicality_output_clean, density=True, alpha=0.5, label='Clean')
            plt.hist(typicality_output_mislabeled, density=True, alpha=0.5, label='Mislabeled Clean')
            plt.hist(typicality_output_poisoned, density=True, alpha=0.5, label='Poisoned')
            plt.legend()
            plt.title('RESISC-10 DLBD: source: {}, target: {}'.format(
                LABELS[self.source_class],
                LABELS[self.target_class]
                )
            )
            plt.savefig('resisc10_dlbd_source_{}_target_{}.png'.format(self.source_class, self.target_class))
            ##################

        self.x_adv = x_adv
        self.y_pred_adv = y_pred_adv
