"""
D-APRICOT scenario for object detection in the presence of targeted adversarial patches.
"""

import copy
import logging

from armory.scenarios.scenario import Scenario
from armory.utils import metrics

logger = logging.getLogger(__name__)


class ObjectDetectionTask(Scenario):
    def __init__(self, *args, skip_benign=None, **kwargs):
        if skip_benign is False:
            logger.warning(
                "--skip-benign=False is being ignored since the D-APRICOT"
                " scenario doesn't include benign evaluation."
            )
        super().__init__(*args, skip_benign=True, **kwargs)
        if self.skip_misclassified:
            raise ValueError(
                "skip_misclassified shouldn't be set for D-APRICOT scenario"
            )
        if self.skip_attack:
            raise ValueError("--skip-attack should not be set for D-APRICOT scenario.")

    def load_attack(self):
        attack_config = self.config["attack"]
        attack_type = attack_config.get("type")
        if not attack_config.get("kwargs").get("targeted", False):
            raise ValueError(
                "attack['kwargs']['targeted'] must be set to True for D-APRICOT scenario"
            )
        elif attack_type == "preloaded":
            raise ValueError(
                "attack['type'] should not be set to 'preloaded' for D-APRICOT scenario "
                "and does not need to be specified."
            )
        elif "targeted_labels" not in attack_config:
            raise ValueError(
                "attack['targeted_labels'] must be specified, as the D-APRICOT"
                " threat model is targeted."
            )
        elif attack_config.get("use_label"):
            raise ValueError(
                "The D-APRICOT scenario threat model is targeted, and"
                " thus attack['use_label'] should be set to false or unspecified."
            )
        generate_kwargs = attack_config.get("generate_kwargs", {})
        if "threat_model" not in generate_kwargs:
            raise ValueError(
                "D-APRICOT scenario requires attack['generate_kwargs']['threat_model'] to be set to"
                " one of ('physical', 'digital')"
            )
        elif generate_kwargs["threat_model"].lower() not in ("physical", "digital"):
            raise ValueError(
                "D-APRICOT scenario requires attack['generate_kwargs']['threat_model'] to be set to"
                f"' one of ('physical', 'digital'), not {generate_kwargs['threat_model']}."
            )
        super().load_attack()

    def load_dataset(self):
        if self.config["dataset"].get("batch_size") != 1:
            raise ValueError(
                "dataset['batch_size'] must be set to 1 for D-APRICOT scenario."
            )
        super().load_dataset()

    def load_model(self, defended=True):
        model_config = self.config["model"]
        generate_kwargs = self.config["attack"]["generate_kwargs"]
        if (
            model_config["model_kwargs"].get("batch_size") != 3
            and generate_kwargs["threat_model"].lower() == "physical"
        ):
            logger.warning(
                "If using Armory's baseline mscoco frcnn model,"
                " model['model_kwargs']['batch_size'] should be set to 3 for physical attack."
            )
        super().load_model(defended=defended)

    def fit(self, train_split_default="train"):
        raise NotImplementedError(
            "Training has not yet been implemented for object detectors"
        )

    def load_metrics(self):
        super().load_metrics()
        # The D-APRICOT scenario has no non-targeted tasks
        self.metrics_logger.adversarial_tasks = []

    def run_benign(self):
        raise NotImplementedError("D-APRICOT has no benign task")

    def run_attack(self):
        x, y = self.x, self.y

        with metrics.resource_context(name="Attack", **self.profiler_kwargs):

            if x.shape[0] != 1:
                raise ValueError("D-APRICOT batch size must be set to 1")
            # (nb=1, num_cameras, h, w, c) --> (num_cameras, h, w, c)
            x = x[0]
            y_object, y_patch_metadata = y

            generate_kwargs = copy.deepcopy(self.generate_kwargs)
            generate_kwargs["y_patch_metadata"] = y_patch_metadata
            y_target = self.label_targeter.generate(y_object)
            generate_kwargs["y_object"] = y_target

            x_adv = self.attack.generate(x=x, **generate_kwargs)

        # Ensure that input sample isn't overwritten by model
        x_adv.flags.writeable = False
        y_pred_adv = self.model.predict(x_adv)
        for img_idx in range(len(y_object)):
            y_i_target = y_target[img_idx]
            y_i_pred = y_pred_adv[img_idx]
            self.metrics_logger.update_task(
                [y_i_target], [y_i_pred], adversarial=True, targeted=True
            )

        self.metrics_logger.update_perturbation(x, x_adv)

        if self.sample_exporter is not None:
            self.sample_exporter.export(x, x_adv, y_object, y_pred_adv)
        self.x_adv, self.y_target, self.y_pred_adv = x_adv, y_target, y_pred_adv

    def finalize_results(self):
        self.metrics_logger.log_task(adversarial=True, targeted=True)
        self.results = self.metrics_logger.results()
