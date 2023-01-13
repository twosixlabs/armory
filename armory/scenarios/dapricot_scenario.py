"""
D-APRICOT scenario for object detection in the presence of targeted adversarial patches.
"""

import copy

import numpy as np

from armory.instrument.export import DApricotExporter, ExportMeter
from armory.logs import log
from armory.scenarios.scenario import Scenario


class ObjectDetectionTask(Scenario):
    def __init__(self, *args, skip_benign=None, **kwargs):
        if skip_benign is False:
            log.warning(
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

    def next(self):
        super().next()
        self.y, self.y_patch_metadata = self.y
        self.probe.update(y=self.y, y_patch_metadata=self.y_patch_metadata)

    def load_model(self, defended=True):
        model_config = self.config["model"]
        generate_kwargs = self.config["attack"]["generate_kwargs"]
        if (
            model_config["model_kwargs"].get("batch_size") != 3
            and generate_kwargs["threat_model"].lower() == "physical"
        ):
            log.warning(
                "If using Armory's baseline mscoco frcnn model,"
                " model['model_kwargs']['batch_size'] should be set to 3 for physical attack."
            )
        super().load_model(defended=defended)

    def fit(self, train_split_default="train"):
        raise NotImplementedError(
            "Training has not yet been implemented for object detectors"
        )

    def run_benign(self):
        raise NotImplementedError("D-APRICOT has no benign task")

    def run_attack(self):
        self._check_x("run_attack")
        self.hub.set_context(stage="attack")
        x, y = self.x, self.y

        with self.profiler.measure("Attack"):
            if x.shape[0] != 1:
                raise ValueError("D-APRICOT batch size must be set to 1")
            # (nb=1, num_cameras, h, w, c) --> (num_cameras, h, w, c)
            x = x[0]

            generate_kwargs = copy.deepcopy(self.generate_kwargs)
            generate_kwargs["y_patch_metadata"] = self.y_patch_metadata
            y_target = self.label_targeter.generate(y)
            generate_kwargs["y_object"] = y_target

            x_adv = self.attack.generate(x=x, **generate_kwargs)

        self.hub.set_context(stage="adversarial")
        # Ensure that input sample isn't overwritten by model
        x_adv.flags.writeable = False
        y_pred_adv = self.model.predict(x_adv)
        self.probe.update(y_pred_adv_batch=[y_pred_adv])
        for img_idx in range(len(y)):
            y_i_target = y_target[img_idx]
            y_i_pred = y_pred_adv[img_idx]
            self.probe.update(y_target=[y_i_target], y_pred_adv=[y_i_pred])

        # Add batch dimension (3, H, W, C) --> (1, 3, H, W, C)
        self.probe.update(x_adv=np.expand_dims(x_adv, 0))

        self.x_adv, self.y_target, self.y_pred_adv = x_adv, y_target, y_pred_adv

    def load_export_meters(self):
        # Load default export meters
        super().load_export_meters()

        # Add export meters that export examples with boxes overlaid
        self.sample_exporter_with_boxes = DApricotExporter(
            self.scenario_output_dir,
            default_export_kwargs={"with_boxes": True},
        )
        export_with_boxes_meter = ExportMeter(
            "x_adv_with_boxes_exporter",
            self.sample_exporter_with_boxes,
            "scenario.x_adv",
            y_pred_probe="scenario.y_pred_adv_batch",
            max_batches=self.num_export_batches,
        )
        self.hub.connect_meter(export_with_boxes_meter, use_default_writers=False)

    def _load_sample_exporter(self):
        return DApricotExporter(self.export_dir)
