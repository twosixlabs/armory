"""
CARLA Multi-Object Tracking Scenario

"""

from armory.instrument import GlobalMeter
from armory.instrument.config import ResultsLogWriter
from armory.logs import log
from armory.metrics.task import GlobalHOTA
from armory.scenarios.carla_video_tracking import CarlaVideoTracking


class CarlaMOT(CarlaVideoTracking):
    def __init__(
        self, config, tracked_classes=("pedestrian",), coco_format=False, **kwargs
    ):
        self.tracked_classes = list(tracked_classes)
        self.coco_format = coco_format
        if self.coco_format and not config["dataset"].get("coco_format"):
            log.warning(
                "Overriding dataset kwarg coco_format to True, mirroring scenario config"
            )
            config["dataset"]["coco_format"] = True

        super().__init__(config, **kwargs)

    def load_metrics(self):
        all_tasks = self.config["metric"]["task"]
        if all_tasks is None:
            super().load_metrics()
            return

        hotas = ["hota", "deta", "assa"]
        self.hota_tasks = [t for t in all_tasks if t in hotas]
        self.config["metric"]["task"] = [t for t in all_tasks if t not in hotas]
        super().load_metrics()
        self.config["metric"]["task"] = all_tasks  # revert to original
        means = self.config["metric"].get("means", True)
        record_metric_per_sample = self.config["metric"].get(
            "record_metric_per_sample", True
        )

        if self.hota_tasks:
            if not self.skip_benign:
                meter = GlobalMeter(
                    "benign_hota_metrics",
                    GlobalHOTA(
                        metrics=self.hota_tasks,
                        coco_format=self.coco_format,
                        tracked_classes=self.tracked_classes,
                        means=means,
                        record_metric_per_sample=record_metric_per_sample,
                    ),
                    "scenario.y",
                    "scenario.y_pred",
                )
                for writer in self.hub.writers:
                    if (
                        isinstance(writer, ResultsLogWriter)
                        and writer.task_type == "benign"
                    ):
                        meter.add_writer(writer)
                        break

                self.hub.connect_meter(meter)

            if not self.skip_attack:
                meter = GlobalMeter(
                    "adversarial_hota_metrics",
                    GlobalHOTA(
                        metrics=self.hota_tasks,
                        coco_format=self.coco_format,
                        tracked_classes=self.tracked_classes,
                        means=means,
                        record_metric_per_sample=record_metric_per_sample,
                    ),
                    "scenario.y",
                    "scenario.y_pred_adv",
                )
                for writer in self.hub.writers:
                    if (
                        isinstance(writer, ResultsLogWriter)
                        and writer.task_type == "adversarial"
                        and writer.wrt == "ground truth"
                    ):
                        meter.add_writer(writer)
                        break

                self.hub.connect_meter(meter)

    def run_benign(self):
        self._check_x("run_benign")
        self.hub.set_context(stage="benign")
        x = self.x
        x.flags.writeable = False
        with self.profiler.measure("Inference"):
            y_pred = self.model.predict(x, **self.predict_kwargs)
        self.y_pred = y_pred
        self.probe.update(y_pred=y_pred)

    def run_attack(self):
        self._check_x("run_attack")
        self.hub.set_context(stage="attack")
        x, y = self.x, self.y
        with self.profiler.measure("Attack"):
            if self.use_label:
                y_target = y
            elif self.targeted:
                y_target = self.label_targeter.generate(y)
            else:
                y_target = None

            x_adv = self.attack.generate(
                x=x,
                y=y_target,
                y_patch_metadata=[self.y_patch_metadata],
                **self.generate_kwargs,
            )

        self.hub.set_context(stage="adversarial")
        # Ensure that input sample isn't overwritten by model
        x_adv.flags.writeable = False

        y_pred_adv = self.model.predict(x_adv, **self.predict_kwargs)

        self.probe.update(x_adv=x_adv, y_pred_adv=y_pred_adv)
        if self.targeted:
            self.probe.update(y_target=y_target)

        self.x_adv, self.y_target, self.y_pred_adv = x_adv, y_target, y_pred_adv
