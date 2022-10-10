"""
CARLA Multi-Object Tracking Scenario

"""

from armory.scenarios.carla_video_tracking import CarlaVideoTracking

from armory.metrics.task import HOTA_metrics
from armory.logs import log


class CarlaMOT(CarlaVideoTracking):
    def __init__(self, config, **kwargs):
        self.tracked_classes = config.get("scenario", {}).get(
            "tracked_classes", ["pedestrian"]
        )
        super().__init__(config, **kwargs)

    def load_metrics(self):
        all_tasks = self.config["metric"]["task"]
        hotas = ["hota", "deta", "assa"]
        self.hota_tasks = [t for t in all_tasks if t in hotas]
        self.config["metric"]["task"] = [t for t in all_tasks if t not in hotas]
        super().load_metrics()
        self.config["metric"]["task"] = all_tasks  # revert to original

        # metrics collector
        self.hota_metrics_benign = HOTA_metrics(tracked_classes=self.tracked_classes)
        self.hota_metrics_adversarial = HOTA_metrics(
            tracked_classes=self.tracked_classes
        )

    def run_benign(self):
        self._check_x("run_benign")
        self.hub.set_context(stage="benign")
        x, y = self.x, self.y
        x.flags.writeable = False
        with self.profiler.measure("Inference"):
            y_pred = self.model.predict(x[0], **self.predict_kwargs)
            for tracked_class in self.tracked_classes:
                self.hota_metrics_benign.calculate_hota_metrics_per_class_per_video(
                    y[0], y_pred, tracked_class, self.i
                )
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

        y_pred_adv = self.model.predict(x_adv[0], **self.predict_kwargs)

        for tracked_class in self.tracked_classes:
            self.hota_metrics_adversarial.calculate_hota_metrics_per_class_per_video(
                y[0], y_pred_adv, tracked_class, self.i
            )

        self.probe.update(x_adv=x_adv, y_pred_adv=y_pred_adv)
        if self.targeted:
            self.probe.update(y_target=y_target)

        self.x_adv, self.y_target, self.y_pred_adv = x_adv, y_target, y_pred_adv

    def finalize_results(self):
        super().finalize_results()

        for tracked_class in self.tracked_classes:
            self.hota_metrics_benign.calculate_hota_metrics_per_class_all_videos(
                tracked_class
            )
            self.hota_metrics_adversarial.calculate_hota_metrics_per_class_all_videos(
                tracked_class
            )

        log.success("Final HOTA metrics")
        bengin_per_class_per_video_metrics = (
            self.hota_metrics_benign.get_per_class_per_video_metrics()
        )
        bengin_per_class_all_videos_metrics = (
            self.hota_metrics_benign.get_per_class_all_videos_metrics()
        )
        adversarial_per_class_per_video_metrics = (
            self.hota_metrics_adversarial.get_per_class_per_video_metrics()
        )
        adversarial_per_class_all_videos_metrics = (
            self.hota_metrics_adversarial.get_per_class_all_videos_metrics()
        )
        for tracked_class in self.tracked_classes:
            log.success(f"Benign metrics for each video of {tracked_class} class")
            for k in ["hota", "deta", "assa"]:
                self.results[f"benign_{k}"] = []
            for vid in bengin_per_class_per_video_metrics[tracked_class].keys():
                for k in ["HOTA", "DetA", "AssA"]:
                    # there are many HOTA sub-metrics. We care mostly about the mean values of these three.
                    value = bengin_per_class_per_video_metrics[tracked_class][vid][
                        k
                    ].mean()
                    log.success(f"Video {vid}, {k} metric: {value}")
                    self.results[f"benign_{k.lower()}"].append(value)

            log.success(f"Benign metrics for all videos of {tracked_class} class")
            for k in ["HOTA", "DetA", "AssA"]:
                value = bengin_per_class_all_videos_metrics[tracked_class][k].mean()
                log.success(f"{k} metric: {value}")
                self.results[f"benign_mean_{k.lower()}"] = value

            log.success(f"Adversarial metrics for each video of {tracked_class} class")
            for k in ["hota", "deta", "assa"]:
                self.results[f"adversarial_{k}"] = []
            for vid in adversarial_per_class_per_video_metrics[tracked_class].keys():
                for k in ["HOTA", "DetA", "AssA"]:
                    value = adversarial_per_class_per_video_metrics[tracked_class][vid][
                        k
                    ].mean()
                    log.success(f"Video {vid}, {k} metric: {value}")
                    self.results[f"adversarial_{k.lower()}"].append(value)

            log.success(f"Adversarial metrics for all videos of {tracked_class} class")
            for k in ["HOTA", "DetA", "AssA"]:
                value = adversarial_per_class_all_videos_metrics[tracked_class][
                    k
                ].mean()
                log.success(f"{k} metric: {value}")
                self.results[f"adversarial_mean_{k.lower()}"] = value
