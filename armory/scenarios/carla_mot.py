"""
CARLA Multi-Object Tracking Scenario

"""

from armory.scenarios.carla_video_tracking import CarlaVideoTracking

from armory.metrics.task import HOTA_metrics


class CarlaMOT(CarlaVideoTracking):
    def __init__(self, config, **kwargs):
        self.tracked_classes = config.get("scenario", {}).get(
            "tracked_classes", ["pedestrian"]
        )

        # metrics collector
        self.hota_metrics_benign = HOTA_metrics(tracked_classes=self.tracked_classes)
        self.hota_metrics_adversarial = HOTA_metrics(
            tracked_classes=self.tracked_classes
        )
        super().__init__(config, **kwargs)

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

        print("Final HOTA metrics")
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
            print("Benign metrics for each video of {} class".format(tracked_class))
            for vid in bengin_per_class_per_video_metrics[tracked_class].keys():
                for key_metric in [
                    "HOTA",
                    "DetA",
                    "AssA",
                ]:  # there are many HOTA sub-metrics. We care mostly about the mean values of these three.
                    print(
                        "Video {}, {} metric: {}".format(
                            vid,
                            key_metric,
                            bengin_per_class_per_video_metrics[tracked_class][vid][
                                key_metric
                            ].mean(),
                        )
                    )

            print("Benign metrics for all videos of {} class".format(tracked_class))
            for key_metric in ["HOTA", "DetA", "AssA"]:
                print(
                    "{} metric: {}".format(
                        key_metric,
                        bengin_per_class_all_videos_metrics[tracked_class][
                            key_metric
                        ].mean(),
                    )
                )

            print(
                "Adversarial metrics for each video of {} class".format(tracked_class)
            )
            for vid in adversarial_per_class_per_video_metrics[tracked_class].keys():
                for key_metric in ["HOTA", "DetA", "AssA"]:
                    print(
                        "Video {}, {} metric: {}".format(
                            vid,
                            key_metric,
                            adversarial_per_class_per_video_metrics[tracked_class][vid][
                                key_metric
                            ].mean(),
                        )
                    )

            print(
                "Adversarial metrics for all videos of {} class".format(tracked_class)
            )
            for key_metric in ["HOTA", "DetA", "AssA"]:
                print(
                    "{} metric: {}".format(
                        key_metric,
                        adversarial_per_class_all_videos_metrics[tracked_class][
                            key_metric
                        ].mean(),
                    )
                )
