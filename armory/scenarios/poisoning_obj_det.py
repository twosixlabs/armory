import copy

from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imgaug.augmenters as iaa
import numpy as np
import torch
from torchvision.ops import nms
from tqdm import tqdm

from armory import metrics
from armory.instrument import GlobalMeter, LogWriter, Meter, ResultsWriter
from armory.instrument.export import ExportMeter, ObjectDetectionExporter
from armory.logs import log
from armory.scenarios.poison import Poison
from armory.utils import config_loading

"""
Paper link: https://arxiv.org/pdf/2205.14497.pdf
"""


class ObjectDetectionPoisoningScenario(Poison):
    def apply_augmentation(
        self,
        images,
        y_dicts,
        resize_dims=None,
    ):
        # Apply data augmentations using imgaug following https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/pytorchyolo/utils/transforms.py
        # image: np.ndarray
        # y_dicts: list of y objects containing "labels" and "boxes"
        # resize_dims: tuple, new image dimension
        # If resize_dims is provided, this will resize data.  Otherwise it will apply augmentations.

        # Define augmentations
        if resize_dims is not None:
            augmentations = iaa.Sequential(
                [
                    iaa.PadToAspectRatio(1.0, position="center-center"),
                    iaa.Resize(
                        {"height": resize_dims[0], "width": resize_dims[1]},
                        interpolation="nearest",
                    ),
                ]
            )
        else:
            augmentations = iaa.Sequential(
                [
                    iaa.Fliplr(0.5),
                ]
            )

        aug_images = []
        aug_ydicts = []
        for image, y_dict in zip(images, y_dicts):

            if len(image.shape) == 4 and image.shape[0] == 1:
                image = np.squeeze(image, axis=0)

            boxes = y_dict["boxes"]
            labels = y_dict["labels"]

            # Convert bounding boxes to imgaug
            bounding_boxes = BoundingBoxesOnImage(
                [BoundingBox(*box, label) for box, label in zip(boxes, labels)],
                shape=image.shape,
            )

            # Apply augmentations
            img, bounding_boxes = augmentations(
                image=image, bounding_boxes=bounding_boxes
            )

            # Clip out of image boxes
            bounding_boxes = bounding_boxes.clip_out_of_image()

            # Convert bounding boxes back to numpy
            boxes = np.zeros((len(bounding_boxes), 4))
            labels = np.zeros(len(bounding_boxes))
            for box_idx, box in enumerate(bounding_boxes):
                labels[box_idx] = box.label
                boxes[box_idx, 0] = box.x1
                boxes[box_idx, 1] = box.y1
                boxes[box_idx, 2] = box.x2
                boxes[box_idx, 3] = box.y2

            aug_images.append(img)
            aug_ydicts.append(
                {"boxes": boxes, "labels": labels, "scores": np.ones_like(labels)}
            )

        return np.array(aug_images, dtype=np.float32), aug_ydicts

    def filter_label(self, y):
        # Remove boxes/labels from y if the patch wouldn't fit.
        # If no boxes/labels are left, return None as a signal to skip this image completely.

        new_y = {"boxes": [], "labels": [], "scores": []}

        if len(y) > 1:
            raise ValueError(
                f"filter_label accepts lists of length 1, got length {len(y)}"
            )
        y = y[0]
        for box, label in zip(y["boxes"], y["labels"]):
            if (
                box[2] - box[0] > self.patch_x_dim
                and box[3] - box[1] > self.patch_y_dim
            ):
                new_y["boxes"].append(box)
                new_y["labels"].append(label)
                new_y["scores"].append(1)

            else:
                self.n_boxes_removed_by_class[label] += 1

        if len(new_y["labels"]) != len(y["labels"]):
            self.n_images_affected += 1

        if len(new_y["labels"]) > 0:
            new_y["boxes"] = np.array(new_y["boxes"])
            new_y["labels"] = np.array(new_y["labels"])
            new_y["scores"] = np.array(new_y["scores"])
            return [new_y]
        else:
            self.n_images_removed += 1
            return None

    def load_train_dataset(self, train_split_default=None):
        """
        Load and create in memory dataset
            detect_poison does not currently support data generators
        """
        # This is inherited and modified slightly to apply image resizing and augmentation
        # which in the future will be done by a preprocessor in ART.
        if train_split_default is not None:
            raise ValueError(
                "train_split_default not used in this loading method for poison"
            )
        adhoc_config = self.config.get("adhoc") or {}
        self.attack_variant = self.config["attack"]["kwargs"]["attack_variant"]
        self.train_epochs = adhoc_config["train_epochs"]
        self.fit_batch_size = adhoc_config.get(
            "fit_batch_size", self.config["dataset"]["batch_size"]
        )
        self.config["dataset"]["batch_size"] = 1
        # Set batch size to 1 for loading data, this simplifies filtering small boxes.
        # Note, training is unaffected since self.fit_batch_size is already set above.

        self.label_function = lambda y: y

        dataset_config = self.config["dataset"]
        log.info(f"Loading dataset {dataset_config['name']}...")
        ds = config_loading.load_dataset(
            dataset_config,
            split=dataset_config.get("train_split", "train"),
            **self.dataset_kwargs,
        )
        log.info("Resizing data")
        # It is desired to resize the data before poisoning occurs,
        # which is why this is done here and not in the model.
        self.patch_x_dim, self.patch_y_dim = self.config["attack"]["kwargs"][
            "backdoor_kwargs"
        ]["size"]

        self.n_boxes_removed_by_class = {0: 0, 1: 0, 2: 0}
        self.n_images_removed = 0
        self.n_images_affected = 0
        x_clean, y_clean = [], []
        for xc, yc in list(ds):
            img, yc = self.apply_augmentation([xc], yc, (416, 416))
            if self.attack_variant in [
                "BadDetRegionalMisclassificationAttack",
                "BadDetObjectDisappearanceAttack",
            ]:
                # Not necessary for OGA or GMA because the patch is not applied in a box
                yc = self.filter_label(yc)
            if yc is not None:
                x_clean.append(img)
                y_clean.extend(yc)

        self.x_clean = np.concatenate(x_clean, axis=0)
        self.y_clean = np.array(y_clean, dtype="object")

        if self.n_images_affected > 0:
            log.info(
                f"Filtered out boxes where patch wouldn't fit:\n \
                    N boxes removed by class: {self.n_boxes_removed_by_class}\n \
                    N total images removed: {self.n_images_removed}\n \
                    N images with at least one box removed: {self.n_images_affected}"
            )

    def load_poisoner(self):
        adhoc_config = self.config.get("adhoc") or {}
        attack_config = self.config["attack"]
        if attack_config.get("type") == "preloaded":
            raise ValueError("preloaded attacks not currently supported for poisoning")

        self.use_poison = bool(adhoc_config["poison_dataset"])
        self.source_class = adhoc_config.get("source_class")  # None for GMA, OGA
        self.target_class = adhoc_config.get("target_class")  # None for ODA

        if self.use_poison:

            # Set additional attack config kwargs
            kwargs = attack_config["kwargs"]
            self.attack_variant = kwargs["attack_variant"]
            kwargs["percent_poison"] = adhoc_config["fraction_poisoned"]
            if self.target_class is not None:
                kwargs["class_target"] = self.target_class
            if self.source_class is not None:
                kwargs["class_source"] = self.source_class

            self.num_test_triggers = kwargs.pop("num_test_triggers", 1)
            if self.num_test_triggers > 1 and "Generation" not in self.attack_variant:
                raise ValueError(
                    f"{self.attack_variant} does not support multiple test-time triggers.  Please remove 'num_test_triggers' from config"
                )

            self.poisoner = config_loading.load(attack_config)

            # Need separate poisoner for test time because this attack is constructed
            # with a poison percentage, which differs at train and test times.
            test_attack_config = copy.deepcopy(attack_config)
            test_attack_config["kwargs"]["percent_poison"] = 1
            self.test_poisoner = config_loading.load(test_attack_config)

    def poison_dataset(self):
        self.hub.set_context(stage="poison")
        if self.use_poison:
            self.x_poison, self.y_poison = self.poisoner.poison(
                self.x_clean, self.y_clean
            )
            self.y_poison = np.array(self.y_poison)

            # this attack does not return poison indices, find them manually
            poison_index = np.array(
                [
                    i
                    for i in range(len(self.x_clean))
                    if (self.x_clean[i] != self.x_poison[i]).any()
                ]
            )

        else:
            self.x_poison, self.y_poison, poison_index = (
                self.x_clean,
                self.y_clean,
                np.array([], dtype=np.int64),
            )

        self.poison_index = poison_index
        self.record_poison_and_data_info()

    def record_poison_and_data_info(self):
        self.n_poisoned = int(len(self.poison_index))
        self.n_clean = len(self.y_poison) - self.n_poisoned
        self.poisoned = np.zeros_like(self.y_poison, dtype=bool)
        self.poisoned[self.poison_index.astype(np.int64)] = True
        self.probe.update(poisoned=self.poisoned, poison_index=self.poison_index)
        self.hub.record("N_poisoned_train_samples", self.n_poisoned)
        self.hub.record("N_clean_train_samples", self.n_clean)
        self.probe.update(y_clean=self.y_clean)

    def fit(self):
        # This function is over-ridden to apply random image augmentation every epoch.
        # Supposedly, augmentation will be handled in ART in a future update, at which
        # point this should be unnecessary.
        # Also, it appears that if you pass a new NumpyDataGenerator to model.fit_generator()
        # at each epoch, the old Generators do not get released from memory, resulting in a leak.
        # So here we manually batch the data and use model.fit().

        if len(self.x_train):
            self.hub.set_context(stage="fit")
            log.info("Fitting model")

            #  Every epoch, apply random augmentation
            for epoch in range(self.train_epochs):
                log.info(f"Augmenting and training for epoch {epoch}")
                aug_x_train, aug_y_train = self.apply_augmentation(
                    self.x_train, self.y_train
                )

                # Manually call model.fit with small batches
                for i in range(0, len(aug_y_train), self.fit_batch_size):
                    batch_end = min(len(aug_y_train), i + self.fit_batch_size)
                    self.model.fit(
                        aug_x_train[i:batch_end],
                        self.label_function(aug_y_train[i:batch_end]),
                        batch_size=self.fit_batch_size,
                        nb_epochs=1,
                        verbose=False,
                        shuffle=True,
                    )
        else:
            log.warning("All data points filtered by defense. Skipping training")

    def add_asr_metric(self, name, metric, labels, kwargs={}):
        # Helper function for connecting ASR meters
        kwargs["score_threshold"] = self.score_threshold
        self.hub.connect_meter(
            Meter(
                name,
                metrics.get(metric),
                labels,
                "scenario.y_pred_adv",
                metric_kwargs=kwargs,
                final=np.mean,
                final_name=name,
                record_final_only=True,
            )
        )

    def load_metrics(self):
        self.score_threshold = self.config["adhoc"].get("score_threshold", 0.05)
        if self.use_filtering_defense:
            # Filtering metrics
            self.hub.connect_meter(
                Meter(
                    "filter",
                    metrics.get("tpr_fpr"),
                    "scenario.poisoned",
                    "scenario.removed",
                )
            )

        for metric_name in [
            "object_detection_AP_per_class",
            "object_detection_mAP_tide",
        ]:
            short_name = metric_name[17:]  # remove "object_detection_"

            # mAP on benign test data
            self.hub.connect_meter(
                GlobalMeter(
                    f"{short_name}_on_benign_test_data",
                    metrics.get(metric_name),
                    "scenario.y",
                    "scenario.y_pred",
                )
            )

            if self.use_poison:
                # mAP adv preds, poison labels
                self.hub.connect_meter(
                    GlobalMeter(
                        f"{short_name}_on_adv_test_data_with_poison_labels",
                        metrics.get(metric_name),
                        "scenario.y_adv",
                        "scenario.y_pred_adv",
                    )
                )
                # mAP adv preds, clean labels
                #    Not applicable to Generation
                if self.attack_variant != "BadDetObjectGenerationAttack":
                    self.hub.connect_meter(
                        GlobalMeter(
                            f"{short_name}_on_adv_test_data_with_clean_labels",
                            metrics.get(metric_name),
                            "scenario.y",
                            "scenario.y_pred_adv",
                        )
                    )

        if self.use_poison:
            # Attack success rate
            if "Misclassification" in self.attack_variant:
                self.add_asr_metric(
                    "attack_success_rate_misclassification",
                    "object_detection_poisoning_targeted_misclassification_rate",
                    "scenario.y",
                    {
                        "target_class": self.target_class,
                        "source_class": self.source_class,
                    },
                )
            elif "Disappearance" in self.attack_variant:
                self.add_asr_metric(
                    "attack_success_rate_disappearance",
                    "object_detection_poisoning_targeted_disappearance_rate",
                    "scenario.y",
                    {"source_class": self.source_class},
                )
            elif "Generation" in self.attack_variant:
                self.add_asr_metric(
                    "attack_success_rate_generation",
                    "object_detection_poisoning_targeted_generation_rate",
                    "scenario.y_adv",
                    {"num_triggers": self.num_test_triggers},
                )

        if self.config["adhoc"].get("compute_fairness_metrics"):
            self.load_fairness_metrics()
        self.results_writer = ResultsWriter(sink=None)
        self.hub.connect_writer(self.results_writer, default=True)
        self.hub.connect_writer(LogWriter(), default=True)

    def non_maximum_supression(self, predictions):
        # This may be handled in ART in the future.

        MAX_PRE_NMS_BOXES = 10000
        MAX_POST_NMS_BOXES = 100
        IOU_THRESHOLD = 0.5

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        boxes = predictions["boxes"]
        labels = predictions["labels"]
        scores = predictions["scores"]

        # Filter by confidence score
        boxes = boxes[scores > self.score_threshold]
        labels = labels[scores > self.score_threshold]
        scores = scores[scores > self.score_threshold]

        # Sort by decreasing scores
        sort_ind = np.argsort(scores)[::-1]
        boxes = boxes[sort_ind[:MAX_PRE_NMS_BOXES]]
        labels = labels[sort_ind[:MAX_PRE_NMS_BOXES]]
        scores = scores[sort_ind[:MAX_PRE_NMS_BOXES]]

        # Calculate non-maximum-suppression indices
        boxes = torch.from_numpy(boxes).to(device)
        labels = torch.from_numpy(labels).to(device)
        scores = torch.from_numpy(scores).to(device)
        idx = nms(boxes, scores, IOU_THRESHOLD)
        if idx.shape[0] > MAX_POST_NMS_BOXES:
            idx = idx[:MAX_POST_NMS_BOXES]

        boxes = boxes[idx].detach().cpu().numpy()
        labels = labels[idx].detach().cpu().numpy()
        scores = scores[idx].detach().cpu().numpy()

        return {"boxes": boxes, "labels": labels, "scores": scores}

    def evaluate_all(self):
        log.info("Running inference on benign and adversarial examples")
        for _ in tqdm(range(len(self.test_dataset)), desc="Evaluation"):
            self.next()
            if not self.skip_this_sample:  # skip ones with boxes too small for trigger
                self.evaluate_current()
        self.hub.set_context(stage="finished")

    def next(self):
        # Over-ridden to resize data.

        self.hub.set_context(stage="next")
        x, y = next(self.test_dataset)
        i = self.i + 1
        self.hub.set_context(batch=i)
        self.y_pred, self.y_target, self.x_adv, self.y_pred_adv = None, None, None, None
        x, y = self.apply_augmentation(x, y, (416, 416))

        self.skip_this_sample = False
        if self.attack_variant in [
            "BadDetRegionalMisclassificationAttack",
            "BadDetObjectDisappearanceAttack",
        ]:
            y = self.filter_label(y)
        if y is None:
            self.skip_this_sample = True
        else:
            self.i, self.x, self.y = i, x, y
            self.probe.update(i=i, x=x, y=y)

    def run_benign(self):
        self.hub.set_context(stage="benign")
        x = self.x

        x.flags.writeable = False
        y_pred = self.model.predict(x, **self.predict_kwargs)
        y_pred = [self.non_maximum_supression(pred) for pred in y_pred]
        self.probe.update(y_pred=y_pred)
        self.y_pred = y_pred
        if self.explanatory_model is not None:
            self.run_explanatory()

    def run_attack(self):
        self.hub.set_context(stage="attack")
        x, y = self.x, self.y
        for i in range(self.num_test_triggers):
            # Generation can add multiple triggers to one image
            x, y = self.test_poisoner.poison(x, y)
        x_adv, y_adv = x, y
        self.hub.set_context(stage="adversarial")
        x_adv.flags.writeable = False
        y_pred_adv = self.model.predict(x_adv, **self.predict_kwargs)
        y_pred_adv = [self.non_maximum_supression(pred) for pred in y_pred_adv]
        self.probe.update(x_adv=x_adv, y_adv=y_adv, y_pred_adv=y_pred_adv)

        self.x_adv = x_adv
        self.y_pred_adv = y_pred_adv

    def load_fairness_metrics(self):
        raise NotImplementedError(
            "As currently defined, the fairness metrics are not applicable to object detection data."
        )

    def load_export_meters(self):
        super().load_export_meters()
        self.sample_exporter_with_boxes = self._load_sample_exporter_with_boxes()
        for probe_data, probe_pred in [("x", "y_pred"), ("x_adv", "y_pred_adv")]:
            export_with_boxes_meter = ExportMeter(
                f"{probe_data}_with_boxes_exporter",
                self.sample_exporter_with_boxes,
                f"scenario.{probe_data}",
                y_probe="scenario.y",
                y_pred_probe=f"scenario.{probe_pred}",
                max_batches=self.num_export_batches,
            )
            self.hub.connect_meter(export_with_boxes_meter, use_default_writers=False)
            if self.skip_attack:
                break

    def _load_sample_exporter(self):
        return ObjectDetectionExporter(self.export_dir)

    def _load_sample_exporter_with_boxes(self):
        return ObjectDetectionExporter(
            self.export_dir,
            default_export_kwargs={
                "with_boxes": True,
                "score_threshold": self.config["adhoc"].get("export_threshold", 0.05),
            },
        )
