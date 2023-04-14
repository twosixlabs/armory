import copy
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import numpy as np
import torch
from torchvision.ops import nms

from armory.logs import log
from armory import metrics
from armory.data.datasets import NumpyDataGenerator
from armory.instrument import LogWriter, Meter, ResultsWriter
from armory.scenarios.poison import Poison
from armory.utils import config_loading
from armory.instrument.export import (
    CocoBoxFormatMeter,
    ExportMeter,
    ObjectDetectionExporter,
)

"""
Paper link: https://arxiv.org/pdf/2205.14497.pdf
"""

import tracemalloc

class ObjectDetectionPoisoningScenario(Poison):

    def apply_augmentation(self, images, y_dicts, resize_width=None, resize_height=None):
        # Apply data augmentations using imgaug following https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/pytorchyolo/utils/transforms.py
        # image: np.ndarray
        # y_dicts: list of y objects containing "labels" and "boxes"
        # resize_width, resize_height: new image dimension
        # If resize_width and height are provided, this will resize data.  Otherwise it will apply augmentations.
        

        # Define augmentations
        if resize_width is not None: # TODO raise error if only one is None
            augmentations = iaa.Sequential([            
                iaa.PadToAspectRatio(1.0, position='center-center'),
                iaa.Resize({'height': resize_height, 'width': resize_width}, interpolation='nearest')
            ])
        else:
            augmentations = iaa.Sequential([
                iaa.Affine(rotate=(0, 0), translate_percent=(0.0, 0.0), scale=(0.8, 1.2)),
                iaa.Fliplr(0.5),
            ])

        aug_images = []
        aug_ydicts = []

        for image, y_dict in zip(images, y_dicts):
            # TODO sequential should be able to do a batch of images instead of looping.  
            # not sure about BoundingBoxesOnImage though

            if len(image.shape) == 4 and image.shape[0] == 1:
                image = np.squeeze(image, axis=0)

            boxes = y_dict["boxes"]
            labels = y_dict["labels"]

            # Convert bounding boxes to imgaug
            bounding_boxes = BoundingBoxesOnImage(
                [BoundingBox(*box, label) for box, label in zip(boxes, labels)],
                shape=image.shape)

            # Apply augmentations
            img, bounding_boxes = augmentations(
                image=image,
                bounding_boxes=bounding_boxes)

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
            aug_ydicts.append({
                        'boxes': boxes,
                        'labels': labels,
                        'scores': np.ones_like(labels)
                    })

        return np.array(aug_images, dtype=np.float32), aug_ydicts


    def filter_label(self, y):
        # Remove boxes/labels from y if the patch wouldn't fit.
        # If no boxes/labels are left, return False as a signal to skip this iamge completely.
        # TODO
        return y

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
        self.train_epochs = adhoc_config["train_epochs"]
        self.fit_batch_size = adhoc_config.get(
            "fit_batch_size", self.config["dataset"]["batch_size"]
        )

        self.label_function = lambda y: y

        dataset_config = copy.deepcopy(self.config["dataset"])
        dataset_config["batch_size"] = 1  # load with batch size 1 to simplify looping and filtering small boxes
        log.info(f"Loading dataset {dataset_config['name']}...")
        ds = config_loading.load_dataset(
            dataset_config,
            split=dataset_config.get("train_split", "train"),
            **self.dataset_kwargs,
        )
        log.info("Loading and resizing data")
        # It is desired to resize the data before poisoning occurs,
        # which is why this is done here and not in the model.
        self.patch_x_dim, self.patch_y_dim = self.config["attack"]["kwargs"]["backdoor_kwargs"]["size"]

        x_clean, y_clean = [], []
        for xc, yc in list(ds):
            img, yc = self.apply_augmentation([xc], yc, 416, 416)
            self.filter_label(yc) 
            x_clean.append(img)
            y_clean.append(yc[0])

        self.x_clean = np.concatenate(x_clean, axis=0)
        self.y_clean = np.array(y_clean, dtype='object')


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

            # attack = config_loading.load(attack_config)
            # self.poisoner = DatasetPoisoner(
            #     attack,
            #     self.source_class, #
            #     self.target_class, #
            #     fraction = kwargs["percent_poison"]
            # )
            self.poisoner = config_loading.load(attack_config)
            # try not having an extra DataPoisoner object which adds no functionality
            # TODO if this works out, either way clean up around here.

            # Need separate poisoner for test time because this attack is constructed
            # with a poison percentage, which differs at train and test times.
            test_attack_config = copy.deepcopy(attack_config)
            test_attack_config["kwargs"]["percent_poison"] = 1
            self.test_poisoner = config_loading.load(test_attack_config)
            # self.test_poisoner = DatasetPoisoner(
            #     test_attack,
            #     self.source_class, #
            #     self.target_class, #
            #     fraction = 1
            # )

    def poison_dataset(self):
        self.hub.set_context(stage="poison")
        if self.use_poison:
            self.x_poison, self.y_poison = self.poisoner.poison(
                self.x_clean, self.y_clean
            )
            # self.x_poison = np.array(self.x_poison)
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
        # self.train_set_class_labels = sorted(np.unique(self.y_clean))
        self.probe.update(y_clean=self.y_clean)
        # for y in self.train_set_class_labels: TODO need this?
        #     self.hub.record(
        #         f"class_{y}_N_train_samples", int(np.sum(self.y_clean == y))
        #     )


    def fit(self):
        # This function is over-ridden to apply random image augmentation every epoch.
        # Supposedly, this will be handled in ART in a future update, at which point this 
        # should be unnecessary.
        # Also, it appears that if you pass a new NumpyDataGenerator to model.fit_generator()
        # at each epoch, the old Generators do not get released from memory, resulting in a leak.
        # So here we manually batch the data and use model.fit().

        if len(self.x_train):
            self.hub.set_context(stage="fit")
            log.info("Fitting model")

            #  Every epoch, apply random augmentation
            for epoch in range(self.train_epochs):
                log.info(f"Applying augmentations for epoch {epoch}")
                aug_x_train, aug_y_train = self.apply_augmentation(self.x_train, self.y_train)
                log.info("Augmentation finished, training model")

                # Manually call model.fit with small batches
                for i in range(0, len(aug_y_train), self.fit_batch_size):
                    if i + self.fit_batch_size < len(aug_y_train):
                        self.model.fit(
                            aug_x_train[i: i+self.fit_batch_size],
                            self.label_function(aug_y_train[i: i+self.fit_batch_size]),
                            batch_size=self.fit_batch_size,
                            nb_epochs=1,
                            verbose=False,
                            shuffle=True,
                        )
        else:
            log.warning("All data points filtered by defense. Skipping training")


    def make_AP_meter(self, name, y, y_pred, target_class=None):
        # A little helper function to make metrics
        metric_kwargs = {}
        if target_class is not None:
            metric_kwargs["class_list"] = [target_class]
        self.hub.connect_meter(
            Meter(
                name,
                metrics.get("object_detection_mAP"),
                y,
                y_pred,
                metric_kwargs=metric_kwargs,
                final=np.mean,
                final_name=name,
                record_final_only=True,
            )
        )

    def load_metrics(self):
        self.score_threshold = 0.0 # TODO set final score threshold
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

        # The paper uses short but vague names for the metrics, which are here replaced with longer
        # more descriptive names.  The paper's names will be mentioned in the comments for reference.
        target = (
            self.target_class if self.target_class is not None else self.source_class
        )

        #  1 mAP benign test all classes  #
        #    mAP_benign
        self.make_AP_meter(
            "mAP_on_benign_test_data_all_classes",
            "scenario.y",
            "scenario.y_pred",
        )

        #  2 AP benign test target class  #
        #    AP_benign
        self.make_AP_meter(
            "AP_on_benign_test_data_target_class",
            "scenario.y",
            "scenario.y_pred",
            target,
        )

        if self.use_poison:

            #  3 mAP adv test, adv labels  #
            #    mAP_attack
            self.make_AP_meter(
                "mAP_on_adv_test_data_with_poison_labels_all_classes",
                "scenario.y_adv",
                "scenario.y_pred_adv",
            )

            #  4 mAP adv test, adv labels, target class  #
            #    AP_attack
            #    Not applicable to Disappearance
            if self.target_class is not None:
                self.make_AP_meter(
                    "AP_on_adv_test_data_with_poison_labels_target_class",
                    "scenario.y_adv",
                    "scenario.y_pred_adv",
                    target,
                )

            #  5 mAP adv test, clean labels   #
            #    mAP_attack+benign
            #    Not applicable to Generation
            if self.attack_variant != "BadDetObjectGenerationAttack":
                self.make_AP_meter(
                    "mAP_on_adv_test_data_with_clean_labels_all_classes",
                    "scenario.y",
                    "scenario.y_pred_adv",
                )

            #  6 AP adv test, clean labels, target class  #
            #    AP_attack+benign
            #    Not applicable to Generation
            if self.attack_variant != "BadDetObjectGenerationAttack":
                self.make_AP_meter(
                    "AP_on_adv_test_data_with_clean_labels_target_class",
                    "scenario.y",
                    "scenario.y_pred_adv",
                    target,
                )

            #  7 Attack Success Rate  #

            # ASR -- Misclassification
            if self.attack_variant in [
                "BadDetRegionalMisclassificationAttack",
                "BadDetGlobalMisclassificationAttack",
            ]:
                self.hub.connect_meter(
                    Meter(
                        "attack_success_rate_misclassification",
                        metrics.get(
                            "object_detection_poisoning_targeted_misclassification_rate"
                        ),
                        "scenario.y",
                        "scenario.y_pred_adv",
                        metric_kwargs={
                            "target_class": self.target_class,
                            "source_class": self.source_class,
                            "score_threshold": self.score_threshold
                        },
                        final=np.mean,
                        final_name="attack_success_rate_misclassification",
                        record_final_only=True,
                    )
                )

            # ASR -- Disappearance
            if self.attack_variant == "BadDetObjectDisappearanceAttack":
                self.hub.connect_meter(
                    Meter(
                        "attack_success_rate_disappearance",
                        metrics.get(
                            "object_detection_poisoning_targeted_disappearance_rate"
                        ),
                        "scenario.y",
                        "scenario.y_pred_adv",
                        metric_kwargs={"source_class": self.source_class, "score_threshold": self.score_threshold},
                        final=np.mean,
                        final_name="attack_success_rate_disappearance",
                        record_final_only=True,
                    )
                )

            # ASR -- Generation
            if self.attack_variant == "BadDetObjectGenerationAttack":
                self.hub.connect_meter(
                    Meter(
                        "attack_success_rate_generation",
                        metrics.get(
                            "object_detection_poisoning_targeted_generation_rate"
                        ),
                        "scenario.y_adv",
                        "scenario.y_pred_adv",
                        metric_kwargs={"score_threshold": self.score_threshold},
                        final=np.mean,
                        final_name="attack_success_rate_generation",
                        record_final_only=True,
                    )
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

        boxes = predictions['boxes']
        labels = predictions['labels']
        scores = predictions['scores']

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

        return {'boxes': boxes, 'labels': labels, 'scores': scores}

    def next(self):
        # Over-ridden to resize data.

        self.hub.set_context(stage="next")
        x, y = next(self.test_dataset)
        i = self.i + 1
        self.hub.set_context(batch=i)
        
        self.y_pred, self.y_target, self.x_adv, self.y_pred_adv = None, None, None, None
        x, y = self.apply_augmentation(x, y, 416, 416)
        if len(x.shape) == 3:
            print("APPLY AUGMENTATION did not return x with batch dimension")
            x = np.array([x])  # add batch dim back on

        # TODO self.filter_label():
        # If patch does not fit in boxes, skip image or remove small boxes
        # 
        
        self.i, self.x, self.y = i, x, y
        self.probe.update(i=i, x=x, y=y)

    def run_benign(self):
        self.hub.set_context(stage="benign")
        x = self.x

        x.flags.writeable = False
        y_pred = self.model.predict(x, **self.predict_kwargs)
        y_pred = [self.non_maximum_supression(pred) for pred in y_pred]
        self.probe.update(y_pred=y_pred)
        # source = y == self.source_class TODO
        # uses source->target trigger
        # if source.any():
        #     self.probe.update(y_source=y[source], y_pred_source=y_pred[source])

        self.y_pred = y_pred
        # self.source = source
        if self.explanatory_model is not None:
            self.run_explanatory()

    def run_attack(self):
        self.hub.set_context(stage="attack")
        x, y = self.x, [self.y[0]]  # TODO carla data... what to do about metadata dict
        # source = self.source

        x_adv, y_adv = self.test_poisoner.poison(x, y)

        self.hub.set_context(stage="adversarial")
        x_adv.flags.writeable = False
        y_pred_adv = self.model.predict(x_adv, **self.predict_kwargs)
        y_pred_adv = [self.non_maximum_supression(pred) for pred in y_pred_adv]
        self.probe.update(x_adv=x_adv, y_adv=y_adv, y_pred_adv=y_pred_adv)

        # uses source->target trigger TODO
        # if source.any():
        #     self.probe.update(
        #         target_class_source=[self.target_class] * source.sum(),
        #         y_pred_adv_source=y_pred_adv[source],
        #     )

        self.x_adv = x_adv
        self.y_pred_adv = y_pred_adv

    def load_fairness_metrics(self):
        raise NotImplementedError(
            "The fairness metrics have not been implemented for object detection poisoning"
        )
        # TODO

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

        # TODO y objects are missing "image_id" which is used by coco_format_meter
        # coco_box_format_meter = CocoBoxFormatMeter(
        #     "coco_box_format_meter",
        #     self.export_dir,
        #     y_probe="scenario.y",
        #     y_pred_clean_probe="scenario.y_pred" if not self.skip_benign else None,
        #     y_pred_adv_probe="scenario.y_pred_adv" if not self.skip_attack else None,
        #     max_batches=self.num_export_batches,
        # )
        # self.hub.connect_meter(coco_box_format_meter, use_default_writers=False)

    def _load_sample_exporter(self):
        return ObjectDetectionExporter(self.export_dir)

    def _load_sample_exporter_with_boxes(self):
        return ObjectDetectionExporter(
            self.export_dir, default_export_kwargs={"with_boxes": True, "score_threshold": self.score_threshold}
        )
