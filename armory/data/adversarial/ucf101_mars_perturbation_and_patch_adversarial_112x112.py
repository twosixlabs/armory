"""
UFC101 action recognition adversarial dataset generated using
perturbation and patch attacks
"""

from __future__ import absolute_import, division, print_function

import os

import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """
@article{DBLP:journals/corr/abs-1212-0402,
  author    = {Khurram Soomro and
               Amir Roshan Zamir and
               Mubarak Shah},
  title     = {{UCF101:} {A} Dataset of 101 Human Actions Classes From Videos in
               The Wild},
  journal   = {CoRR},
  volume    = {abs/1212.0402},
  year      = {2012},
  url       = {http://arxiv.org/abs/1212.0402},
  archivePrefix = {arXiv},
  eprint    = {1212.0402},
  timestamp = {Mon, 13 Aug 2018 16:47:45 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1212-0402},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DESCRIPTION = """
Dataset contains five randomly chosen videos from each class, taken from test split,
totaling 505 videos. Each video is broken into its component frames.  Therefore,
the dataset comprises sum_i(N_i) images, where N_i is the number of frames in video
i.  All frames are of the size (112, 112, 3). For each video, a clean version,
an adversarially perturbed version (using whole video perturbation attacks), and
an adversarially patched version (using regional patch attacks) are included.
"""

_LABELS = [
    "ApplyEyeMakeup",
    "ApplyLipstick",
    "Archery",
    "BabyCrawling",
    "BalanceBeam",
    "BandMarching",
    "BaseballPitch",
    "Basketball",
    "BasketballDunk",
    "BenchPress",
    "Biking",
    "Billiards",
    "BlowDryHair",
    "BlowingCandles",
    "BodyWeightSquats",
    "Bowling",
    "BoxingPunchingBag",
    "BoxingSpeedBag",
    "BreastStroke",
    "BrushingTeeth",
    "CleanAndJerk",
    "CliffDiving",
    "CricketBowling",
    "CricketShot",
    "CuttingInKitchen",
    "Diving",
    "Drumming",
    "Fencing",
    "FieldHockeyPenalty",
    "FloorGymnastics",
    "FrisbeeCatch",
    "FrontCrawl",
    "GolfSwing",
    "Haircut",
    "Hammering",
    "HammerThrow",
    "HandstandPushups",
    "HandstandWalking",
    "HeadMassage",
    "HighJump",
    "HorseRace",
    "HorseRiding",
    "HulaHoop",
    "IceDancing",
    "JavelinThrow",
    "JugglingBalls",
    "JumpingJack",
    "JumpRope",
    "Kayaking",
    "Knitting",
    "LongJump",
    "Lunges",
    "MilitaryParade",
    "Mixing",
    "MoppingFloor",
    "Nunchucks",
    "ParallelBars",
    "PizzaTossing",
    "PlayingCello",
    "PlayingDaf",
    "PlayingDhol",
    "PlayingFlute",
    "PlayingGuitar",
    "PlayingPiano",
    "PlayingSitar",
    "PlayingTabla",
    "PlayingViolin",
    "PoleVault",
    "PommelHorse",
    "PullUps",
    "Punch",
    "PushUps",
    "Rafting",
    "RockClimbingIndoor",
    "RopeClimbing",
    "Rowing",
    "SalsaSpin",
    "ShavingBeard",
    "Shotput",
    "SkateBoarding",
    "Skiing",
    "Skijet",
    "SkyDiving",
    "SoccerJuggling",
    "SoccerPenalty",
    "StillRings",
    "SumoWrestling",
    "Surfing",
    "Swing",
    "TableTennisShot",
    "TaiChi",
    "TennisSwing",
    "ThrowDiscus",
    "TrampolineJumping",
    "Typing",
    "UnevenBars",
    "VolleyballSpiking",
    "WalkingWithDog",
    "WallPushups",
    "WritingOnBoard",
    "YoYo",
]

_URL = "https://www.crcv.ucf.edu/data/UCF101.php"
_DL_URL = (
    "https://armory-public-data.s3.us-east-2.amazonaws.com"
    "/ucf101-adv/ucf101_mars_perturbation_and_patch_adversarial_112x112.tar.gz"
)


class Ucf101MarsPerturbationAndPatchAdversarial112x112(tfds.core.GeneratorBasedBuilder):
    """Ucf101 action recognition adversarial dataset"""

    VERSION = tfds.core.Version("1.1.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "videos": {
                        "clean": tfds.features.Video(shape=(None, 112, 112, 3)),
                        "adversarial_perturbation": tfds.features.Video(
                            shape=(None, 112, 112, 3)
                        ),
                        "adversarial_patch": tfds.features.Video(
                            shape=(None, 112, 112, 3)
                        ),
                    },
                    "labels": {
                        "clean": tfds.features.ClassLabel(names=_LABELS),
                        "adversarial_perturbation": tfds.features.ClassLabel(
                            names=_LABELS
                        ),
                        "adversarial_patch": tfds.features.ClassLabel(names=_LABELS),
                    },
                    "videoname": tfds.features.Text(),
                }
            ),
            supervised_keys=("videos", "labels"),
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """
        Returns SplitGenerators.

        Adversarial dataset only has TEST split
        """
        dl_path = dl_manager.download_and_extract(_DL_URL)
        return [
            tfds.core.SplitGenerator(
                name="adversarial",
                gen_kwargs={
                    "data_dir_path": dl_path,
                },
            ),
        ]

    def _generate_examples(self, data_dir_path):
        """Yields examples."""
        root_dir = "data"
        split_dirs = ["clean", "adversarial_perturbation", "adversarial_patch"]
        labels = tf.io.gfile.listdir(
            os.path.join(data_dir_path, root_dir, split_dirs[0])
        )
        labels.sort(key=str.casefold)  # _LABELS is sorted case insensitive
        assert labels == _LABELS
        for i, label in enumerate(labels):
            videonames = tf.io.gfile.listdir(
                os.path.join(data_dir_path, root_dir, split_dirs[0], label)
            )
            videonames.sort()
            for videoname in videonames:
                # prepare clean data
                video_clean = tf.io.gfile.glob(
                    os.path.join(
                        data_dir_path,
                        root_dir,
                        split_dirs[0],
                        label,
                        videoname,
                        "*.jpg",
                    )
                )
                video_clean.sort()
                # prepare adversarial perturbation data
                adv_pert = tf.io.gfile.glob(
                    os.path.join(
                        data_dir_path,
                        root_dir,
                        split_dirs[1],
                        label,
                        videoname,
                        "*.png",
                    )
                )
                adv_pert.sort()
                # prepare adversarial patch data
                adv_patch = tf.io.gfile.glob(
                    os.path.join(
                        data_dir_path,
                        root_dir,
                        split_dirs[2],
                        label,
                        videoname,
                        "*.png",
                    )
                )
                adv_patch.sort()
                example = {
                    "videos": {
                        "clean": video_clean,
                        "adversarial_perturbation": adv_pert,
                        "adversarial_patch": adv_patch,
                    },
                    "labels": {
                        "clean": label,
                        "adversarial_perturbation": label,  # untargeted, label not used
                        "adversarial_patch": labels[(i + 1) % len(_LABELS)],  # targeted
                    },
                    "videoname": videoname,
                }
                yield videoname, example
