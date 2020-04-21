"""
UFC101 actiona recognition adversarial dataset generated using non-universal,
non-patch perturbation attack
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
Dataset contains five randomly chosen adversarial videos from each class,
totaling 505 videos. Each video is broken into its component frames.  Therefore,
the dataset comprises sum_i(N_i) images, where N_i is the number of frames in video
i.  All frames are of the size (112, 112, 3)
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
_DL_URL = "/armory/datasets/ucf101_mars_perturbation_adversarial_112x112.tar.gz" #TODO: Update to S3 bucket

class Ucf101MarsPerturbationAdversarial112x112(tfds.core.GeneratorBasedBuilder):
    """ Ucf101 action recognition adversarial dataset"""

    VERSION = tfds.core.Version('1.0.0')

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "video": tfds.features.Video(shape=(None, 112, 112, 3)),
                "label": tfds.features.ClassLabel(names=_LABELS),
                "videoname": tfds.features.Text(),
            }),
            supervised_keys=("video", "label"),
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        """Adversarial dataset only has TEST split"""
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
        root_dir = 'data'
        for label in tf.io.gfile.listdir(os.path.join(data_dir_path, root_dir)):
            for videoname in tf.io.gfile.listdir(os.path.join(data_dir_path, root_dir, label)):
                video = []
                for filename in tf.io.gfile.glob(os.path.join(data_dir_path, root_dir, label, videoname, "*.png")):
                    video.append(filename)
                example = {
                    "video": video,
                    "label": label,
                    "videoname": videoname,
                }
                yield videoname, example

