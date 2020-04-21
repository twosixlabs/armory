"""
TensorFlow Dataset for adversarial librispeech
"""

import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """\
LibriSpeech-dev-clean adversarial audio dataset for SincNet

TODO: <Attack hyper-parameters here>
"""

_URL = "/armory/datasets/LibriSpeech_SincNet_UniversalPerturbation.tar.gz"


class LibrispeechAdversarial(tfds.core.GeneratorBasedBuilder):
    """LibriSpeech_SincNet_UniversalPerturbation.tfrecords"""

    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    # TODO might need to specify shape?
                    "sound": tfds.features.Tensor(shape=(), dtype=tf.float64),
                    "label": tfds.features.Tensor(shape=(), dtype=tf.int64),
                }
            ),
            supervised_keys=("sound", "label"),
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        path = dl_manager.download_and_extract(_URL)
        splits = [
            tfds.core.SplitGenerator(name=x, gen_kwargs={"path": path, "split": x})
            for x in ("adversarial", "clean")
        ]
        return splits

    def _generate_examples(self, path, split):
        """Yields examples."""
        if split == "adversarial":
            key = "adv-sound"
        else:
            raise ValueError(f"split {split} not in ('adversarial')")

        def _parse(serialized_example, key):
            ds_features = {
                "label": tf.io.FixedLenFeature([], tf.int64),
                "adv-sound": tf.io.VarLenFeature(tf.string),
            }
            example = tf.io.parse_single_example(serialized_example, ds_features)

            sound = tf.io.decode_raw(example[key], tf.float64)
            return sound, example["label"]

        ds = tf.data.TFRecordDataset(filenames=[path])
        ds = ds.map(lambda x: _parse(x, key))
        ds = ds.batch(1)
        default_graph = tf.compat.v1.keras.backend.get_session().graph
        ds = tfds.as_numpy(ds, graph=default_graph)
        for i, (sound, label) in enumerate(ds):
            yield str(i), {"sound": sound[0], "label": label[0]}
