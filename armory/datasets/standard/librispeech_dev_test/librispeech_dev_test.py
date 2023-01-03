"""
Subset of librispeech containing just 'dev' and 'test' splits.

checksums.tsv is empty as it uses the underlying librispeech class.

NOTE: In order to build, this requires apache beam installed.
    In the container, do: `pip install apache-beam`
    This is not installed by default due to older dependencies

NOTE: when building, armory does not provide beam options by default
    This makes building VERY slow unless overrides are provided
    It is recommended that this is built directly using tfds on the command line

Using DirectRunner with apache beam, can build with this:
    tfds build /workspace/armory/datasets/standard/librispeech_dev_test --data_dir /armory/datasets/new_builds --force_checksums_validation --beam_pipeline_options="runner=DirectRunner,direct_num_workers=16,direct_running_mode=multi_processing"
    See: https://beam.apache.org/releases/pydoc/2.43.0/_modules/apache_beam/options/pipeline_options.html#DirectOptions
"""

import tensorflow_datasets as tfds
from tensorflow_datasets.audio import librispeech

_SUBSET = (
    "dev_clean",
    "dev_other",
    "test_clean",
    "test_other",
)
_DL_URLS = {k: v for k, v in librispeech._DL_URLS.items() if k in _SUBSET}


class LibrispeechDevTest(librispeech.Librispeech):
    """DatasetBuilder for subset of Librispeech"""

    def _split_generators(self, dl_manager):
        extracted_dirs = dl_manager.download_and_extract(_DL_URLS)
        self._populate_metadata(extracted_dirs)
        splits = [
            tfds.core.SplitGenerator(name=k, gen_kwargs={"directory": v})
            for k, v in extracted_dirs.items()
        ]
        return splits
