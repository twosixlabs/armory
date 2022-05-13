from .builder.utils import SUPPORTED_DATASETS, DEFAULT_DATASET_DIRECTORY
from .loader import preprocessing_chain, load, generator_from_dataset
import sys
from typing import Callable
from typing import Optional


def _dataset_generator(name, directory, as_supervised: bool = True, supervised_xy_keys: Optional[tuple] = None):
    def _generator(preprocessing_fn: Callable = None,
        fit_preprocessing_fn: Callable = None,
        **kwargs):
        chain = preprocessing_chain(preprocessing_fn, fit_preprocessing_fn)
        ds_info, ds = load(name, directory, as_supervised)
        generator = generator_from_dataset(ds_info,
                                           ds,
                                           preprocessing_fn=chain,
                                           as_supervised=as_supervised,
                                           supervised_xy_keys=supervised_xy_keys, **kwargs)
        return generator
    return _generator

pars = {
    'ucf101':{'as_supervised':False, 'supervised_xy_keys':("video", "label")}
}

for k,v in SUPPORTED_DATASETS.items():

    as_supervised = pars.get(k).get('as_supervised') if pars.get(k) else True
    xy_keys = pars.get(k).get('supervised_xy_keys') if pars.get(k) else None
    setattr(sys.modules[__name__],
            k,
            _dataset_generator(k,
                               DEFAULT_DATASET_DIRECTORY,
                               as_supervised=as_supervised,
                               supervised_xy_keys=xy_keys))
    setattr(sys.modules[__name__],
            f"{v['expected_name']}:{v['expected_version']}",
            _dataset_generator(k,
                               DEFAULT_DATASET_DIRECTORY,
                               as_supervised=as_supervised,
                               supervised_xy_keys=xy_keys))

