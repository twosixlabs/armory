import pytest
import armory.environment as ae
import os
from pydantic import ValidationError

@pytest.fixture
def temp_armory_dir(tmp_path):
    armory_dir = tmp_path / ".armory"
    os.makedirs(armory_dir)
    default_paths = ae.Paths()
    for k,v in default_paths.__fields__.items():
        base_val = os.path.basename(v)
        new_val = os.path.join(armory_dir, base_val)
        if k == 'setup_file':
            with open(new_val,"w") as f:
                f.write("")
        else:
            os.makedirs(new_val)
        default_paths[k] = new_val
    return default_paths

def check_paths(kwargs, temp_armory_dir):
    paths = ae.Paths(**kwargs)
    for k, v in kwargs:
        assert paths[k] == v
        paths[k] = v
    default_paths = ae.Paths()
    for k, v in default_paths.__fields__.items():
        if k not in kwargs:
            assert paths[k] == v

@pytest.mark.parametrize(
    "kwargs, use_temp_dir, error",
    [
        ({'setup_file':"/nothing/profile"}, False, ValidationError),
        ({'setup_file':"profile"}, True, ValidationError)
    ]
)
def test_paths(kwargs, use_temp_dir, error, temp_armory_dir):
    if use_temp_dir:
        print(temp_armory_dir['setup_file'])
        for k, v in kwargs.items():
            base_val = os.path.join(os.path.basename(temp_armory_dir[k]))
            temp_armory_dir[k] = os.path.join(base_val,v)

    if error is not None:
        with pytest.raises(error):
            check_paths(kwargs, temp_armory_dir)
    else:
        check_paths(kwargs, temp_armory_dir)


