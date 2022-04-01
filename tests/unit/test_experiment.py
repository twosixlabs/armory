import pytest
from armory.experiment import Experiment


# @pytest.mark.parametrize("file", ["tests/scenarios/pytorch/image_classification.json"])
# def test_scenario_read(file):
#     with open(file, "r") as f:
#         exp = Experiment.parse_raw(f.read())
#     print(exp)
#     assert exp.model.fit is True
#
@pytest.mark.parametrize("file", ["tests/scenarios/pytorch/image_classification.json"])
def test_scenario_read(file):
    with open(file, "r") as f:
        exp = Experiment.parse_raw(f.read())
    print(exp)
    assert exp.model.fit is True


@pytest.mark.parametrize("file", ["tests/scenarios/pytorch/image_classification.json"])
def test_scenario_load(file):
    exp = Experiment.load(file)
    print(type(exp))
