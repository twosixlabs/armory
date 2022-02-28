import torch
import pytest

# this test is marked xfail because I don't care if it is present or not, I just want to know
# in the pytest summary this appears as X when present and lower-case x when not
@pytest.mark.xfail
def test_cuda_present():
    assert torch.cuda.is_available()
