import pytest

# Mark all tests in this file as `unit`
pytestmark = pytest.mark.unit


@pytest.mark.docker_required
def test_one(docker_client):
    docker_client.images.list()
