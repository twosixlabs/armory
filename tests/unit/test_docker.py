import pytest


@pytest.mark.docker_required
def test_one(docker_client):
    docker_client.images.list()
