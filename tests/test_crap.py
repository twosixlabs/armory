def test_one(docker_client):
    docker_client.images.list()
