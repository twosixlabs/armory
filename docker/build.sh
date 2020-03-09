version=$(python -m armory --version)

docker build --file docker/Dockerfile --target armory -t twosixarmory/armory:${version} .
docker build --file docker/tf1/Dockerfile --build-arg armory_version=${version} --target armory-tf1 -t twosixarmory/tf1:${version} .
docker build --file docker/tf2/Dockerfile --build-arg armory_version=${version} --target armory-tf2 -t twosixarmory/tf2:${version} .
docker build --file docker/pytorch/Dockerfile --build-arg armory_version=${version} --target armory-pytorch -t twosixarmory/pytorch:${version} .
