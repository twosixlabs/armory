version=$(python -m armory --version)

docker build --file docker/Dockerfile --target armory -t twosixarmory/armory:${version} .

docker build --file docker/tf1/Dockerfile --build-arg armory_version=${version} --target armory-tf1-base -t twosixarmory/tf1-base:${version} .
docker build --file docker/tf1-dev/Dockerfile --build-arg armory_version=${version} --target armory-tf1-dev -t twosixarmory/tf1:${version} .
