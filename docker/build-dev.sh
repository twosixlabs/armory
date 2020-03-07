version=$(python -c "import armory; print(armory.__version__)")

docker build --file docker/Dockerfile --target armory -t twosixarmory/armory:${version} .

docker build --file docker/tf1/Dockerfile --build-arg armory_version=${version} --target armory-tf1-base -t twosixarmory/tf1-base:${version} .
docker build --file docker/tf1-dev/Dockerfile --build-arg armory_version=${version} --target armory-tf1-dev -t twosixarmory/tf1:${version} .

docker build --file docker/tf2/Dockerfile --build-arg armory_version=${version} --target armory-tf2-base -t twosixarmory/tf2-base:${version} .
docker build --file docker/tf2-dev/Dockerfile --build-arg armory_version=${version} --target armory-tf2-dev -t twosixarmory/tf2:${version} .

docker build --file docker/pytorch/Dockerfile --build-arg armory_version=${version} --target armory-pytorch-base -t twosixarmory/pytorch-base:${version} .
docker build --file docker/pytorch-dev/Dockerfile --build-arg armory_version=${version} --target armory-pytorch-dev -t twosixarmory/pytorch:${version} .
