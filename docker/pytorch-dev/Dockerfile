########## PyTorch 1.4 Dev #################

ARG armory_version
FROM twosixarmory/pytorch-base:${armory_version} AS armory-pytorch-dev
COPY . /armory_dev/
RUN /opt/conda/bin/pip install /armory_dev/ --no-cache-dir
CMD tail -f /dev/null
