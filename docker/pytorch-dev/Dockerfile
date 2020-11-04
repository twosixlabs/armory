########## PyTorch 1 Dev #################

ARG armory_version
FROM twosixarmory/pytorch-base:${armory_version} AS armory-pytorch-dev
COPY . /armory_dev/
RUN /opt/conda/bin/pip install /armory_dev/ adversarial-robustness-toolbox==1.4.2 --no-cache-dir
CMD tail -f /dev/null
