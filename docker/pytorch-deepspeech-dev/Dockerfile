########## PyTorch 1 Deep Speech Dev #################

ARG armory_version
FROM twosixarmory/pytorch-deepspeech-base:${armory_version} AS armory-pytorch-deepspeech-dev
COPY . /armory_dev/
RUN /opt/conda/bin/pip install /armory_dev/ adversarial-robustness-toolbox==1.4.2 --no-cache-dir
CMD tail -f /dev/null
