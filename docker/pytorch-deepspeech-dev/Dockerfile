########## PyTorch 1 Deep Speech Dev #################

ARG armory_version
FROM twosixarmory/pytorch-deepspeech-base:${armory_version} AS armory-pytorch-deepspeech-dev
COPY . /armory_dev/
RUN /opt/conda/bin/pip install /armory_dev/ \ 
  git+https://github.com/Trusted-AI/adversarial-robustness-toolbox.git@dev_1.6.0 --no-cache-dir
CMD tail -f /dev/null
