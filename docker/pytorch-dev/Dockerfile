########## PyTorch 1 Dev #################

ARG armory_version
FROM twosixarmory/pytorch-base:${armory_version} AS armory-pytorch-dev
COPY . /armory_dev/
RUN /opt/conda/bin/pip install /armory_dev/ --no-cache-dir \
   git+https://github.com/Trusted-AI/adversarial-robustness-toolbox.git@dev_1.6.0 

CMD tail -f /dev/null
