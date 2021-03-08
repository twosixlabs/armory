########## TF 2 Dev #################

ARG armory_version
FROM twosixarmory/tf2-base:${armory_version} AS armory-tf2-dev
COPY . /armory_dev/
RUN /opt/conda/bin/pip install /armory_dev/ --no-cache-dir \ 
   git+https://github.com/Trusted-AI/adversarial-robustness-toolbox.git@dev_1.6.0 --no-cache-dir
CMD tail -f /dev/null
