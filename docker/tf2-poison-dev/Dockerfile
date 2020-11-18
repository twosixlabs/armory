ARG armory_version
FROM twosixarmory/tf2:${armory_version} AS armory-tf2-poison-dev
RUN /opt/conda/bin/pip uninstall -y adversarial-robustness-toolbox
RUN /opt/conda/bin/pip install --no-cache-dir \
    git+https://github.com/Trusted-AI/adversarial-robustness-toolbox.git@9d5489911590a7aaa90ae2293b28a0a63e995c42
CMD tail -f /dev/null