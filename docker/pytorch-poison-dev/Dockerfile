ARG armory_version
FROM twosixarmory/pytorch:${armory_version} AS armory-pytorch
RUN /opt/conda/bin/pip uninstall -y adversarial-robustness-toolbox
RUN /opt/conda/bin/pip install --no-cache-dir \
    git+https://github.com/Trusted-AI/adversarial-robustness-toolbox.git@e0efc7c95282d63e22997f2a3c6d5a050ce0d673
CMD tail -f /dev/null