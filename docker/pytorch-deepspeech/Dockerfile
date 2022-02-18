########## PyTorch 1 Deep Speech Base #################

ARG armory_version

FROM twosixarmory/pytorch:${armory_version} AS armory-pytorch-deepspeech-dev

# NOTE: installed after pip installs in base container,
#       but doesn't seem to cause issues at the moment
RUN /opt/conda/bin/conda install -c conda-forge librosa

# TODO: determine if this environment setup is needed
#ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"

RUN /opt/conda/bin/pip install --no-cache-dir \
    pytorch-lightning==1.4.* \
    git+https://github.com/romesco/hydra-lightning/\#subdirectory=hydra-configs-pytorch-lightning \
    python-levenshtein \
    hydra-core \
    google-cloud-storage \
    sox
# pytorch-lightning >= 1.5.0 will break Deep Speech 2
# hydra-lightning installs omegaconf
# google-cloud-storage needed for checkpoint.py import
# only sox python bindings are installed; underlying sox binaries not needed

# These are listed as dependencies in PyTorch Deep Speech 2, but they do not appear to be used for inference (only for training), so we do not install them:
#torchelastic
#wget
#flask
#fairscale

# NOTE: target this back to ART once https://github.com/Trusted-AI/adversarial-robustness-toolbox/pull/1550 is merged into a release branch
RUN /opt/conda/bin/pip install --no-cache-dir \
    git+https://github.com/davidslater/adversarial-robustness-toolbox.git@1.9.1-deepspeech-fix

WORKDIR /workspace
CMD tail -f /dev/null
