########## PyTorch 1 Deep Speech Base #################

ARG armory_version

FROM twosixarmory/pytorch:${armory_version} AS armory-pytorch-deepspeech-dev

RUN /opt/conda/bin/conda install -c conda-forge librosa

#ENV NUMBA_CACHE_DIR /tmp
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
# google-cloud-storage needed for checkpoint.py
# only sox python bindings are installed; underlying sox binaries not needed


#torchelastic
#wget
#flask
#sox
#fairscale



#RUN /opt/conda/bin/pip install hydra-core==1.0.6 \
#    tensorflow-gpu==2.4.1 \
#    python-levenshtein \
#    --ignore-installed llvmlite==0.35.0 \
#    soundfile \
#    sox \
#    warpctc-pytorch==0.2.1+torch16.cuda102 \

WORKDIR /workspace
CMD tail -f /dev/null
