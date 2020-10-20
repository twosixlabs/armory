########## PyTorch 1 Deep Speech Base #################

ARG armory_version
FROM twosixarmory/pytorch-base:${armory_version} AS armory-pytorch-deepspeech-base

ENV NUMBA_CACHE_DIR /tmp
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"

RUN /opt/conda/bin/conda install -c conda-forge librosa 
RUN /opt/conda/bin/pip install hydra-core==1.0.0rc1 \
  python-levenshtein \
  torchaudio \
  numba==0.43.0 \
  --ignore-installed llvmlite==0.32.1 \
  soundfile \
  sox \
  warpctc-pytorch==0.2.1+torch16.cuda100

########## PyTorch 1 Deep Speech Release #################

FROM pytorch-deepspeech-base as armory-pytorch-deepspeech
ARG armory_version
RUN /opt/conda/bin/pip install armory-testbed==${armory_version} --no-cache-dir
CMD tail -f /dev/null
