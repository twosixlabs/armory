########## PyTorch 1 Deep Speech Base #################

ARG armory_version
FROM twosixarmory/armory:${armory_version} AS armory-pytorch-deepspeech-base

ENV NUMBA_CACHE_DIR /tmp
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"

RUN /opt/conda/bin/conda install tensorflow-gpu==2.2.0 \
 pytorch==1.4 \
 torchvision==0.5.0 \
 cudatoolkit=10.1 -c pytorch && \
    /opt/conda/bin/conda clean --all

RUN /opt/conda/bin/conda install -c conda-forge librosa 
RUN /opt/conda/bin/pip install hydra-core==1.0.0rc1 \
  python-levenshtein \
  torchaudio==0.4 \
  numba==0.43.0 \
  --ignore-installed llvmlite==0.32.1 \
  soundfile \
  sox \
  warpctc-pytorch==0.2.1+torch14.cuda100

########## PyTorch 1 Deep Speech Release #################

FROM armory-pytorch-deepspeech-base as armory-pytorch-deepspeech
ARG armory_version
RUN /opt/conda/bin/pip install --no-cache-dir \ 
    armory-testbed==${armory_version} \
    adversarial-robustness-toolbox==1.4.2
CMD tail -f /dev/null
