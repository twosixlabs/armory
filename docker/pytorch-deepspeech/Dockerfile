########## PyTorch 1 Deep Speech Base #################

ARG armory_version
FROM twosixarmory/armory:${armory_version} AS armory-pytorch-deepspeech-base

ENV NUMBA_CACHE_DIR /tmp
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"

RUN /opt/conda/bin/conda install pytorch==1.6 \
  torchvision==0.7.0 \
  cudatoolkit=10.1 -c pytorch && \
    /opt/conda/bin/conda clean --all


RUN /opt/conda/bin/conda install -c conda-forge librosa 
RUN /opt/conda/bin/pip install hydra-core==1.0.6 \
  tensorflow-gpu==2.4.1 \
  python-levenshtein \
  torchaudio==0.6.0 \
  numba==0.52.0 \
  --ignore-installed llvmlite==0.35.0 \
  soundfile \
  sox \
  warpctc-pytorch==0.2.1+torch16.cuda100

########## PyTorch 1 Deep Speech Release #################

FROM armory-pytorch-deepspeech-base as armory-pytorch-deepspeech
ARG armory_version
RUN /opt/conda/bin/pip install --no-cache-dir \ 
    armory-testbed==${armory_version} \
    git+https://github.com/Trusted-AI/adversarial-robustness-toolbox.git@dev_1.6.0 --no-cache-dir
CMD tail -f /dev/null
