########## PyTorch 1 Base #################

ARG armory_version
FROM twosixarmory/armory:${armory_version} AS armory-pytorch-base

# TF used for dataset loading
RUN /opt/conda/bin/conda install tensorflow-gpu==2.2.0 \
 pytorch==1.5 \
 torchvision==0.6.0 \
 cudatoolkit=10.1 -c pytorch && \
    /opt/conda/bin/conda clean --all

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
  warpctc_pytorch

RUN apt-get install -y sox

########## PyTorch 1 Release #################

FROM armory-pytorch-base AS armory-pytorch
ARG armory_version
RUN /opt/conda/bin/pip install armory-testbed==${armory_version} --no-cache-dir
CMD tail -f /dev/null
