########## ARMORY Base #################

FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04 AS armory

# Temporary fix for broken nvidia package checksum
RUN rm -f /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get -y -qq update && \
    apt-get install -y wget vim build-essential git curl

# Install Conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    echo 'alias ll="ls -al"' >> ~/.bashrc

ENV PATH=/opt/conda/bin:$PATH

RUN /opt/conda/bin/pip install --no-cache-dir \
    tensorflow-datasets==3.2.0 \
    jupyterlab==3.0.9 \
    boto3==1.17.20 \
    Pillow==8.2.0 \
    pydub==0.24.1 \
    apache-beam==2.22.0 \
    dill==0.3.1.1 \
    pytest==6.2.2 \
    pandas==1.2.4 \
    ffmpeg-python==0.2.0 \
    tensorboardX==2.4.1


RUN /opt/conda/bin/conda install -c conda-forge ffmpeg==4.2.3 && \
    /opt/conda/bin/conda clean --all

WORKDIR /workspace


########## PyTorch 1 Deep Speech Base #################

ARG armory_version
FROM armory AS armory-pytorch-deepspeech-base

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
    warpctc-pytorch==0.2.1+torch16.cuda102 \
    adversarial-robustness-toolbox==1.9.1 --no-cache-dir


########## PyTorch 1 Deep Speech Dev #################

FROM armory-pytorch-deepspeech-base AS armory-pytorch-deepspeech-dev
ARG armory_version

COPY . /armory_dev/
RUN /opt/conda/bin/pip install /armory_dev/ --no-cache-dir

WORKDIR /workspace
CMD tail -f /dev/null

########## PyTorch 1 Deep Speech Release #################

FROM armory-pytorch-deepspeech-base as armory-pytorch-deepspeech
ARG armory_version

RUN /opt/conda/bin/pip install armory-testbed==${armory_version} --no-cache-dir

WORKDIR /workspace
CMD tail -f /dev/null
