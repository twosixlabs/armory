###########################################################################################
#                           ARMORY Baseline Docker Image
#
# This File contains the baseline image for Armory docker images.  All framework
# based images should inherit from this image using:
#       FROM twosixlabs/armory-baseline AS armory-baseline
#
#
###########################################################################################

FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

# Temporary fix for broken nvidia package checksum
RUN rm -f /etc/apt/sources.list.d/nvidia-ml.list

# Basic Apt-get Bits
RUN apt-get -y -qq update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y \
    wget \
    vim \
    build-essential \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    echo 'alias ll="ls -al"' >> ~/.bashrc

# conda installations ordered by channel

# Conda Install for Channel: pytorch
RUN /opt/conda/bin/conda install \
    pytorch \
    torchvision \
    torchaudio \
    cudatoolkit=11.6 \
    -c pytorch \
    -c conda-forge \
    && /opt/conda/bin/conda clean --all
# pytorch==1.10.2
# torchvision==0.11.3
# torchaudio==0.10.2
# cudatoolkit==1.11.3
# numpy==1.21.2
# Pillow==9.0.1

# Conda Install for Channel: conda-forge
RUN /opt/conda/bin/conda install \
    scikit-learn=1.0 \
    jupyterlab \
    jupyterlab_widgets \
    ipywidgets \
    matplotlib \
    ffmpeg \
    librosa \
    cudnn \
    -c conda-forge \
    && /opt/conda/bin/conda clean --all
# ART requires scikit-learn >=0.22.2,<1.1.0
# cudnn required for tensorflow
# conda-forge ffmpeg comes with libx264 encoder, which the pytorch channel version does not include
#     this encoder is required for video compression defenses (ART) and video exporting
#     future work could migrate this to libopenh264 encoder, which is available in both channels

# Conda Install for packages with no channel
RUN /opt/conda/bin/conda install \
    numba \
    pandas \
    protobuf \
    && /opt/conda/bin/conda clean --all
#    # ART requires numba >= 0.53.1

ENV PATH=/opt/conda/bin:$PATH
# TensorFlow requirement
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/

RUN echo "Installing TensorFlow and ART via pip"
RUN /opt/conda/bin/pip install --no-cache-dir \
    tensorflow-datasets==4.6 \
    tensorflow==2.10 \
    tensorboardx \
    boto3 \
    opencv-python \
    ffmpeg-python \
    pytest \
    loguru \
    docker \
    jsonschema \
    requests \
    pydub \
    transformers \
    adversarial-robustness-toolbox==1.12.1
# transformers is used for the Entailment metric only
# pydub required for ART mp3 defense

RUN echo "Building Armory from local source"
WORKDIR /armory-repo/
COPY setup.py LICENSE MANIFEST.in README.md requirements.txt test-requirements.txt armory-base-requirements.txt /armory-repo/
COPY pytest.ini /armory-repo/
COPY ./tests /armory-repo/tests/
COPY ./scenario_configs /armory-repo/scenario_configs/
COPY ./armory /armory-repo/armory/
COPY ./.git /armory-repo/.git/
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${armory_version} /opt/conda/bin/pip install --use-feature=in-tree-build --no-cache-dir .
RUN armory configure --use-default

WORKDIR /workspace
