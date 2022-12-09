##########################################################################################
#
#                           ARMORY Docker Image
#
# This file contains the instructions to build the Armory docker image. All framework
# based images should inhereit from this image using:
#       FROM twosixlabs/armory AS armory-baseline
#
#            ~~! Please remove/modify the following lines as updates are made to the image. !~~
# Notes:
#   [1] https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/gpu.Dockerfile
#   [2] https://bcourses.berkeley.edu/courses/1478831/pages/glibcxx-missing
#   [3] https://docs.ycrc.yale.edu/clusters-at-yale/guides/conda/#mamba-the-conda-alternative
#   [4] https://github.com/tensorflow/models/tree/master/research/object_detection
#   [5]
#
##########################################################################################
ARG UBUNTU_VERSION=20.04
ARG CUDNN_VERSION=8
ARG CUDA_VERSION=11.6.2
ARG CUDA=11.6
ARG LIBNVINFER_MAJOR_VERSION=8


FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS base

ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_DISABLE_PIP_VERSION_CHECK=1
ARG PIP_NO_CACHE_DIR=1
ARG USER="user"
ARG PASSWORD="armory"
ARG UID=1000
ARG GID=$UID

ENV PATH="/opt/conda/envs/base/bin:/opt/conda/bin:/usr/local/cuda/lib64:${PATH}"
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/base/lib/


USER root

SHELL ["/bin/bash", "-c"]

WORKDIR /armory-repo

# NOTE: This COPY command is filtered using the `.dockerignore` file
#       in the root of the repo.
COPY ./ /armory-repo

# Basic Apt-get Bits
RUN apt-get -y -qq update && \
    apt-get install -y       \
        build-essential      \
        curl                 \
        git                  \
        sudo                 \
        vim                  \
        wget                 \
        # Needed for cv2 (opencv-python) and ffmpeg-python
        libgl1-mesa-glx \
        libglib2.0-0

# Install Conda
# NOTE: with conda version 5, will need to set channel priority to flexible (as strict will become default)
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda                             && \
    rm ~/miniconda.sh                                                     && \
    conda install --yes --channel conda-forge --name base nomkl mamba     && \
    ln -s $(which python3) /usr/local/bin/python                          && \
    conda init bash


# NOTE: using mamba because conda fails when trying to solve for environment
RUN mamba env update -f environment.yml -n base --prune && \
    armory configure --use-default

# Clean Up
RUN apt-get purge $( dpkg --list | grep -P -o "linux-image-\d\S+"| head -n-1 ) -y && \
    apt-get autoremove -y && \
    apt-get autoclean -y && \
    apt-get clean -y && \
    conda clean -afy


WORKDIR /workspace

# Create armory user, set $HOME to /tmp, and add to sudoers
RUN useradd                                      \
        --user-group                             \
        --no-log-init                            \
        --home-dir /tmp                          \
        --shell /bin/bash                        \
        --uid ${UID}                             \
        ${USER}                               && \
    usermod -aG sudo ${USER}                  && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" >      \
        /etc/sudoers.d/${USER}                && \
    cp /etc/skel/{.bashrc,.profile} /tmp/     && \
    chmod 0440 /etc/sudoers.d/${USER}         && \
    chown -R --from=root ${USER} /workspace   && \
    chown -R --from=root ${USER} /armory-repo


USER ${USER}

VOLUME ["/workspace", "/armory-repo"]

# Jupyter
EXPOSE 8888

STOPSIGNAL SIGQUIT
