##########################################################################################
#
#                           ARMORY Baseline Docker Image
#
# This File contains the baseline image for Armory docker images.  All framework
# based images should inhereit from this image using:
#       FROM twosixlabs/armory-baseline AS armory-baseline
#
##########################################################################################

FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
# pip Configuration - https://pip.pypa.io/en/stable/user_guide/#config-file
ARG PIP_DISABLE_PIP_VERSION_CHECK=1
ARG PIP_NO_CACHE_DIR=1

ENV PATH="/opt/conda/envs/base/bin:/opt/conda/bin:/usr/local/cuda/lib64:${PATH}"
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/base/lib/

USER root

SHELL ["/bin/bash", "-c"]

WORKDIR /armory-repo

# NOTE: This COPY command is filtered using the `.dockerignore` file
#       in the root of the repo.
COPY ./ /armory-repo

# Temporary fix for broken nvidia package checksum
# RUN rm -f /etc/apt/sources.list.d/nvidia-ml.list

# Basic Apt-get Bits
RUN apt-get -y -qq update && \
    apt-get install -y \
        sudo \
        vim \
        build-essential \
        git \
        wget \
        curl \
        # Needed for cv2 (opencv-python) and ffmpeg-python
        libgl1-mesa-glx \
        # libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# Install Conda
# NOTE: with conda version 5, will need to set channel priority to flexible (as strict will become default)
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda                             && \
    rm ~/miniconda.sh                                                     && \
    conda install --yes --channel conda-forge --name base nomkl mamba
    # conda config --set auto_activate_base true                          && \
    # ln -s $(which python3) /usr/local/bin/python                        && \
    # conda init bash


# NOTE: using mamba because conda fails when trying to solve for environment
RUN mamba env update -f environment.yml -n base --prune && \
    armory configure --use-default && \
    mamba clean --all


# Clean Up
# RUN apt-get purge $( dpkg --list | grep -P -o "linux-image-\d\S+"| head -n-1 ) -y && \
#     apt-get autoremove -y && \
#     apt-get autoclean -y && \
#     apt-get clean -y && \
#     conda clean -afy


# Create armory user  to avoid running as root
RUN useradd \
        --user-group \
        --no-log-init \
        --home-dir /tmp \
        --shell /bin/bash \
        --uid 1000 \
        user && \
    chown -R --from=root user /workspace && \
    chown -R --from=root user /armory-repo

VOLUME ["/workspace", "/armory-repo"]

# Jupyter
EXPOSE 8888

STOPSIGNAL SIGQUIT

WORKDIR /workspace


# ------------------------------------------------------------------
# DEVELOPER NOTES:
# ------------------------------------------------------------------
# NOTE:
#  - pytorch-lightning >= 1.5.0 will break Deep Speech 2
#  - torchmetrics >= 0.8.0 will break pytorch-lightning 1.4
#  - hydra-lightning installs omegaconf
#  - google-cloud-storage needed for checkpoint.py import
#  - only sox python bindings are installed; underlying sox binaries not needed
# NOTE: Listed dependencies of PyTorch Deep Speech 2, but do not appear
#       to be used for inference (only for training), they are not installed:
#         - torchelastic
#         - wget
#         - flask
#         - fairscale
# ------------------------------------------------------------------
