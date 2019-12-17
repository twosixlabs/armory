# Armory Container v0.1.0

FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04
MAINTAINER Sean Morgan <sean.morgan@twosixlabs.com>

RUN apt-get -y -qq update && \
    apt-get install -y wget vim build-essential

# Install Conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    echo 'alias ll="ls -al"' >> ~/.bashrc

ENV PATH=/opt/conda/bin:$PATH

# TODO: Temporarily just install TF for testing
RUN /opt/conda/bin/pip install coloredlogs docker tensorflow-datasets tensorflow==1.15
