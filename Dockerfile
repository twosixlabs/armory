# Armory Containers v0.2.0

########## Base #################

FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04 AS armory-base

RUN apt-get -y -qq update && \
    apt-get install -y wget vim build-essential git curl

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

RUN /opt/conda/bin/pip install tensorflow-datasets requests docker jupyterlab coloredlogs boto3 adversarial-robustness-toolbox==1.1.0

WORKDIR /armory

########## TF 1.15 #################

FROM armory-base AS armory-tf1
RUN /opt/conda/bin/conda install tensorflow==1.15.0
CMD tail -f /dev/null

########## TF 2.1 #################

FROM armory-base AS armory-tf2
RUN /opt/conda/bin/pip install tensorflow==2.1.0
CMD tail -f /dev/null

########## PyTorch 1.4 #################

# TF used for dataset loading
FROM armory-tf1 AS armory-pytorch

RUN /opt/conda/bin/conda install pytorch==1.4 torchvision cudatoolkit=10.1 -c pytorch
CMD tail -f /dev/null

#####################################