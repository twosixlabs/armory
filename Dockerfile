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

RUN /opt/conda/bin/pip install \
    tensorflow-datasets==2.0.0 \
    jupyterlab==1.2.6 \
    boto3==1.11.13 \
    adversarial-robustness-toolbox==1.1.1 \
    Pillow==7.0.0

WORKDIR /workspace


########## TF 1.15 Base #################

FROM armory-base AS armory-base-tf1
RUN /opt/conda/bin/conda install tensorflow-gpu==1.15.0

########## TF 1.15 #################

FROM armory-base-tf1 AS armory-tf1
ARG armory_version
RUN /opt/conda/bin/pip install armory-testbed==${armory_version}
CMD tail -f /dev/null

########## TF 2.1 Base #################

FROM armory-base AS armory-base-tf2
RUN /opt/conda/bin/conda install tensorflow-gpu==2.1.0

########## TF 2.1 #################

FROM armory-base-tf2 AS armory-tf2
ARG armory_version
RUN /opt/conda/bin/pip install armory-testbed==${armory_version}
CMD tail -f /dev/null

########## PyTorch 1.4 Base #################

FROM armory-base AS armory-base-pytorch

# TF used for dataset loading
RUN /opt/conda/bin/conda install tensorflow==1.15.0 \
 pytorch==1.4 \
 torchvision==0.5.0 \
 cudatoolkit=10.1 -c pytorch

########## PyTorch 1.4 #################

FROM armory-base-pytorch AS armory-pytorch
ARG armory_version
RUN /opt/conda/bin/pip install armory-testbed==${armory_version}
CMD tail -f /dev/null

########## TF 1.15 Dev #################

FROM armory-base-tf1 AS armory-tf1-dev
COPY . /armory_dev/
RUN /opt/conda/bin/pip install /armory_dev/
CMD tail -f /dev/null

########## TF 2.1 Dev #################

FROM armory-base-tf2 AS armory-tf2-dev
COPY . /armory_dev/
RUN /opt/conda/bin/pip install /armory_dev/
CMD tail -f /dev/null

########## PyTorch 1.4 Dev #################

FROM armory-base-pytorch AS armory-pytorch-dev
COPY . /armory_dev/
RUN /opt/conda/bin/pip install /armory_dev/
CMD tail -f /dev/null

#####################################
