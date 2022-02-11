########## ARMORY Base #################

FROM nvidia/cuda:11.2.1-cudnn8-runtime-ubuntu20.04 AS armory

# Temporary fix for broken nvidia package checksum
RUN rm -f /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get -y -qq update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y wget vim build-essential git curl libgl1-mesa-glx libglib2.0-0

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
    opencv-python==4.5.1.48 \
    pandas==1.2.4 \
    ffmpeg-python==0.2.0 \
    numba==0.53.1 \
    tensorboardX==2.4.1

RUN /opt/conda/bin/conda install -c conda-forge ffmpeg==4.2.3 && \
    /opt/conda/bin/conda clean --all

WORKDIR /workspace

########## PyTorch 1 Base #################

ARG armory_version
FROM armory AS armory-pytorch-base

RUN /opt/conda/bin/conda install pytorch==1.7 \
    torchvision==0.8.0 \
    cudatoolkit=11.0 -c pytorch -c conda-forge && \
    /opt/conda/bin/conda clean --all
RUN /opt/conda/bin/pip install --no-cache-dir \
    tensorflow-gpu==2.4.1 \
    adversarial-robustness-toolbox==1.9.1

########## PyTorch 1 Dev #################

FROM armory-pytorch-base AS armory-pytorch-dev
ARG armory_version

COPY . /armory_dev/
RUN /opt/conda/bin/pip install /armory_dev/ --no-cache-dir

WORKDIR /workspace
CMD tail -f /dev/null

########## PyTorch 1 Release #################

FROM armory-pytorch-base AS armory-pytorch
ARG armory_version

RUN /opt/conda/bin/pip install armory-testbed==${armory_version} --no-cache-dir

WORKDIR /workspace
CMD tail -f /dev/null
