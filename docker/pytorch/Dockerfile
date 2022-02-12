########## ARMORY Base #################

FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04 AS armory-pytorch-base

# Temporary fix for broken nvidia package checksum
# RUN rm -f /etc/apt/sources.list.d/nvidia-ml.list

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
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    echo 'alias ll="ls -al"' >> ~/.bashrc

ENV PATH=/opt/conda/bin:$PATH

RUN /opt/conda/bin/conda install \
    pytorch=1.10 \
    torchvision \
    torchaudio \
    cudatoolkit=11.3 \
    -c pytorch \
    && /opt/conda/bin/conda clean --all
# pytorch==1.10.2 \
# torchvision==0.11.3 \
# torchaudio==0.10.2 \
# cudatoolkit==11.3.1 \
# ffmpeg==4.3 \
# numpy==1.21.2 \
# pillow==8.4.0 \
# tqdm==4.61.2 \
# requests==2.25.1 \

RUN /opt/conda/bin/conda install \
    scikit-learn=1.0 \
    jupyterlab \
    matplotlib \
    -c conda-forge \
    && /opt/conda/bin/conda clean --all
    # ART requires scikit-learn >=0.22.2,<1.1.0
# scikit-learn==1.0.2 \
# scipy==1.7.3 \
# jupyterlab==3.2.9 \ 
# jsonschema==4.4.0 \
# six==1.16.0 \
# setuptools==52.0.0 \

#RUN /opt/conda/bin/conda install \
#    opencv \
#    -c conda-forge \
#    && /opt/conda/bin/conda clean --all
##    opencv \
#
#RUN /opt/conda/bin/conda install \
#    tensorflow \
#    tensorflow-datasets \
#    tensorboardx \
#    -c conda-forge \
#    && /opt/conda/bin/conda clean --all


RUN /opt/conda/bin/conda install \
    numba \
    -c numba \
    && /opt/conda/bin/conda clean --all
    # ART requires numba >= 0.53.1

RUN /opt/conda/bin/conda install \
    pandas \
    -c anaconda \
    && /opt/conda/bin/conda clean --all

RUN /opt/conda/bin/pip install --no-cache-dir \
    tensorflow-datasets \
    tensorflow \
    tensorboardX \
    boto3 \
    opencv-python \
    ffmpeg-python \
    pytest \
    pymongo \
    coloredlogs \
    docker \
    adversarial-robustness-toolbox==1.9.1

#    pydub==0.24.1 \
#    dill==0.3.1.1 \

#RUN /opt/conda/bin/pip install --no-cache-dir \
#     pytest \
#     pymongo \
#     coloredlogs \
#     docker \
#     boto3 \
#     adversarial-robustness-toolbox==1.9.1 \
# pytest==6.2.2 \
     

#RUN /opt/conda/bin/pip install --no-cache-dir \
#    tensorflow-datasets==3.2.0 \
#    boto3==1.17.20 \
#    pydub==0.24.1 \
#    dill==0.3.1.1 \
#    opencv-python==4.5.1.48 \
#    ffmpeg-python==0.2.0 \
#    tensorboardX==2.4.1 \
#    tensorflow-gpu==2.4.1 \
#    adversarial-robustness-toolbox==1.9.1

WORKDIR /workspace

########## PyTorch 1 Dev #################

FROM armory-pytorch-base AS armory-pytorch-dev
ARG armory_version

#COPY . /armory_dev/
#RUN /opt/conda/bin/pip install /armory_dev/ --no-cache-dir

WORKDIR /workspace
CMD tail -f /dev/null

########## PyTorch 1 Release #################

FROM armory-pytorch-base AS armory-pytorch
ARG armory_version

RUN /opt/conda/bin/pip install armory-testbed==${armory_version} --no-cache-dir

WORKDIR /workspace
CMD tail -f /dev/null
