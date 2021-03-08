########## ARMORY Base #################

FROM nvidia/cuda:11.2.1-cudnn8-runtime-ubuntu20.04 AS armory

# Temporary fix for broken nvidia package checksum
RUN rm -f /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get -y -qq update && \
    apt-get install -y wget vim build-essential git curl

# Install Conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh && \
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
    Pillow==8.1.1 \
    pydub==0.24.1 \
    apache-beam==2.22.0 \
    dill==0.3.1.1 \
    pytest==6.2.2 \

RUN /opt/conda/bin/conda install -c conda-forge ffmpeg==4.2.3 && \
    /opt/conda/bin/conda clean --all

WORKDIR /workspace
