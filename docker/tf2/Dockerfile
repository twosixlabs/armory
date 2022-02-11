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
    tensorboardX==2.4.1


RUN /opt/conda/bin/conda install -c conda-forge ffmpeg==4.2.3 && \
    /opt/conda/bin/conda clean --all

WORKDIR /workspace

########## TF 2 Base #################

ARG armory_version
FROM armory AS armory-tf2-base

RUN /opt/conda/bin/conda install protobuf==3.14.0 && \
    /opt/conda/bin/conda clean --all
RUN /opt/conda/bin/pip install tensorflow-gpu==2.4.1 --no-cache-dir
WORKDIR /tmp
RUN git clone https://github.com/tensorflow/models.git && cd models/research && git checkout 79354e14a4b41ff9019f4a5ebd12cfa498917182
WORKDIR /tmp/models/research
RUN protoc object_detection/protos/*.proto --python_out=.
RUN cp object_detection/packages/tf2/setup.py .
RUN /opt/conda/bin/pip install .
RUN /opt/conda/bin/pip install --no-cache-dir adversarial-robustness-toolbox==1.9.1

WORKDIR /workspace

########## TF 2 Dev #################

FROM armory-tf2-base AS armory-tf2-dev
ARG armory_version

COPY . /armory_dev/
RUN /opt/conda/bin/pip install /armory_dev/ --no-cache-dir

WORKDIR /workspace
CMD tail -f /dev/null

########## TF 2 Release #################

FROM armory-tf2-base AS armory-tf2
ARG armory_version

RUN /opt/conda/bin/pip install armory-testbed==${armory_version} --no-cache-dir

WORKDIR /workspace
CMD tail -f /dev/null
