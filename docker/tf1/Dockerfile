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

RUN /opt/conda/bin/conda install \
    tensorflow-gpu==1.15.0 \
    numpy==1.19.2 \
    h5py==2.10.0 \
    pandas==1.3.3 \
    jupyterlab==3.1.7 \
    boto3==1.18.21 \
    dill=0.3.4 \
    pytest==6.2.4

RUN /opt/conda/bin/conda install -c conda-forge ffmpeg==4.2.3

RUN /opt/conda/bin/pip install --no-cache-dir \
    tensorflow-datasets==3.2.0 \
    Pillow==8.2.0 \
    pydub==0.24.1 \
    apache-beam==2.22.0 \
    opencv-python==4.5.1.48 \
    ffmpeg-python==0.2.0 \
    tensorboardX==2.4.1



# the numpy install was moved above in order to group conda installs together before pip installs
# as recommended by https://www.anaconda.com/blog/using-pip-in-a-conda-environment
# the numpy version is pinned until a TF issue is resolved https://github.com/tensorflow/models/issues/9706
# pip packages that exist in conda have been moved to the conda install above to allow the
# dependency resolution to cover as many packages as possible


WORKDIR /workspace

########## TF 1 Base #################

ARG armory_version
FROM armory AS armory-tf1-base


WORKDIR /tmp
RUN git clone https://github.com/tensorflow/models.git && cd models/research && git checkout 79354e14a4b41ff9019f4a5ebd12cfa498917182
WORKDIR /tmp/models/research
RUN protoc object_detection/protos/*.proto --python_out=.
RUN cp object_detection/packages/tf1/setup.py .
RUN /opt/conda/bin/pip install .
RUN /opt/conda/bin/pip install --no-cache-dir adversarial-robustness-toolbox==1.9.1


WORKDIR /workspace

########## TF 1 Dev #################

FROM armory-tf1-base AS armory-tf1-dev
ARG armory_version

COPY . /armory_dev/
RUN /opt/conda/bin/pip install /armory_dev/ --no-cache-dir

WORKDIR /workspace
CMD tail -f /dev/null

########## TF 1 Release #################

FROM armory-tf1-base AS armory-tf1
ARG armory_version

RUN /opt/conda/bin/pip install armory-testbed==${armory_version} --no-cache-dir

WORKDIR /workspace
CMD tail -f /dev/null
