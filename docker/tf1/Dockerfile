########## TF 1 Base #################

ARG armory_version
FROM twosixarmory/armory:${armory_version} AS armory-tf1-base
RUN /opt/conda/bin/conda install tensorflow-gpu==1.15.0 && \
    /opt/conda/bin/conda clean --all
WORKDIR /tmp
RUN git clone https://github.com/tensorflow/models.git && cd models/research && git checkout 79354e14a4b41ff9019f4a5ebd12cfa498917182
WORKDIR /tmp/models/research
RUN protoc object_detection/protos/*.proto --python_out=.
RUN cp object_detection/packages/tf1/setup.py .
RUN /opt/conda/bin/pip install .
WORKDIR /workspace

########## TF 1 Release #################

FROM armory-tf1-base AS armory-tf1
ARG armory_version
RUN /opt/conda/bin/pip install armory-testbed==${armory_version} --no-cache-dir
CMD tail -f /dev/null
