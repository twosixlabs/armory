########## TF 1.15 Base #################

ARG armory_version
FROM twosixarmory/armory:${armory_version} AS armory-tf1-base
RUN /opt/conda/bin/conda install tensorflow-gpu==1.15.0

########## TF 1.15 Release #################

FROM armory-tf1-base AS armory-tf1
ARG armory_version
RUN /opt/conda/bin/pip install armory-testbed==${armory_version}
CMD tail -f /dev/null
