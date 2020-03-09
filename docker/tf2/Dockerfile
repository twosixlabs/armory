########## TF 2.1 Base #################

ARG armory_version
FROM twosixarmory/armory:${armory_version} AS armory-tf2-base
RUN /opt/conda/bin/conda install tensorflow-gpu==2.1.0

########## TF 2.1 Release #################

FROM armory-tf2-base AS armory-tf2
ARG armory_version
RUN /opt/conda/bin/pip install armory-testbed==${armory_version}
CMD tail -f /dev/null
