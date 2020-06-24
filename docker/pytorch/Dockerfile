########## PyTorch 1 Base #################

ARG armory_version
FROM twosixarmory/armory:${armory_version} AS armory-pytorch-base

# TF used for dataset loading
RUN /opt/conda/bin/conda install tensorflow==1.15.0 \
 pytorch==1.5 \
 torchvision==0.6.0 \
 cudatoolkit=10.1 -c pytorch && \
    /opt/conda/bin/conda clean --all

RUN pip install --no-cache-dir dareblopy==0.0.3

########## PyTorch 1 Release #################

FROM armory-pytorch-base AS armory-pytorch
ARG armory_version
RUN /opt/conda/bin/pip install armory-testbed==${armory_version} --no-cache-dir
CMD tail -f /dev/null
