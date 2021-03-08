########## PyTorch 1 Base #################

ARG armory_version
FROM twosixarmory/armory:${armory_version} AS armory-pytorch-base

RUN /opt/conda/bin/conda install pytorch==1.7 \
   torchvision==0.8.0 \ 
   cudatoolkit=11.0 -c pytorch -c conda-forge && \
   /opt/conda/bin/conda clean --all
RUN /opt/conda/bin/pip install tensorflow-gpu==2.4.1 --no-cache-dir

########## PyTorch 1 Release #################

FROM armory-pytorch-base AS armory-pytorch
ARG armory_version

RUN /opt/conda/bin/pip install --no-cache-dir \ 
    armory-testbed==${armory_version} \
    git+https://github.com/Trusted-AI/adversarial-robustness-toolbox.git@dev_1.6.0 

CMD tail -f /dev/null
