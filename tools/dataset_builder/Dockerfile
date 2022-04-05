###########################################################################################
#                           ARMORY Baseline Docker Image
#
# This file contains the a helper image for building armory datasets.
#
#
###########################################################################################


FROM tensorflow/tensorflow

WORKDIR dataset_builder
COPY ./requirements.txt /dataset_builder/
RUN pip install -r /dataset_builder/requirements.txt

COPY ./*.py /dataset_builder/
COPY ./build_classes /dataset_builder/build_classes/
