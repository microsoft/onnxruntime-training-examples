#!/bin/bash
  
CMD=${1:-/bin/bash}
NV_VISIBLE_DEVICES=${2:-"all"}
DOCKER_BRIDGE=${3:-"host"}

docker run -it --rm \
  --gpus device=$NV_VISIBLE_DEVICES \
  --net=$DOCKER_BRIDGE \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v <replace-with-path-to-phase1-hdf5-training-data>:/data/128 \
  -v <replace-with-path-to-phase2-hdf5-training-data>:/data/512 \
  -v $PWD:/workspace/bert \
  --workdir=/workspace/bert \
  onnxruntime-pytorch-for-bert $CMD
