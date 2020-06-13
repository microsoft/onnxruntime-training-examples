#!/bin/bash
  
CMD=${1:-/bin/bash}
NV_VISIBLE_DEVICES=${2:-"all"}
DOCKER_BRIDGE=${3:-"host"}
DATA_DIR=<replace-with-path-to-training-data>

docker run -it --rm \
  --gpus device=$NV_VISIBLE_DEVICES \
  --net=$DOCKER_BRIDGE \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $DATA_DIR:/data/ \
  -v $PWD:/workspace/ \
  -v $PWD/results:/results \
  --workdir=/workspace/transformers \
  onnxruntime-gpt $CMD
