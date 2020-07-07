#!/bin/bash
  
CMD=${1:-/bin/bash}
NV_VISIBLE_DEVICES=${2:-"all"}
DOCKER_BRIDGE=${3:-"host"}

docker run -it --rm \
  --net=$DOCKER_BRIDGE \
  --gpus device=$NV_VISIBLE_DEVICES \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $PWD:/workspace \
  --workdir=/workspace \
onnxruntime-pytorch-for-examples /bin/bash -c "$CMD"
