#!/bin/bash
  
CMD=${1:-/bin/bash}

docker run -it --rm \
  --gpus all \
  -v $PWD:/workspace \
  --workdir=/workspace \
  ort-training-getting-started $CMD
