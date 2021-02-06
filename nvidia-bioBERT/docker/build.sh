#!/bin/bash
docker build --network=host . --rm -t onnxruntime-pytorch-for-bert 2>&1 | tee build.log
