#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Microsoft Corp. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
from pathlib import Path
import json

from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment, BuildContext 
from azure.identity import AzureCliCredential

# run test on automode workspace
ws_config = json.load(open("ws_config.json"))
subscription_id = ws_config["subscription_id"]
resource_group = ws_config["resource_group"]
workspace_name = ws_config["workspace_name"]
compute = ws_config["compute"]
nproc_per_node = ws_config["nproc_per_node"]

def get_args(raw_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", default="stable-diffusion-ort-experiment", help="Experiment name for AML Workspace")

    args = parser.parse_args(raw_args)
    return args

def main(raw_args=None):
    args = get_args(raw_args)

    ml_client = MLClient(
        AzureCliCredential(), subscription_id, resource_group, workspace_name
    )

    root_dir = Path(__file__).resolve().parent
    environment_dir = root_dir / "environment"
    code_dir = root_dir / "finetune-code"

    model = "CompVis/stable-diffusion-v1-4"
    dataset = "lambdalabs/pokemon-blip-captions"
    max_train_steps = 1000
    bs = 1

    pytorch_job = command(
        code=code_dir,  # local path where the code is stored
        command=f"accelerate launch --config_file=accelerate_config.yaml --mixed_precision=fp16  \
                    train_text_to_image.py \
                    --pretrained_model_name_or_path={model} \
                    --dataset_name={dataset} \
                    --use_ema \
                    --resolution=512 --center_crop --random_flip \
                    --train_batch_size={bs} \
                    --gradient_accumulation_steps=4 \
                    --gradient_checkpointing \
                    --max_train_steps={max_train_steps} \
                    --learning_rate=1e-05 \
                    --max_grad_norm=1 \
                    --lr_scheduler=constant --lr_warmup_steps=0 \
                    --output_dir=sd-pokemon-model \
                    && python aml_upload_util.py --output_dir=sd-pokemon-model",
        environment=Environment(build=BuildContext(path=environment_dir)),
        experiment_name=args.experiment_name,
        compute=compute,
        display_name="pytorch-stable-diffusion",
        description=f"Train a vision DNN with PyTorch on the {dataset} dataset.",
        tags={"max_train_steps": str(max_train_steps), "batch_size": str(bs)},
        shm_size="16g"
    )

    print("submitting PyTorch job for " + model)
    pytorch_returned_job = ml_client.create_or_update(pytorch_job)
    print("submitted job")

    pytorch_aml_url = pytorch_returned_job.studio_url
    print("job link:", pytorch_aml_url)

    ort_job = command(
        code=code_dir,  # local path where the code is stored
        command=f"accelerate launch --config_file=accelerate_config.yaml --mixed_precision=fp16  \
                    train_text_to_image.py --ort \
                    --pretrained_model_name_or_path={model}\
                    --dataset_name={dataset} \
                    --use_ema \
                    --resolution=512 --center_crop --random_flip \
                    --train_batch_size={bs} \
                    --gradient_accumulation_steps=4 \
                    --gradient_checkpointing \
                    --max_train_steps={max_train_steps} \
                    --learning_rate=1e-05 \
                    --max_grad_norm=1 \
                    --lr_scheduler=constant --lr_warmup_steps=0 \
                    --output_dir=sd-pokemon-model \
                    && python aml_upload_util.py --output_dir=sd-pokemon-model",
        environment=Environment(build=BuildContext(path=environment_dir)),
        environment_variables={"ORTMODULE_FALLBACK_POLICY": "FALLBACK_DISABLE"},
        experiment_name=args.experiment_name,
        compute=compute,
        display_name= "ort-stable-diffusion",
        description=f"Train a vision DNN with ONNX Runtime on the {dataset} dataset.",
        tags={"max_train_steps": str(max_train_steps), "batch_size": str(bs)},
        shm_size="16g"
    )

    print("submitting ORT job for " + model)
    ort_returned_job = ml_client.create_or_update(ort_job)
    print("submitted job")

    ort_aml_url = ort_returned_job.studio_url
    print("job link:", ort_aml_url)

if __name__ == "__main__":
    main()
