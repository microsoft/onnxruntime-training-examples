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
import os

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

    parser.add_argument("--experiment_name", default="Phi-3-ORT-CLM-Stage2-Experiment", help="Experiment name for AML Workspace")

    args = parser.parse_args(raw_args)
    return args

def main(raw_args=None):
    args = get_args(raw_args)

    ml_client = MLClient(
        AzureCliCredential(), subscription_id, resource_group, workspace_name
    )

    root_dir = Path(__file__).resolve().parent
    environment_dir = root_dir / "environment"
    code_dir = root_dir / "finetune-clm"

    model = "microsoft/Phi-3-mini-4k-instruct"
    num_train_epochs = 5
    bsz = 1
    max_steps = -1
    block_size = 64

    dataset_name = "wikitext"
    dataset_config_name = "wikitext-2-raw-v1"
    text_column_name = "text"
    label_column_name = "label"

    pytorch_job = command(
        code=code_dir,  # local path where the code is stored
        command=f"torchrun --nproc_per_node {nproc_per_node} run_clm.py \
            --model_name_or_path {model} \
            --dataset_name {dataset_name} \
            --dataset_config_name {dataset_config_name} \
            --do_train \
            --save_strategy 'no' \
            --per_device_train_batch_size {bsz} \
            --num_train_epochs {num_train_epochs} \
            --output_dir results --overwrite_output_dir \
            --fp16 --max_steps {max_steps} \
            --block_size {block_size} \
            --deepspeed zero_stage_2.json \
            --evaluation_strategy epoch --remove_unused_columns False --save_strategy no \
            --report_to tensorboard --logging_steps 100",
        environment=Environment(build=BuildContext(path=environment_dir)),
        experiment_name="Phi-3-Pytorch-CLM-LORA-Stage2-Experiment",
        compute=compute,
        display_name=model.replace(
            "microsoft/phi-2",
            f"pytorch+DS2_lora-{bsz}"
        ),
        description=f"Finetune HuggingFace's Phi-3 using PyTorch and transformers branch",
        tags={"model": model,
              "bsz": bsz,
              "dataset_name": dataset_name,
              "block_size": block_size},
        shm_size="16g"
    )
    
    print("submitting PyTorch job for " + model)
    pytorch_returned_job = ml_client.create_or_update(pytorch_job)
    print("submitted job")

    pytorch_aml_url = pytorch_returned_job.studio_url
    print("job link:", pytorch_aml_url)

    ort_job = command(
        code=code_dir,  # local path where the code is stored
        command=f"torchrun --nproc_per_node {nproc_per_node} run_clm.py \
            --model_name_or_path {model} \
            --dataset_name {dataset_name} \
            --dataset_config_name {dataset_config_name} \
            --do_train \
            --save_strategy 'no' \
            --per_device_train_batch_size {bsz} \
            --num_train_epochs {num_train_epochs} \
            --output_dir results --overwrite_output_dir \
            --fp16 --max_steps {max_steps} \
            --block_size 2048 \
            --deepspeed zero_stage_2.json \
            --evaluation_strategy no --remove_unused_columns False",
        environment=Environment(build=BuildContext(path=environment_dir)),
        environment_variables={"APPLY_ORT": "True",
                               "ORTMODULE_FALLBACK_POLICY": "FALLBACK_DISABLE",
                               "ORTMODULE_DEEPCOPY_BEFORE_MODEL_EXPORT": "0"},
        experiment_name="Phi-3-ORT-CLM-Stage2-Experiment",
        compute=compute,
        display_name=model.replace(
            "microsoft/phi-3",
            f"ort+DS2+Lora-{bsz}"
        ),
        description=f"Finetune HuggingFace's Phi-3 using ONNX Runtime and transformers branch",
        tags={"model": model,
              "bsz": bsz,
              "dataset_name": dataset_name,
              "block_size": block_size},
        shm_size="16g"
    )

    print("submitting ORT job for " + model)
    ort_returned_job = ml_client.create_or_update(ort_job)
    print("submitted job")

    ort_aml_url = ort_returned_job.studio_url
    print("job link:", ort_aml_url)

if __name__ == "__main__":
    main()
