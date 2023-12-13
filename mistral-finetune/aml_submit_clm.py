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

    parser.add_argument("--experiment_name", default="MISTRAL-7B-ORT-CLM-Stage2-Experiment", help="Experiment name for AML Workspace")

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

    model = "mistralai/Mistral-7B-v0.1"
    num_train_epochs = 5
    bsz = 1
    max_steps = 500

    # https://huggingface.co/datasets/dair-ai/emotion
    dataset_name = "databricks/databricks-dolly-15k"
    dataset_config_name = "split"
    text_column_name = "text"
    label_column_name = "label"

    pytorch_job = command(
        code=code_dir,  # local path where the code is stored
        command=f"torchrun --nproc_per_node {nproc_per_node} run_clm.py \
            --model_name_or_path {model} \
            --dataset_name {dataset_name} \
            --per_device_train_batch_size {bsz} \
            --do_train \
            --num_train_epochs {num_train_epochs} \
            --output_dir results --overwrite_output_dir \
            --save_strategy 'no' \
            --fp16 --max_steps {max_steps} \
            --gradient_accumulation_steps 1 \
            --learning_rate 0.00001 \
            --adam_beta1 0.9 \
            --adam_beta2 0.999 \
            --adam_epsilon 1e-8 \
            --deepspeed zero_stage_2.json",
        environment=Environment(build=BuildContext(path=environment_dir)),
        experiment_name="MISTRAL-7B-Pytorch-CLM-LORA-Stage2-Experiment",
        compute=compute,
        display_name=model.replace(
            "mistral-ai",
            f"pytorch+LoRA+DS2-{bsz}"
        ),
        description=f"Train Mistral AI 7B using PyTorch",
        tags={"model": model,
              "bsz": bsz,
              "dataset_name": dataset_name},
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
            --per_device_train_batch_size {bsz} \
            --do_train \
            --num_train_epochs {num_train_epochs} \
            --output_dir results --overwrite_output_dir \
            --save_strategy 'no' \
            --fp16 --max_steps {max_steps} \
            --gradient_accumulation_steps 1 \
            --learning_rate 0.00001 \
            --adam_beta1 0.9 \
            --adam_beta2 0.999 \
            --adam_epsilon 1e-8 \
            --deepspeed zero_stage_2.json",
        environment=Environment(build=BuildContext(path=environment_dir)),
        environment_variables={"APPLY_ORT": "True",
                               "ORTMODULE_FALLBACK_POLICY": "FALLBACK_DISABLE"},
        experiment_name="MISTRAL-7B-ORT-CLM-Stage2-Experiment",
        compute=compute,
        display_name=model.replace(
            "mistral-ai",
            f"ort+LoRA+DS2-{bsz}"
        ),
        description=f"Train Hugging Face's Open-LLaMA using ONNX Runtime",
        tags={"model": model,
              "bsz": bsz,
              "dataset_name": dataset_name},
        shm_size="16g"
    )

    print("submitting ORT job for " + model)
    ort_returned_job = ml_client.create_or_update(ort_job)
    print("submitted job")

    ort_aml_url = ort_returned_job.studio_url
    print("job link:", ort_aml_url)

if __name__ == "__main__":
    main()
