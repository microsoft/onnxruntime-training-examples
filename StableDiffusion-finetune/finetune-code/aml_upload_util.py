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
from azureml.core.run import Run
from pathlib import Path

def get_args(raw_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    args = parser.parse_args(raw_args)
    return args

def main():
    args = get_args()
    # upload weights to AML
    # documentation: https://learn.microsoft.com/en-us/python/api/azureml-core/azureml.core.run(class)?view=azure-ml-py
    run = Run.get_context()
    print("Uploading", args.output_dir, "to AzureML...")
    run.upload_folder(name=args.output_dir, path=str(Path(args.output_dir)))
    print("...done!")

if __name__ == "__main__":
    main()
    