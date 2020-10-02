# coding=utf-8
# Copyright (c) 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2020 Microsoft Corporation. All rights reserved.

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
# limitations under the License.

import os
import argparse
import textwrap

def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError('%s must be positive'.format(value))
    return ivalue

def positive_float(value):
    fvalue = float(value)
    if fvalue <= 0.0:
        raise argparse.ArgumentTypeError('%s must be positive'.format(value))
    return fvalue

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # model specific
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The BERT model configuration as json file.")

    # data specific
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="Path to directory containing .hdf5 samples.")

    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=positive_int,
        help=textwrap.dedent(
            """
            The maximum input sequence length after WordPiece tokenization.
            Sentence pairs requiring more tokens than this will be truncated.
            Sentence pairs requiring fewer tokens than this will be padded.
            """))

    parser.add_argument(
        "--max_predictions_per_seq",
        default=80,
        type=positive_int,
        help="The maximum number of masked tokens in an input sequence.")

    # training loop specific
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="The seed used to initialize random number generators.")

    parser.add_argument(
        "--gpu_feed_batch_size",
        default=32,
        type=positive_int,
        help=textwrap.dedent(
            """
            The number of samples used in one forward/backward pass (per GPU).
            The effective batch size will be (batch size) x (gradient accumulation steps) x (number GPUs).
            The effective batch size is the total samples involved in a single weight update.
            """))

    parser.add_argument(
        '--gradient_accumulation_passes',
        type=positive_int,
        default=1,
        help=textwrap.dedent(
            """
            Number of training backward passes before performing a weight update.
            Gradients are accumulated between weight updates.
            """))

    parser.add_argument(
        '--allreduce_post_accumulation',
        default=False,
        action='store_true',
        help="Whether to do all reduces during gradient accumulation steps.")            

    parser.add_argument(
        "--max_epochs",
        default=None,
        type=positive_float,
        help="Total number of training epochs to perform.")

    parser.add_argument(
        "--max_steps",
        default=None,
        type=positive_int,
        help="Total number of training weight update steps to perform.")

    # learning rate scheduler
    parser.add_argument("--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for optimizer.")

    parser.add_argument("--warmup_proportion",
        default=0.01,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
                "E.g., 0.1 = 10%% of training.")

    # output specific
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=False,
        help="Path for any logs, checkpoints, and final onnx model")

    parser.add_argument(
        '--num_steps_to_smooth_output',
        type=positive_int,
        default=5,
        help="Number of steps to average loss, timing, and throughput.")
    
    parser.add_argument(
        '--debug',
        default=False,
        action='store_true',
        help='Enable verbose logging for debugging')
        
    # checkpoint specific
    parser.add_argument(
        "--init_checkpoint",
        default=None,
        type=str,
        help="Path to initial pytorch checkpoint to start training from.")

    parser.add_argument(
        "--skip_checkpointing",
        default=None,
        type=str,
        help="Whether to skip checkpointing.")    

    parser.add_argument(
        '--resume_from_step',
        type=positive_int,
        default=None,
        help="Step to resume training from.")

    parser.add_argument(
        '--num_steps_per_checkpoint',
        type=int,
        default=100,
        help="Number of update steps until a model checkpoint is saved to disk.")

    # mixed precision specific
    parser.add_argument(
        '--fp16',
        default=False,
        action='store_true',
        help="Whether to use 16-bit float precision instead of 32-bit")

    # compute specific 
    parser.add_argument(
        '--deepspeed_zero_stage',
        default=False,
        action='store_true',
        help="Whether ORT will partition optimizer.")

    return parser.parse_args()

args = parse_arguments()