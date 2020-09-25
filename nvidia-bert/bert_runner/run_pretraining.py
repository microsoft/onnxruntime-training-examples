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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import datetime
import itertools
import logging
import os
import statistics
import sys
import time
import random

from tqdm import tqdm, trange
import numpy
import torch
import onnx
import onnxruntime
import onnxruntime.training

from . import bert_model
from . import bert_dataset
from . import configuration
from . import distributed

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

def main():
    initialize_environment()
    print_header()

    bert = build_pytorch_model()
    trainer = build_onnxruntime_trainer(bert)
    dataset = build_training_dataset()
    
    if configuration.initial_weights_provided():
        load_initial_weights(trainer)

    step = 1
    if configuration.resume_from_checkpoint():
        step = configuration.arguments.resume_step
        restore_from_checkpoint(step, dataset, trainer)

    results = training_loop(step, dataset, trainer)

    save_onnx_model(trainer)
    print_footer(results)

def initialize_environment():
    initialize_logger()
    initialize_seeds()
    initialize_dirs()

def initialize_logger():
    if not configuration.arguments.debug:
        logging.basicConfig(format='%(message)s', level=logging.INFO)
    else:
        fmt='['
        fmt+='%(asctime)s %(msecs)3d ms'
        fmt+=', Rank {}, Pid %(process)d'.format(distributed.world_rank)
        fmt+='] %(message)s'
        logging.basicConfig(format=fmt, datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

def initialize_seeds():
    worker_seed = configuration.arguments.seed + distributed.local_rank
    random.seed(worker_seed)
    numpy.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
    onnxruntime.set_seed(worker_seed)

def initialize_dirs():
    if not os.path.exists(configuration.output_dir()):
        os.makedirs(configuration.output_dir())
    if configuration.is_checkpointing():
        if not os.path.exists(configuration.checkpoint_dir()):
            os.makedirs(configuration.checkpoint_dir())

def print_header():
    if distributed.is_world_leader():
        args = configuration.arguments
        logging.info(datetime.datetime.now().strftime('%m/%d/%Y %I:%M:%S %p'))        
        logging.info('World Size: {}'.format(distributed.world_size))
        logging.info('GPU Feed Batch Size: {}'.format(args.gpu_feed_batch_size))
        logging.info('Gradient Accumulation Passes: {}'.format(args.gradient_accumulation_passes))
        logging.info('Sequence Length: {}'.format(args.max_seq_length))
        logging.info('Max Steps: {}'.format(args.max_steps))
        logging.info('Seed: {}'.format(args.seed))
        if configuration.resume_from_checkpoint():
            logging.info('Resuming from step: {}'.format(args.resume_step))
    distributed.world_barrier()

    gpuid = distributed.local_rank
    gpu_prop = torch.cuda.get_device_properties('cuda:{}'.format(gpuid))
    logging.info('GPU {}: {} with {} GB'.format(gpuid, gpu_prop.name, gpu_prop.total_memory/1024/1024))

def build_pytorch_model():
    config = bert_model.BertConfig.from_json_file(configuration.arguments.config_file)

    # additional configuration is due to 'dense_sequence' optimization
    config.max_predictions_per_seq = configuration.arguments.max_predictions_per_seq
    config.dense_seq = True
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    return bert_model.BertForPreTraining(config)

def build_onnxruntime_trainer(model):
    model_desc = build_onnx_model_description()
    optim_config = build_onnxruntime_optimizer_configuration(model)
    lr_scheduler = build_onnxruntime_learning_rate_schedule()
    trainer_config = build_onnxruntime_trainer_options(model_desc, lr_scheduler)
    return onnxruntime.training.ORTTrainer(
        model, model_desc, optim_config, loss_fn=None, options=trainer_config)

def build_onnx_model_description():
    bs = configuration.arguments.gpu_feed_batch_size
    sl = configuration.arguments.max_seq_length
    return {
        'inputs': [
            ('input_ids', [bs, sl]),
            ('token_type_ids', [bs, sl]),
            ('attention_mask', [bs, sl]),
            ('masked_lm_labels', [bs, sl]),
            ('next_sentence_labels', [bs, 2])
        ],
        'outputs': [
            ('total_loss', [], True)
        ]
    }

def build_onnxruntime_optimizer_configuration(model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
    return onnxruntime.training.optim.LambConfig(
        lr=configuration.arguments.learning_rate, 
        alpha=0.9, beta=0.999, lambda_coef=0.01, epsilon=1e-6,
        do_bias_correction=True,
        params = [{
            'params' : [n for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'alpha': 0.9, 'beta': 0.999, 'lambda_coef': 0.00, 'epsilon': 1e-6
        }]
    )

def build_onnxruntime_learning_rate_schedule():
    return onnxruntime.training.optim.lr_scheduler.LinearWarmupLRScheduler(
        total_steps=configuration.arguments.max_steps, 
        warmup=configuration.arguments.warmup_proportion)

def build_onnxruntime_trainer_options(model_desc, lr_scheduler):
    gpuid = 'cuda:{}'.format(distributed.local_rank)
    gpu_memory = torch.cuda.get_device_properties(gpuid).total_memory

    args = configuration.arguments
    return onnxruntime.training.ORTTrainerOptions({
        'device': {
            'id': gpuid,
            'mem_limit': 1024*gpu_memory
        },
        'batch': {
            'gradient_accumulation_steps' : args.gradient_accumulation_passes
        },
        'distributed': {
            'world_size': distributed.world_size,
            'world_rank': distributed.world_rank,
            'allreduce_post_accumulation': True if args.allreduce_post_accumulation else False,
            'deepspeed_zero_optimization': {
                'stage': 1 if args.deepspeed_zero_stage else 0,
            }        
        },
        'lr_scheduler': lr_scheduler,
        'mixed_precision': {
            'enabled': True if args.fp16 else False,
        }
    })
    
def build_training_dataset():
    data_dir = configuration.arguments.data_dir
    training_datafiles = [
        os.path.join(data_dir, component) for component in os.listdir(data_dir) if
        os.path.isfile(os.path.join(data_dir, component)) and 'training' in component
    ]

    if len(training_datafiles) == 0:
        raise ValueError('No prospective training data files found.')

    training_datafiles = sorted(training_datafiles)
    return bert_dataset.BertMultiFileDataset(
        training_datafiles[distributed.world_rank::distributed.world_size],
        loop = True)

def load_initial_weights(trainer):
    checkpoint = torch.load(configuration.arguments.init_checkpoint, map_location="cpu")
    onnxruntime.training.checkpoint.experimental_load_state_dict(trainer, checkpoint['model'], strict=False)
    logging.info('Restored checkpoint {}'.format(configuration.arguments.init_checkpoint))

def restore_from_checkpoint(step, dataset, trainer):
    mbs = configuration.arguments.gpu_feed_batch_size
    gas = configuration.arguments.gradient_accumulation_passes
    total_samples = step*gas*mbs
    dataset.forward(total_samples)

    checkpoint_dir = configuration.checkpoint_dir()
    checkpoint_path = os.path.join(checkpoint_dir, 'ckpt_{}.pt'.format(step))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    onnxruntime.training.checkpoint.experimental_load_state_dict(trainer, checkpoint['model'], strict=False)
    logging.info('Restored checkpoint {}'.format(checkpoint_path))

def training_loop(step, dataset, trainer):
    t0 = time.perf_counter()
    results = Results()

    accumulate_steps = configuration.arguments.gradient_accumulation_passes
    checkpoint_steps = configuration.arguments.num_steps_per_checkpoint

    weight_step = step
    execution_step = step * accumulate_steps

    batch_generator = iter(build_training_dataloader(dataset))
    batch = next(batch_generator)

    while not is_terminal_condition(weight_step):

        # execute backward pass and record backend time
        # warning: backend may exit early because CUDA is non-blocking!
        loss, session_dt = run_timed_training_pass(trainer, batch)
        t0, script_dt = get_split_time(t0)

        # if CPU did not block, we can fetch next batch in downtime
        batch = next(batch_generator)

        # CPU will block on loss.item() if it didn't already
        results.add_datum(loss.item(), session_dt, script_dt)
        if distributed.is_world_leader():
            print_running_statistics(execution_step, weight_step, results)

        if execution_step % accumulate_steps == 0:
            if distributed.is_world_leader() and configuration.is_checkpointing() and weight_step % checkpoint_steps == 0:
                save_checkpoint(weight_step, trainer)
            weight_step += 1
        execution_step += 1

    return results

def is_terminal_condition(step):
    if configuration.arguments.max_steps is not None:
        return configuration.arguments.max_steps <= step
    return False

def get_split_time(t_previous):
    t_current = time.perf_counter()
    return t_current, t_current - t_previous

def build_training_dataloader(training_dataset):
    return torch.utils.data.DataLoader(
        training_dataset,
        batch_size = configuration.arguments.gpu_feed_batch_size,
        num_workers = 1,
        pin_memory = True, 
        drop_last = True)

class Results():
    def __init__(self, maxlen = None):
        self.start_time = time.perf_counter()
        self.last_time = time.perf_counter()   
        self.data = collections.deque(maxlen=maxlen)

    def add_datum(self, *args):
        self.data.append(args)
        self.last_time = time.perf_counter()   

    def last_datum(self):
        return self.data[-1]

    def get_runtime(self):
        return self.last_time - self.start_time

    def get_mean(self):
        return self._get_mean_as_tuple(self.data)

    def get_mean_lastn(self, count):
        l = len(self.data)
        lastn = collections.deque(itertools.islice(self.data, max(0, l-count), l))
        return self._get_mean_as_tuple(lastn)

    def _get_mean_as_tuple(self, data):
        mean = []
        for i in range(len(self.data[0])):
            mean.append(statistics.mean(value[i] for value in data))
        return tuple(mean)

def run_timed_training_pass(trainer, batch):
    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch

    t0 = time.perf_counter()
    loss = trainer.train_step(input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels)
    t1 = time.perf_counter()

    return loss, t1-t0

def print_running_statistics(execution_step, weight_step, results):
    loss, _, script_dt = results.last_datum()
    logging.debug('pass {} time {:.6f} loss {:.6f}'.format(execution_step, script_dt, loss))

    lastn = 5

    batch_size = configuration.arguments.gpu_feed_batch_size
    stable_loss, stable_session_dt, stable_script_dt = results.get_mean_lastn(lastn)

    init_step = max(0, configuration.arguments.resume_step, execution_step - lastn)
    last_step = execution_step

    passes = '{}-{}'.format(init_step, last_step)
    logging.info('{:^15} {:10.6f} {:10.6f} {:10.6f} {:8.2f}'.format(
        passes,
        stable_loss, 
        stable_session_dt, 
        stable_script_dt, 
        batch_size/stable_script_dt))

def save_checkpoint(weight_step, trainer):
    checkpoint_path = os.path.join(configuration.checkpoint_dir(), 'ckpt_{}.pt'.format(weight_step))
    state = {'model': onnxruntime.training.checkpoint.experimental_state_dict(trainer)}
    torch.save(state, checkpoint_path)

def save_onnx_model(trainer):
    model_path = os.path.join(configuration.output_dir(), 'final_bert.onnx')
    trainer.save_as_onnx(model_path)

def print_footer(results):
    if distributed.is_world_leader():
        final_loss, _, _ = results.last_datum()
        logging.info('Final Loss: {:8.6f}'.format(final_loss))    
        logging.info('Running Time: {:.2f}'.format(results.get_runtime()))

        logging.info(datetime.datetime.now().strftime('%m/%d/%Y %I:%M:%S %p'))

if __name__ == "__main__":
    main()
