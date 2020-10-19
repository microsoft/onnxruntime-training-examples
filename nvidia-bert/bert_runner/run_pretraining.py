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
import logging
import os
import sys
import time
import random

import numpy
import torch
import onnx
import onnxruntime
import onnxruntime.training

from .arguments import args
from . import bert_model
from . import bert_dataset
from . import distributed

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

def main():
    valid_inputs()
    initialize_environment()
    print_header()

    bert = build_pytorch_model()
    trainer = build_onnxruntime_trainer(bert)
    dataset = build_training_dataset()
    
    if initial_weights_provided():
        load_initial_weights(trainer)

    step = 1
    if resume_from_checkpoint():
        step = args.resume_from_step
        restore_from_checkpoint(step, dataset, trainer)

    results = training_loop(step, dataset, trainer)

    if distributed.is_world_leader():
        save_onnx_model(trainer)
    print_footer(results)

def valid_inputs():
    validate_starting_condition()
    validate_termination_condition()

def validate_starting_condition():
    if args.init_checkpoint is not None and args.resume_from_step is not None:
        raise ValueError('Only one of initial_checkpoint and resume_from_step may be specified.')

def validate_termination_condition():
    # one and only one of max_steps or max_epochs
    if args.max_steps is None and args.max_epochs is None:
        raise ValueError('One of max_steps or max_epochs must be specified.')
    if args.max_steps is not None and args.max_epochs is not None:
        raise ValueError('Only one of max_steps or max_epochs may be specified.')

def initialize_environment():
    reset_cpu_affinity()
    initialize_logger()
    initialize_seeds()
    if distributed.is_world_leader():
        initialize_dirs()

def reset_cpu_affinity():
    # os.system("taskset -p 0xff %d" % os.getpid())
    pass

def initialize_logger():
    if not args.debug:
        logging.basicConfig(format='%(message)s', level=logging.INFO)
    else:
        fmt='['
        fmt+='%(asctime)s %(msecs)3d ms'
        fmt+=', Rank {}, Pid %(process)d'.format(distributed.world_rank())
        fmt+='] %(message)s'
        logging.basicConfig(format=fmt, datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

def initialize_seeds():
    # warning: set torch seed to generate same initial weights across ranks
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    onnxruntime.set_seed(args.seed + distributed.world_rank())

def initialize_dirs():
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if is_checkpointing():
        if not os.path.exists(checkpoint_dir()):
            os.makedirs(checkpoint_dir())

def is_checkpointing():
    return not args.skip_checkpointing

def checkpoint_dir():
    return os.path.join(args.output_dir, 'checkpoints')

def print_header():
    if distributed.is_world_leader():
        gbs = distributed.world_size()*args.gpu_feed_batch_size*args.gradient_accumulation_passes
        logging.info(datetime.datetime.now().strftime('%m/%d/%Y %I:%M:%S %p'))        
        logging.info('World Size: {}'.format(distributed.world_size()))
        logging.info('GPU Feed Batch Size: {}'.format(args.gpu_feed_batch_size))
        logging.info('Gradient Accumulation Passes: {}'.format(args.gradient_accumulation_passes))
        logging.info('Global Batch Size: {}'.format(gbs))
        logging.info('Sequence Length: {}'.format(args.max_seq_length))
        logging.info('Max Steps: {}'.format(args.max_steps))
        logging.info('Seed: {}'.format(args.seed))
        logging.info('Smoothing over {} passes'.format(get_num_passes_to_smooth()))
        logging.info('Weights are updated in one \'step\' and model is backpropagated in one \'pass\'')
        if resume_from_checkpoint():
            logging.info('Resuming from step: {}'.format(args.resume_from_step))
    distributed.world_barrier()

    gpuid = distributed.local_rank()
    gpu_prop = torch.cuda.get_device_properties('cuda:{}'.format(gpuid))
    logging.info('GPU {}: {} with {} GB'.format(gpuid, gpu_prop.name, gpu_prop.total_memory/1024/1024))

def build_pytorch_model():
    config = bert_model.BertConfig.from_json_file(args.config_file)

    # additional configuration is due to 'dense_sequence' optimization
    config.max_predictions_per_seq = args.max_predictions_per_seq
    config.dense_seq = True
    
    config.dense_seq_output = True
    config.fused_mha = False
    config.fused_gelu_bias = False
    config.max_prediction_count = args.max_predictions_per_seq

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
    bs = args.gpu_feed_batch_size
    sl = args.max_seq_length
    return {
        'inputs': [
            ('input_ids', [bs, sl]),
            ('segment_ids', [bs, sl]),
            ('input_mask', [bs, sl]),
            ('masked_lm_labels', [bs, sl]),
            ('next_sentence_labels', [bs, 2])
        ],
        'outputs': [
            ('loss', [], True)
        ]
    }

def build_onnxruntime_optimizer_configuration(model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
    return onnxruntime.training.optim.LambConfig(
        lr=args.learning_rate, 
        alpha=0.9, beta=0.999, lambda_coef=0.01, epsilon=1e-6,
        do_bias_correction=True,
        params = [{
            'params' : [n for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'alpha': 0.9, 'beta': 0.999, 'lambda_coef': 0.00, 'epsilon': 1e-6
        }]
    )

def build_onnxruntime_learning_rate_schedule():
    return onnxruntime.training.optim.lr_scheduler.PolyWarmupLRScheduler(
        total_steps=args.max_steps, 
        warmup=args.warmup_proportion)

def build_onnxruntime_trainer_options(model_desc, lr_scheduler):
    gpuid = 'cuda:{}'.format(distributed.local_rank())
    gpu_memory = torch.cuda.get_device_properties(gpuid).total_memory

    return onnxruntime.training.ORTTrainerOptions({
        'device': {
            'id': gpuid,
            'mem_limit': 1024*gpu_memory
        },
        'batch': {
            'gradient_accumulation_steps' : args.gradient_accumulation_passes
        },
        'distributed': {
            'world_size': distributed.world_size(),
            'world_rank': distributed.world_rank(),
            'allreduce_post_accumulation': True if args.allreduce_post_accumulation else False,
            'deepspeed_zero_optimization': {
                'stage': 1 if args.deepspeed_zero_stage else 0,
            }        
        },
        'lr_scheduler': lr_scheduler,
        'mixed_precision': {
            'enabled': True if args.fp16 else False,
        },
        # https://github.com/microsoft/onnxruntime/pull/5354/files
        # '_internal_use': {
        #     'enable_gelu_approximation': True
        # }
    })
    
def build_training_dataset():
    data_dir = args.data_dir
    training_datafiles = [
        os.path.join(data_dir, component) for component in os.listdir(data_dir) if
        os.path.isfile(os.path.join(data_dir, component)) and 'training' in component
    ]

    if len(training_datafiles) == 0:
        raise ValueError('No prospective training data files found.')

    training_datafiles = sorted(training_datafiles)
    return bert_dataset.BertMultiFileDataset(
        training_datafiles[distributed.world_rank()::distributed.world_size()],
        shuffle = True,
        loop = True)

def initial_weights_provided():
    return args.init_checkpoint is not None

def load_initial_weights(trainer):
    checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
    onnxruntime.training.checkpoint.experimental_load_state_dict(trainer, checkpoint['model'], strict=False)
    logging.info('Restored checkpoint {}'.format(args.init_checkpoint))

def resume_from_checkpoint():
    return args.resume_from_step is not None

def restore_from_checkpoint(step, dataset, trainer):
    mbs = args.gpu_feed_batch_size
    gas = args.gradient_accumulation_passes
    total_samples = step*gas*mbs
    dataset.forward(total_samples)

    checkpoint_path = os.path.join(checkpoint_dir(), 'ckpt_{}.pt'.format(step))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    onnxruntime.training.checkpoint.experimental_load_state_dict(trainer, checkpoint['model'], strict=False)
    logging.info('Restored checkpoint {}'.format(checkpoint_path))

def training_loop(step, dataset, trainer):
    t0 = t1 = time.perf_counter()
    results = Results()

    accumulate_steps = args.gradient_accumulation_passes
    checkpoint_steps = args.num_steps_per_checkpoint

    weight_step = step
    execution_step = 1 + (step-1)*accumulate_steps

    batch_generator = iter(build_training_dataloader(dataset))
    batch = next(batch_generator, None)

    if distributed.is_world_leader() or distributed.have_separate_log():
        print_statistics_header()

    while not is_terminal_condition(weight_step, batch):

        # execute backward pass and record backend time
        # warning: backend may exit early because CUDA is non-blocking!
        loss, session_dt = run_timed_training_pass(trainer, batch)
        t0, script_dt = get_split_time(t0)

        # if CPU did not block, we can fetch next batch in downtime
        # (if num_workers > 0, the batch should fetch in background)        
        batch = next(batch_generator, None)

        # CPU will block on loss.item() if it didn't already
        results.add_datum(loss.item(), session_dt, script_dt)

        increment_weight_step = execution_step % accumulate_steps == 0
        if weight_step % args.num_steps_per_log_entry == 0:
            if increment_weight_step: 
                print_statistics_line(weight_step, execution_step, results)
                print_internals_line_if_debugging(trainer)                

        if increment_weight_step:
            if (distributed.is_world_leader() and
                is_checkpointing() and
                weight_step % checkpoint_steps == 0):
                save_checkpoint(weight_step, trainer)
            weight_step += 1
        execution_step += 1

    return results

def is_terminal_condition(step, batch):
    if batch is None:
        return False
    if args.max_steps is not None:
        return args.max_steps <= step
    return False

def get_split_time(t_previous):
    t_current = time.perf_counter()
    return t_current, t_current - t_previous

def build_training_dataloader(training_dataset):
    return torch.utils.data.DataLoader(
        training_dataset,
        batch_size = args.gpu_feed_batch_size,
        num_workers = 1,
        pin_memory = True, 
        drop_last = True)

class Results():
    def __init__(self, max_retained = None):
        self.start_time = time.perf_counter()
        self.data = collections.deque(maxlen=max_retained)
        self.time = collections.deque(maxlen=max_retained)

    def add_datum(self, *args):
        self.time.append(time.perf_counter())        
        self.data.append(args)

    def last_datum(self):
        return self.data[-1]

    def get_training_time(self):
        return self.time[-1] - self.start_time

    def get_timespan_on_tail(self, steps):
        return self.time[-1] - self.time[max(0, len(self.data)-steps)]

def run_timed_training_pass(trainer, batch):
    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch

    t0 = time.perf_counter()
    loss = trainer.train_step(input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels)
    t1 = time.perf_counter()

    return loss, t1-t0

def print_statistics_header():
    if not args.debug:
        logging.info('{:^4} {:^4} {:^2} {:^10} {:^10} {:8^}'.format(
            'Step', 'Pass', 'Rank', 'Loss', 'Step(s)', 'Seq/sec'))

def print_statistics_line(weight_step, execution_step, results):
    loss, session_dt, script_dt = results.last_datum()
    if not args.debug:
        smooth_n = get_num_passes_to_smooth()
        smooth_dt = results.get_timespan_on_tail(smooth_n)
        batch_size = args.gpu_feed_batch_size
        logging.info('{:^4} {:^4} {:2d} {:10.6f} {:10.6f} {:8.2f}'.format(
            weight_step,
            execution_step,
            distributed.world_rank(),
            loss, 
            smooth_dt/smooth_n, 
            batch_size*smooth_n/smooth_dt))
    else:
        logging.debug('step {} pass {} session-time {:.6f} script-time {:.6f} loss {:.6f}'.format(
            weight_step, execution_step, session_dt, script_dt-session_dt, loss))

def print_internals_line_if_debugging(trainer):
    if args.debug:
        logging.debug('internal step {} learning rate {:.6f} loss scalar {} gradients finite {}'.format(
            trainer._train_step_info.optimization_step,
            trainer.options.lr_scheduler._last_lr[0],
            trainer.options.mixed_precision.loss_scaler.loss_scale,
            trainer._train_step_info.all_finite.item()))

def get_num_passes_to_smooth():
    return max(args.num_passes_to_smooth_throughput, args.gradient_accumulation_passes)

def save_checkpoint(weight_step, trainer):
    logging.info('{}'.format(' <-- checkpoint'))
    checkpoint_path = os.path.join(checkpoint_dir(), 'ckpt_{}.pt'.format(weight_step))
    state = {'model': onnxruntime.training.checkpoint.experimental_state_dict(trainer)}
    torch.save(state, checkpoint_path)

def save_onnx_model(trainer):
    model_path = os.path.join(args.output_dir, 'trained_bert.onnx')
    trainer.save_as_onnx(model_path)
    logging.info('Saved onnx model to {}'.format(model_path))

def print_footer(results):
    if distributed.is_world_leader():
        final_loss, _, _ = results.last_datum()
        logging.info('Final Loss: {:8.6f}'.format(final_loss))    
        logging.info('Running Time: {:.2f}'.format(results.get_training_time()))

        logging.info(datetime.datetime.now().strftime('%m/%d/%Y %I:%M:%S %p'))

if __name__ == "__main__":
    main()
