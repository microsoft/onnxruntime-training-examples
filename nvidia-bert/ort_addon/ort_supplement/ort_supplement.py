# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import onnx
import torch

# onnxruntime training API
from onnxruntime.experimental import orttrainer, optim, amp as ort_amp

# converts azureml environment variables into context for distributed run
from ort_supplement.azureml_adapter import \
    set_environment_variables_for_nccl_backend, \
    get_local_rank, get_local_size, get_global_size, get_world_size, get_world_rank 

def setup_onnxruntime_with_mpi(args):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    has_aml = 'AZ_BATCH_MASTER_NODE' in os.environ.keys() or 'AZ_BATCHAI_MPI_MASTER_NODE' in os.environ.keys()
    if not has_aml:
        print('Detected local run')
        args.local_rank = comm.Get_rank() % torch.cuda.device_count()
        args.world_rank = comm.Get_rank()
        args.world_size = comm.Get_size()

        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1

    else:
        print('Detected Azure batch run')
        set_environment_variables_for_nccl_backend(get_local_size() == get_global_size(), IB = args.use_ib)
        args.local_rank = get_local_rank()
        args.local_size = get_local_size()
        args.world_rank = get_world_rank()
        args.world_size = get_global_size()

        print('Local rank: {}'.format(args.local_rank))
        print('Local size: {}'.format(args.local_size))
        print('World rank: {}'.format(args.world_rank))
        print('World size: {}'.format(args.world_size))
        print('CUDA device: {}'.format(args.local_rank))

        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1

        torch.distributed.init_process_group(backend='nccl')

    from onnxruntime.capi._pybind_state import set_cuda_device_id 
    set_cuda_device_id(args.local_rank)

    from onnxruntime.capi._pybind_state import set_arena_extend_strategy, ArenaExtendStrategy
    set_arena_extend_strategy(ArenaExtendStrategy.kSameAsRequested)

    return device

def optimizer_parameters_mutiple_groups(model):
    '''A method to assign different hyper parameters for different model parameter groups'''
    no_decay_keys = ["bias", "gamma", "beta", "LayerNorm"]
    no_decay_param_group = []
    decay_param_group = []
    for initializer in model.graph.initializer:
        if any(key in initializer.name for key in no_decay_keys):
            no_decay_param_group.append(initializer.name)
        else:
            decay_param_group.append(initializer.name)
    params = [{'params': no_decay_param_group, "alpha": 0.9, "beta": 0.999, "lambda_coef": 0.0, "epsilon": 1e-6},
              {'params': decay_param_group, "alpha": 0.9, "beta": 0.999, "lambda_coef": 0.01, "epsilon": 1e-6}]
    return params

def create_ort_trainer(args, device, model):

    # MODEL DESCRIPTION
    vocab_size = 30528
    micro_batch = args.train_batch_size // args.gradient_accumulation_steps
    model_desc = {
        'inputs': [
            ('input_ids', [args.train_batch_size, args.max_seq_length]),
            ('segment_ids', [args.train_batch_size, args.max_seq_length]),
            ('input_mask', [args.train_batch_size, args.max_seq_length]),
            ('masked_lm_labels', [args.train_batch_size, args.max_seq_length]),
            ('next_sentence_labels', [args.train_batch_size, 2])
        ],
        'outputs': [
            ('loss', [], True)
        ]
    }

    # OPTIMIZER CONVERSION
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
    optim_config = optim.LambConfig(
        lr=args.learning_rate, alpha=0.9, beta=0.999, lambda_coef=0.01, epsilon=1e-6,
        params = [{
            'params' : [n for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'alpha': 0.9, 'beta': 0.999, 'lambda_coef': 0.0, 'epsilon': 1e-6
        },
        {
            'params' : [n for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'alpha': 0.9, 'beta': 0.999, 'lambda_coef': 0.01, 'epsilon': 1e-6
        }],
    )

    # LEARNING RATE SCHEDULER CONVERSION 
    lr_scheduler = optim.lr_scheduler.LinearWarmupLRScheduler(
        total_steps=int(args.max_steps), warmup=args.warmup_proportion)

    # LOSS SCALAR CONVERSION
    loss_scaler = ort_amp.loss_scaler.DynamicLossScaler()

    # ORT TRAINER OPTIONS 
    trainer_config = orttrainer.ORTTrainerOptions({
        'device': {
            'id': str(device), 
            'mem_limit': int(args.gpu_memory_limit_gb * 1024 * 1024 *1024)
        },
        'batch': {
            'gradient_accumulation_steps' : args.gradient_accumulation_steps
        },
        'distributed': {
            'world_size': args.world_size,
            'world_rank': args.world_rank,
            'allreduce_post_accumulation': True if args.allreduce_post_accumulation else False,
            # 'missing? deepspeed_zero_stage': 1 if args.deepspeed_zero_stage else 0,
        },
        'lr_scheduler': lr_scheduler,
        'mixed_precision': {
            'enabled': False # True if args.fp16 else False,
            # 'loss_scaler': loss_scaler
        },
        'debug': {
             'deterministic_compute' : True
        },
        '_internal_use': {
            'onnx_opset_version': 12
        }
    })

    # ORT TRAINER CONSTRUCTION
    trainer = orttrainer.ORTTrainer(
        model, model_desc, optim_config, loss_fn=None, options=trainer_config)

    return trainer

def run_ort_training_step(args, global_step, training_steps, model, batch):
    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch

    if args.fp16:
        loss = model.train_step(input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels)
        all_finite = 1
        if isinstance(loss, (list, tuple)):
            assert len(loss) == 2
            loss, all_finite = loss
    else:
        loss = model.train_step(input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels)

    if training_steps % args.gradient_accumulation_steps == 0:
        global_step += 1

    return loss, global_step
 
