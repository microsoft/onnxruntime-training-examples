# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import onnx
import torch

# onnxruntime training API
from onnxruntime.training import orttrainer, optim, amp as ort_amp
from onnxruntime.training.orttrainer import ORTTrainer, ORTTrainerOptions

# converts azureml environment variables into context for distributed run
from ort_supplement.azureml_adapter import \
    set_environment_variables_for_nccl_backend, \
    get_local_rank, get_local_size, get_global_size, get_world_size, get_world_rank 

def setup_onnxruntime_with_mpi(args):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    has_aml = 'AZ_BATCH_MASTER_NODE' in os.environ.keys() or 'AZ_BATCHAI_MPI_MASTER_NODE' in os.environ.keys()
    if not has_aml:
        # outside of Azure we get MPI context from mpi4py
        print('Detected local run')
        args.local_rank = comm.Get_rank() % torch.cuda.device_count()
        args.world_rank = comm.Get_rank()
        args.world_size = comm.Get_size()

        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1

    else:
        # on Azure machine learning compute we get MPI context from environment variables
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

    # tell onnxruntime which device to use and seed its random generators
    from onnxruntime import set_seed
    set_seed(args.seed + args.world_rank)

    return device

def create_ort_trainer(args, device, model):

    # MODEL INPUT AND OUTPUT DESCRIPTION
    # note: These names must match argument names and order from model.forward(...)
    vocab_size = 30528
    micro_batch = args.train_batch_size // args.gradient_accumulation_steps
    model_desc = {
        'inputs': [
            ('input_ids', [args.train_batch_size, args.max_seq_length]),
            ('token_type_ids', [args.train_batch_size, args.max_seq_length]),
            ('attention_mask', [args.train_batch_size, args.max_seq_length]),
            ('masked_lm_labels', [args.train_batch_size, args.max_seq_length]),
            ('next_sentence_labels', [args.train_batch_size, 2])
        ],
        'outputs': [
            ('total_loss', [], True),
            ('mlm_acc', [], False)
        ]
    }

    # TRAINING OPTIMIZER SPECIFICATION
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
    optim_config = optim.LambConfig(
        lr=args.learning_rate, alpha=0.9, beta=0.999, lambda_coef=0.01, epsilon=1e-6,
        do_bias_correction=True,
        params = [{
            'params' : [n for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'alpha': 0.9, 'beta': 0.999, 'lambda_coef': 0.00, 'epsilon': 1e-6
        }]
    )

    # LEARNING RATE SCHEDULE SPECIFICATION
    lr_scheduler = optim.lr_scheduler.LinearWarmupLRScheduler(
        total_steps=int(args.max_steps), warmup=args.warmup_proportion)

    # ONNXRUNTIME TRAINER OPTIONS 
    trainer_config = ORTTrainerOptions({
        'device': {
            'id': str(device)
        },
        'batch': {
            'gradient_accumulation_steps' : args.gradient_accumulation_steps
        },
        'distributed': {
            'world_size': args.world_size,
            'world_rank': args.world_rank,
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

    # ONNXRUNTIME TRAINER CONSTRUCTION (loss fn embedded in model)
    trainer = ORTTrainer(
        model, model_desc, optim_config, loss_fn=None, options=trainer_config)

    return trainer

def run_ort_training_step(args, global_step, training_steps, trainer, batch):
    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
    loss, mlm_acc = trainer.train_step(input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels)
    if training_steps % args.gradient_accumulation_steps == 0:
        global_step += 1
    return loss, global_step
 
