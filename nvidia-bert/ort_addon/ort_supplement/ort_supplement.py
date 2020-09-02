# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import onnx
from onnxruntime.capi.ort_trainer import ORTTrainer, IODescription, ModelDescription
from onnxruntime.capi.ort_trainer import LossScaler
import torch
from ort_supplement.azureml_adapter import set_environment_variables_for_nccl_backend, get_local_rank, get_local_size, get_global_size, get_world_size, get_world_rank 

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

def bert_model_description(args):
    vocab_size = 30528

    # allow variable input sizes:
    # input_ids_desc = IODescription('input_ids', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes = vocab_size)
    # segment_ids_desc = IODescription('segment_ids', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes = 2)
    # input_mask_desc = IODescription('input_mask', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes = 2)
    # masked_lm_labels_desc = IODescription('masked_lm_labels', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes = vocab_size)
    # next_sentence_labels_desc = IODescription('next_sentence_labels', ['batch',], torch.int64, num_classes = 2)

    # set concrete input sizes to permit optimization
    micro_batch = args.train_batch_size // args.gradient_accumulation_steps
    input_ids_desc = IODescription('input_ids', [args.train_batch_size, args.max_seq_length], torch.int64, num_classes = vocab_size)
    segment_ids_desc = IODescription('segment_ids', [args.train_batch_size, args.max_seq_length], torch.int64, num_classes = 2)
    input_mask_desc = IODescription('input_mask', [args.train_batch_size, args.max_seq_length], torch.int64, num_classes = 2)
    masked_lm_labels_desc = IODescription('masked_lm_labels', [args.train_batch_size, args.max_seq_length], torch.int64, num_classes = vocab_size)
    next_sentence_labels_desc = IODescription('next_sentence_labels', [args.train_batch_size,2], torch.int64, num_classes = 2)

    loss_desc = IODescription('loss', [], torch.float32)
    return ModelDescription([input_ids_desc, segment_ids_desc, input_mask_desc, masked_lm_labels_desc, next_sentence_labels_desc], [loss_desc])

def create_ort_trainer(args, device, model):
    
    # set GPU memory limitation (per card!)
    from onnxruntime.capi._pybind_state import set_cuda_mem_limit
    ort_cuda_mem_limit_in_gbs = args.gpu_memory_limit_gb
    set_cuda_mem_limit(int(ort_cuda_mem_limit_in_gbs * 1024 * 1024 *1024))

    # BertLAMB default initial settings: b1=0.9, b2=0.999, e=1e-6
    def map_optimizer_attributes(name):
        no_decay_keys = ["bias", "gamma", "beta", "LayerNorm"]
        no_decay = False
        for no_decay_key in no_decay_keys:
            if no_decay_key in name:
                no_decay = True
                break
        if no_decay:
            return {"alpha": 0.9, "beta": 0.999, "lambda": 0.0, "epsilon": 1e-6}
        else:
            return {"alpha": 0.9, "beta": 0.999, "lambda": 0.01, "epsilon": 1e-6}

    # we request ORTTrainer to create a LambOptimizer with given optimizer_attributes. 
    # train_step does forward, backward, and optimize step.
    model = ORTTrainer(model, None, bert_model_description(args), "LambOptimizer", 
        map_optimizer_attributes,
        IODescription('Learning_Rate', [1,], torch.float32),
        device, 
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        world_rank=args.world_rank, world_size=args.world_size,
        use_mixed_precision = True if args.fp16 else False,
        allreduce_post_accumulation = True if args.allreduce_post_accumulation else False,
        deepspeed_zero_stage = 1 if args.deepspeed_zero_stage else 0,
        _opset_version = 12)

    if args.fp16:
        setattr(args, 'ort_loss_scale', LossScaler(model.loss_scale_input_name, True, up_scale_window=2000))

    return model

from ort_supplement.lr_schedules import SCHEDULES
def get_lr(args, training_steps, schedule='warmup_poly'):
    if args.max_steps == -1:
        return args.learning_rate

    schedule_fct = SCHEDULES[schedule]
    return args.learning_rate * schedule_fct(training_steps / args.max_steps, args.warmup_proportion)

def run_ort_training_step(args, global_step, training_steps, model, batch):
    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch

    lr = get_lr(args, global_step, args.schedule)
    learning_rate = torch.tensor([lr])
    if args.fp16:
        loss_scale = torch.tensor([args.ort_loss_scale.loss_scale_])
        loss = model.train_step(input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels, learning_rate, loss_scale)
        all_finite = 1
        if isinstance(loss, (list, tuple)):
            assert len(loss) == 2
            loss, all_finite = loss
    else:
        loss = model(input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels, learning_rate)
    if training_steps % args.gradient_accumulation_steps == 0:
        if args.fp16:
            args.ort_loss_scale.update_loss_scale(all_finite.item())
        global_step += 1

    return loss, global_step
 
