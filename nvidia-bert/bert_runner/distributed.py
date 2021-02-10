# Copyright (c) 2020 Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import hashlib
import logging
import os
import socket

import torch

from .arguments import args

def is_distributed():
    global _is_distributed
    return _is_distributed

def world_rank():
    global _world_rank
    return _world_rank

def world_size():
    global _world_size
    return _world_size

def local_rank():
    global _local_rank
    return _local_rank

def local_size():
    global _local_size
    return _local_size

def is_world_leader():
    return world_rank() == 0

def is_local_leader():
    return local_rank() == 0

def world_barrier():
    if is_distributed():
        _world_comm.barrier()

def is_azureml_compute():
    aml_variables = ['AZUREML_RUN_ID', 'AZUREML_EXPERIMENT_ID', 'AZUREML_NODE_COUNT']
    return any(key in os.environ.keys() for key in aml_variables)
        
def have_separate_log():
    return is_azureml_compute()

def ensure_no_core_restriction():
     process_cpu_affinities = os.sched_getaffinity(0)
     if sorted(process_cpu_affinities) != sorted(range(os.cpu_count())):
         os.sched_setaffinity(0, range(os.cpu_count()))
         logging.debug('Reset CPU affinity to {}'.format(os.sched_getaffinity(0)))

def _set_torch_device():
    torch.cuda.set_device(_local_rank)

def _discover_local_rank():
    global _local_comm, _local_rank, _local_size
    hostname = socket.gethostname()
    hostid = int(hashlib.md5(hostname.encode('utf-8')).hexdigest()[:7], 16)
    _local_comm = MPI.Comm.Split(_world_comm, hostid)
    _local_rank = _local_comm.Get_rank()
    _local_size = _local_comm.Get_size()

def _set_nccl_debugging_level():
    if is_azureml_compute():
        if args.debug_level > 0:
            os.environ['NCCL_DEBUG'] = 'INFO'
        else:
            del os.environ['NCCL_DEBUG']

try:
    from mpi4py import MPI

    _world_comm = MPI.COMM_WORLD
    _world_rank = _world_comm.Get_rank()
    _world_size = _world_comm.Get_size()

    _is_distributed = True
    _discover_local_rank()
    _set_torch_device()
    _set_nccl_debugging_level()


except ImportError:
    _world_rank = 0
    _world_size = 1
    _local_rank = 0
    _local_size = 1
    
    _is_distributed = False
