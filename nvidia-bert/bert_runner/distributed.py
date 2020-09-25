# Copyright (c) 2020 Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import hashlib
import os
import socket

def is_distributed():
    global _is_distributed
    return _is_distributed

def is_world_leader():
    global world_rank
    return world_rank == 0

def is_local_leader():
    global local_rank
    return local_rank == 0

def world_barrier():
    if is_distributed():
        world_comm.barrier()

def is_azureml_compute():
    return 'AZ_BATCH_MASTER_NODE' in os.environ.keys() or 'AZ_BATCHAI_MPI_MASTER_NODE' in os.environ.keys()

hostname = socket.gethostname()        
try:
    from mpi4py import MPI

    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.Get_rank()
    world_size = world_comm.Get_size()

    # only keep 7 hex digits to prevent overflow to signed int32    
    hostid = int(hashlib.md5(hostname.encode('utf-8')).hexdigest()[:7], 16)
    local_comm = MPI.Comm.Split(world_comm, hostid)
    local_rank = local_comm.Get_rank()
    local_size = local_comm.Get_size()

    _is_distributed = True

except ImportError:
    world_rank = 0
    world_size = 1
    local_rank = 0
    local_size = 1
    
    _is_distributed = False