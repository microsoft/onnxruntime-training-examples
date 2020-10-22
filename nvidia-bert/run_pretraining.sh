#!/bin/bash
set -e

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

# computation
num_gpus=${1:-1}
gpu_feed_batch_size=${2:-48}
gradient_accumulation_passes=${3:-8} 
precision=${4:-"fp16"}
allreduce_post_accumulation="true"
deepspeed_zero_stage="false"
learning_rate="6e-3"
warmup_proportion="0.2843"

# administrative
path_to_phase1_training_data=<path-to-phase1-hdf5-training-data>
path_to_phase2_training_data=<path-to-phase2-hdf5-training-data>
phase="phase1"
training_steps=${5:-50}
seed=${7:-$RANDOM}
results_dir=./results
create_logfile="true"
debug_output="false"
init_checkpoint="None"
skip_checkpointing="false"
save_checkpoint_interval=${6:-200}
resume_from_step=0
logging_step_interval=1
smooth_throughput_passes=16
bert_config=bert_config.json

# basic validation
if [ ! -d "$results_dir" ] ; then
   mkdir $results_dir
fi
if [ ! -d "$path_to_phase1_training_data" ] ; then
   echo "Warning! $path_to_phase1_training_data directory missing. Training cannot start"
fi

# construct complex or optional flags
sequence_description_flag=""
if [ "$phase" = "phase1" ] ; then
   sequence_description_flag=" --max_seq_length=128 --max_predictions_per_seq=20"
   path_to_training_data=$path_to_phase1_training_data
elif [ "$phase" = "phase2" ] ; then
   sequence_description_flag=" --max_seq_length=512 --max_predictions_per_seq=80"
   path_to_training_data=$path_to_phase2_training_data
else
   echo "Unknown <phase> argument"
   exit -2
fi
precision_flag=""
if [ "$precision" = "fp16" ] ; then
   precision_flag="--fp16"
elif [ "$precision" = "fp32" ] ; then
   precision_flag=""
else
   echo "Unknown <precision> argument"
   exit -2
fi
accumulate_gradients_flag=""
if (($gradient_accumulation_passes > 1)) ; then
   accumulate_gradients_flag="--gradient_accumulation_passes=$gradient_accumulation_passes"
fi
deepspeed_zero_stage_flag=""
if [ "$deepspeed_zero_stage" == "true" ] ; then
   deepspeed_zero_stage_flag="--deepspeed_zero_stage"
fi
debug_flag=""
if [ "$debug_output" == "true" ] ; then
   debug_flag="--debug"
fi
resume_from_step_flag=""
if (($resume_from_step > 0)) ; then
   resume_from_step_flag="--resume_from_step $resume_from_step"
fi
allreduce_post_accumulation_flag=""
if [ "$allreduce_post_accumulation" == "true" ] ; then
   allreduce_post_accumulation_flag="--allreduce_post_accumulation"
fi
init_checkpoint_flag=""
if [ "$init_checkpoint" != "None" ] ; then
   init_checkpoint_flag="--init_checkpoint=$init_checkpoint"
fi
skip_checkpointing_flag=""
if [ "$skip_checkpointing" == "true" ] ; then
   skip_checkpointing_flag="--skip_checkpointing"
fi

# construct job command
cmd=" -m bert_runner.run_pretraining"
cmd+=" --data_dir=$path_to_training_data"
cmd+=" --output_dir=$results_dir"
cmd+=" --config_file=$bert_config"
cmd+=" $sequence_description_flag"
cmd+=" --gpu_feed_batch_size=$gpu_feed_batch_size"
cmd+=" --max_steps=$training_steps"
cmd+=" --warmup_proportion=$warmup_proportion"
cmd+=" --num_steps_per_checkpoint=$save_checkpoint_interval"
cmd+=" --learning_rate=$learning_rate"
cmd+=" --seed=$seed"
cmd+=" --num_passes_to_smooth_output=$smooth_throughput_passes"
cmd+=" --num_steps_per_log_entry=$logging_step_interval"
cmd+=" $precision_flag"
cmd+=" $accumulate_gradients_flag"
cmd+=" $allreduce_post_accumulation_flag"
cmd+=" $init_checkpoint_flag"
cmd+=" $resume_from_step_flag"
cmd+=" $skip_checkpointing_flag"
cmd+=" $deepspeed_zero_stage_flag"
cmd+=" $debug_flag"

cmd="mpirun -n $num_gpus python $cmd"

# construct log file name
job_name="bert_pretraining"
if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $num_gpus \* $gpu_feed_batch_size \* $gradient_accumulation_passes)
  printf -v tag "phase1_%s_gbs%d" "$precision" $GBS
  datestamp=`date +'%y%m%d%H%M%S'`
  logfile=$results_dir/$job_name.$tag.$datestamp.log
  printf "Log written to %s\n" "$logfile"
fi

# submit job with logging
runtraced() {
    echo "$@"
    "$@"
}
if [ -z "$logfile" ] ; then
   runtraced $cmd
else
   (
     runtraced $cmd
   ) |& tee $logfile
fi
