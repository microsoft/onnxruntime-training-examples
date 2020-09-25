#!/bin/bash
set -e

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

num_gpus=${1:-1}
gpu_feed_batch_size=${2:-48}
gradient_accumulation_passes=${3:-1} 
precision=${4:-"fp16"}
training_steps=${5:-100}
save_checkpoint_steps=${6:-20}
seed=${7:-$RANDOM}

allreduce_post_accumulation="true"
resume_training="true"
create_logfile="true"
deepspeed_zero_stage="false"
learning_rate="6e-3"
warmup_proportion="0.2843"

PATH_TO_PHASE1_TRAINING_DATA=/bert_data/hdf5/128/train
DATA_DIR_PHASE1=$PATH_TO_PHASE1_TRAINING_DATA
BERT_CONFIG=bert_runner/bert_config.json
init_checkpoint="None"
RESULTS_DIR=./results

job_name="bert_pretraining"

if [ ! -d "$RESULTS_DIR" ] ; then
   mkdir $RESULTS_DIR
fi

if [ ! -d "$DATA_DIR_PHASE1" ] ; then
   echo "Warning! $DATA_DIR_PHASE1 directory missing. Training cannot start"
fi

PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC="--fp16"
elif [ "$precision" = "fp32" ] ; then
   PREC=""
else
   echo "Unknown <precision> argument"
   exit -2
fi

ACCUMULATE_GRADIENTS=""
if [ "$accumulate_gradients" == "true" ] ; then
   ACCUMULATE_GRADIENTS="--gradient_accumulation_passes=$gradient_accumulation_steps"
fi

DEEPSPEED_ZERO_STAGE=""
if [ "$deepspeed_zero_stage" == "true" ] ; then
   DEEPSPEED_ZERO_STAGE="--deepspeed_zero_stage"
fi

CHECKPOINT=""
if [ "$resume_training" == "true" ] ; then
   CHECKPOINT="--resume_from_checkpoint"
fi

ALL_REDUCE_POST_ACCUMULATION=""
if [ "$allreduce_post_accumulation" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION="--allreduce_post_accumulation"
fi

INIT_CHECKPOINT=""
if [ "$init_checkpoint" != "None" ] ; then
   INIT_CHECKPOINT="--init_checkpoint=$init_checkpoint"
fi

INPUT_DIR=$DATA_DIR_PHASE1
CMD=" -m bert_runner.run_pretraining"
CMD+=" --data_dir=$DATA_DIR_PHASE1"
CMD+=" --output_dir=$RESULTS_DIR"
CMD+=" --config_file=$BERT_CONFIG"
CMD+=" --gpu_feed_batch_size=$gpu_feed_batch_size"
CMD+=" --max_seq_length=128"
CMD+=" --max_predictions_per_seq=20"
CMD+=" --max_steps=$training_steps"
CMD+=" --warmup_proportion=$warmup_proportion"
CMD+=" --num_steps_per_checkpoint=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate"
CMD+=" --seed=$seed"
CMD+=" --resume_from_checkpoint"
CMD+=" --resume_step 40"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $CHECKPOINT"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION"
CMD+=" $INIT_CHECKPOINT"
CMD+=" $DEEPSPEED_ZERO_STAGE"
# CMD+=" --debug"

CMD="mpirun -n $num_gpus python $CMD"

if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $num_gpus \* $gpu_feed_batch_size \* $gradient_accumulation_passes)
  printf -v TAG "phase1_%s_gbs%d" "$precision" $GBS
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  LOGFILE=$RESULTS_DIR/$job_name.$TAG.$DATESTAMP.log
  printf "Log written to %s\n" "$LOGFILE"
fi

if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee $LOGFILE
fi