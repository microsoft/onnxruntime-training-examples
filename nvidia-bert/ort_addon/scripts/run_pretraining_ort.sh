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

echo "Container nvidia build = " $NVIDIA_BUILD_ID

precision=${3:-"fp32"}
num_gpus=${4:-8}
gpu_memory_limit_gb=${26:-"32"}

seed=${12:-42}
job_name=${13:-"bert_lamb_pretraining"}
allreduce_post_accumulation=${14:-"true"}
allreduce_post_accumulation_fp16=${15:-"false"}

resume_training=${8:-"false"}
create_logfile=${9:-"true"}
accumulate_gradients=${10:-"true"}
deepspeed_zero_stage=${27:-"false"}

train_batch_size=${1:-4096} 
learning_rate=${2:-"6e-3"}
warmup_proportion=${5:-"0.2843"}
train_steps=${6:-7038}
save_checkpoint_steps=${7:-200}
gradient_accumulation_steps=${11:-64}

train_batch_size_phase2=${17:-4096}
learning_rate_phase2=${18:-"4e-3"}
warmup_proportion_phase2=${19:-"0.128"}
train_steps_phase2=${20:-1563}
gradient_accumulation_steps_phase2=${11:-512}

#PATH_TO_PHASE1_TRAINING_DATA=/data/128
PATH_TO_PHASE1_TRAINING_DATA=/data/wezhan/bert/hdf5/128/train
DATA_DIR_PHASE1=${22:-$PATH_TO_PHASE1_TRAINING_DATA}
BERT_CONFIG=bert_config.json
#CODEDIR=${24:-"/workspace/bert"}
CODEDIR=${24:-"/workspace/BERT"}
init_checkpoint=${25:-"None"}
RESULTS_DIR=$CODEDIR/results
CHECKPOINTS_DIR=$RESULTS_DIR/checkpoints

mkdir -p $CHECKPOINTS_DIR


if [ ! -d "$DATA_DIR_PHASE1" ] ; then
   echo "Warning! $DATA_DIR_PHASE1 directory missing. Training cannot start"
fi
if [ ! -d "$RESULTS_DIR" ] ; then
   echo "Error! $RESULTS_DIR directory missing."
   exit -1
fi
if [ ! -d "$CHECKPOINTS_DIR" ] ; then
   echo "Warning! $CHECKPOINTS_DIR directory missing."
   echo "Checkpoints will be written to $RESULTS_DIR instead."
   CHECKPOINTS_DIR=$RESULTS_DIR
fi
if [ ! -f "$BERT_CONFIG" ] ; then
   echo "Error! BERT large configuration file not found at $BERT_CONFIG"
   exit -1
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
   ACCUMULATE_GRADIENTS="--gradient_accumulation_steps=$gradient_accumulation_steps"
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

ALL_REDUCE_POST_ACCUMULATION_FP16=""
if [ "$allreduce_post_accumulation_fp16" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION_FP16="--allreduce_post_accumulation_fp16"
fi

INIT_CHECKPOINT=""
if [ "$init_checkpoint" != "None" ] ; then
   INIT_CHECKPOINT="--init_checkpoint=$init_checkpoint"
fi

echo $DATA_DIR_PHASE1
INPUT_DIR=$DATA_DIR_PHASE1
CMD=" $CODEDIR/run_pretraining_ort.py"
CMD+=" --input_dir=$DATA_DIR_PHASE1"
CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --config_file=$BERT_CONFIG"
CMD+=" --bert_model=bert-large-uncased"
CMD+=" --train_batch_size=$train_batch_size"
CMD+=" --max_seq_length=128"
CMD+=" --max_predictions_per_seq=20"
CMD+=" --max_steps=$train_steps"
CMD+=" --warmup_proportion=$warmup_proportion"
CMD+=" --num_steps_per_checkpoint=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate"
CMD+=" --seed=$seed"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $CHECKPOINT"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION_FP16"
CMD+=" $INIT_CHECKPOINT"
CMD+=" $DEEPSPEED_ZERO_STAGE"
CMD+=" --do_train"
CMD+=" --json-summary ${RESULTS_DIR}/dllogger.json "
CMD+=" --gpu_memory_limit_gb=$gpu_memory_limit_gb"

CMD="mpirun -n $num_gpus python $CMD"

if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $train_batch_size \* $num_gpus)
  printf -v TAG "pyt_bert_pretraining_phase1_%s_gbs%d" "$precision" $GBS
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  LOGFILE=$RESULTS_DIR/$job_name.$TAG.$DATESTAMP.log
  printf "Logs written to %s\n" "$LOGFILE"
fi

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee $LOGFILE
fi

set +x

echo "finished pretraining"

#Start Phase2

#PATH_TO_PHASE2_TRAINING_DATA=/data/512
PATH_TO_PHASE2_TRAINING_DATA=/data/wezhan/bert/hdf5/512/train
DATA_DIR_PHASE2=${23:-$PATH_TO_PHASE2_TRAINING_DATA}

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
   ACCUMULATE_GRADIENTS="--gradient_accumulation_steps=$gradient_accumulation_steps_phase2"
fi

ALL_REDUCE_POST_ACCUMULATION=""
if [ "$allreduce_post_accumulation" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION="--allreduce_post_accumulation"
fi

ALL_REDUCE_POST_ACCUMULATION_FP16=""
if [ "$allreduce_post_accumulation_fp16" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION_FP16="--allreduce_post_accumulation_fp16"
fi

echo $DATA_DIR_PHASE2
INPUT_DIR=$DATA_DIR_PHASE2
CMD=" $CODEDIR/run_pretraining_ort.py"
CMD+=" --input_dir=$DATA_DIR_PHASE2"
CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --config_file=$BERT_CONFIG"
CMD+=" --bert_model=bert-large-uncased"
CMD+=" --train_batch_size=$train_batch_size_phase2"
CMD+=" --max_seq_length=512"
CMD+=" --max_predictions_per_seq=80"
CMD+=" --max_steps=$train_steps_phase2"
CMD+=" --warmup_proportion=$warmup_proportion_phase2"
CMD+=" --num_steps_per_checkpoint=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate_phase2"
CMD+=" --seed=$seed"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $CHECKPOINT"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION_FP16"
CMD+=" $DEEPSPEED_ZERO_STAGE"
CMD+=" --do_train --phase2 --resume_from_checkpoint --phase1_end_step=$train_steps"
CMD+=" --json-summary ${RESULTS_DIR}/dllogger.json "
CMD+=" --gpu_memory_limit_gb=$gpu_memory_limit_gb"

CMD="mpirun -n $num_gpus python $CMD"

if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $train_batch_size_phase2 \* $num_gpus)
  printf -v TAG "pyt_bert_pretraining_phase2_%s_gbs%d" "$precision" $GBS
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  LOGFILE=$RESULTS_DIR/$job_name.$TAG.$DATESTAMP.log
  printf "Logs written to %s\n" "$LOGFILE"
fi

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee $LOGFILE
fi

set +x

echo "finished phase2"
