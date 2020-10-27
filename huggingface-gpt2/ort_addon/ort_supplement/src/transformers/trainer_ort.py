import json
import time
import logging
import os
import random
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm, trange

import onnxruntime
from onnxruntime.training import _utils, amp, checkpoint, optim, orttrainer, TrainStepInfo
from .data.data_collator import DataCollator, DefaultDataCollator
from .modeling_utils import PreTrainedModel
from .training_args import TrainingArguments
from .trainer import PredictionOutput, TrainOutput, EvalPrediction, set_seed, Trainer

from azureml.core.run import Run
# get the Azure ML run object
run = Run.get_context()


try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter

        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False


def is_tensorboard_available():
    return _has_tensorboard


logger = logging.getLogger(__name__)

PREFIX_CHECKPOINT_DIR = "ort_checkpoint"


class OrtTrainer(Trainer):
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch,
    optimized for Transformers.
    """

    model: PreTrainedModel
    args: TrainingArguments
    data_collator: DataCollator
    train_dataset: Optional[Dataset]
    eval_dataset: Optional[Dataset]
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
    prediction_loss_only: bool
    tb_writer: Optional["SummaryWriter"] = None

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        prediction_loss_only=False,
    ):
        """
        OrtTrainer is a simple but feature-complete training and eval loop for ORT,
        optimized for Transformers.

        Args:
            prediction_loss_only:
                (Optional) in evaluation and prediction, only return the loss
        """
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, compute_metrics, prediction_loss_only)
        onnxruntime.set_seed(self.args.seed)
        torch.cuda.set_device(self.args.local_rank)

    def update_torch_model(self,):
        if self.ort_model:
            logger.info(
                "Updating weights of torch model from ORT model."
            )
            ort_state_dict = checkpoint.experimental_state_dict(self.ort_model)
            self.model.load_state_dict(ort_state_dict, strict=False)
        else:
            logger.warning(
                "No ORT model found to update weights from, assuming torch model is up to date."
            )

    def gpt2_model_description(self, n_head, vocab_size, n_hidden, n_layer, n_ctx, batch_size):

        logger.info("****num of head is: {}".format(n_head))
        logger.info("****vocab size is: {}".format(vocab_size))
        logger.info("****num of hidden layer is: {}".format(n_hidden))
        logger.info("****num of layer is: {}".format(n_layer))
        logger.info("****seq length is: {}".format(n_ctx))

        # We are using hard-coded values for batch size and sequence length in order to trigger 
        # memory planning in ORT, which would reduce the memory footprint during training.
        # Alternatively, one can set these as symbolic dims 'batch_size' and 'n_ctx' to be able
        # to use dynamic input sizes.
        model_desc = {'inputs': [('input_ids', [batch_size, n_ctx]),
                                 ('labels', [batch_size, n_ctx])],
                      'outputs': [('loss', [], True)]}
        return model_desc

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = (
            RandomSampler(self.train_dataset) if self.args.local_rank == -1 else DistributedSampler(self.train_dataset)
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_gpu_train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator.collate_batch,
        )

    def train(self, model_path: Optional[str] = None):
        """
        Main training entry point.

        Args:
            model_path:
                (Optional) Local path to model if model to train has been instantiated from a local path
                If present, we will try reloading the optimizer/scheduler states from there.
        """
        train_dataloader = self.get_train_dataloader()

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        config = self.model.config
        model_desc = self.gpt2_model_description(config.n_head, 
                                                    config.vocab_size, 
                                                    config.n_embd, 
                                                    config.n_layer, 
                                                    config.n_ctx, 
                                                    self.args.per_gpu_train_batch_size)

        from onnxruntime.capi._pybind_state import set_arena_extend_strategy, ArenaExtendStrategy
        set_arena_extend_strategy(ArenaExtendStrategy.kSameAsRequested)

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

        optim_config = optim.AdamConfig(params=[{'params' : [n for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                                                 'lambda_coef': 0.0
                                                 }],
                                        lr=self.args.learning_rate, alpha=0.9, beta=0.999, lambda_coef=self.args.weight_decay, epsilon=self.args.adam_epsilon)

        warmup = self.args.warmup_steps / t_total
        lr_scheduler = optim.lr_scheduler.LinearWarmupLRScheduler(total_steps=t_total, warmup=warmup)
        loss_scaler = amp.DynamicLossScaler(automatic_update=True,
                                            loss_scale=float(1 << 20),
                                            up_scale_window=2000,
                                            min_loss_scale=1.0,
                                            max_loss_scale=float(1 << 24)) if self.args.fp16 else None

        opts = orttrainer.ORTTrainerOptions({
            'device': {'id': str(self.args.device)},
            'distributed': {
                'world_rank': self.args.world_rank,
                'world_size': self.args.world_size,
                'local_rank': self.args.local_rank,
                'allreduce_post_accumulation': True},
            'mixed_precision': {'enabled': self.args.fp16,
                                'loss_scaler': loss_scaler},
            'batch': {'gradient_accumulation_steps': self.args.gradient_accumulation_steps},
            'lr_scheduler': lr_scheduler})

        self.ort_model = orttrainer.ORTTrainer(self.model, model_desc, optim_config, None, options=opts)

        logger.info("****************************Model converted to ORT")
        model = self.ort_model

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())

        # Train!
        if self.is_world_master():
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_dataloader.dataset))
            logger.info("  Num Epochs = %d", num_train_epochs)
            logger.info("  Instantaneous batch size per GPU = %d", self.args.per_gpu_train_batch_size)
            logger.info(
                "  Total train batch size (w. parallel, distributed & accumulation) = %d",
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (self.args.world_size if self.args.local_rank != -1 else 1),
            )
            logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
            logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = 0.0
        logging_loss = 0.0
        global_batch_train_start = time.time()

        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=self.args.local_rank not in [-1, 0],
        )
        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=self.args.local_rank not in [-1, 0])
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                if len(inputs['input_ids']) < self.args.per_gpu_train_batch_size:
                    # skip incomplete batch
                    logger.info('Skipping incomplete batch...')
                    continue

                tr_loss += self._training_step(model, inputs)
                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                    ):

                    global_step += 1
                    global_batch_train_duration = time.time() - global_batch_train_start
                    global_batch_train_start = time.time()

                    if self.args.local_rank in [-1, 0]:
                        if (self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0) or (
                            global_step == 1 and self.args.logging_first_step
                        ):
                            logs = {}
                            loss_avg = (tr_loss - logging_loss) / (self.args.logging_steps * self.args.gradient_accumulation_steps)
                            logs["learning_rate"] = lr_scheduler.get_last_lr()[0]
                            logs["loss"] = loss_avg.item()
                            logs["global_step"] = global_step
                            logs["global_step_time"] = global_batch_train_duration
                            logging_loss = tr_loss.clone()

                            if self.tb_writer:
                                for k, v in logs.items():
                                    self.tb_writer.add_scalar(k, v, global_step)
                                    run.log(k, v)
                            epoch_iterator.write(json.dumps({**logs, **{"step": global_step}}))

                        if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                            # In all cases (even distributed/parallel), self.model is always a reference
                            # to the model we want to save.
                            if hasattr(model, "module"):
                                assert model.module is self.ort_model
                            else:
                                assert model is self.ort_model
                            # Save model checkpoint
                            output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{global_step}")
                            self.save_model(output_dir)
                            self._rotate_checkpoints()

                if self.args.max_steps > 0 and global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break
            if self.args.max_steps > 0 and global_step > self.args.max_steps:
                train_iterator.close()
                break

        if self.tb_writer:
            self.tb_writer.close()
        self.update_torch_model()

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(global_step, tr_loss / global_step)

    def _training_step(
        self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> float:

        loss = model.train_step(**inputs)

        return loss

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)

        self.update_torch_model()
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")
        self.model.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def evaluate_in_ORT(
        self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and return metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent.

        Args:
            eval_dataset: (Optional) Pass a dataset if you wish to override
            the one on the instance.
        Returns:
            A dict containing:
                - the eval loss
        """
        self.infer_sess = None
        onnx_model_path = os.path.join(self.args.output_dir, "final_model.onnx")
        output_names = [o_desc.name for o_desc in self.ort_model.model_desc.outputs]

        # The eval batch size should be the same as finetuned onnx model's batch size
        # as the graph exported for training is being used for inference
        # Alternatively, we can export the onnx graph again to use a symbolic batch size
        assert self.args.per_gpu_eval_batch_size == self.args.per_gpu_train_batch_size
        
        # save the onnx graph 
        self.ort_model.save_as_onnx(onnx_model_path)
        
        # delete the training model to free up GPU memory
        del(self.ort_model)
        self.ort_model = None

        # create the inference session
        self.infer_sess = onnxruntime.InferenceSession(onnx_model_path)

        # load the eval dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        description = "Evaluation"

        if self.is_world_master():
            logger.info("***** Running %s *****", description)
            logger.info("  Num examples = %d", len(eval_dataloader.dataset))
            logger.info("  Batch size = %d", eval_dataloader.batch_size)
        eval_losses: List[float] = []
        
        for inputs in tqdm(eval_dataloader, desc=description):

            # for the last batch, pad to the batch size.
            if len(inputs['input_ids']) < self.args.per_gpu_eval_batch_size:
                pad_len = self.args.per_gpu_eval_batch_size
                inputs['input_ids'] = inputs['input_ids'].repeat(pad_len, 1)
                inputs['labels'] = inputs['labels'].repeat(pad_len, 1)

            step_eval_loss = self.infer_sess.run(output_names,
                                                 {"input_ids": inputs["input_ids"].numpy(),
                                                  "labels": inputs["labels"].numpy()
                                                  })
            eval_losses += [step_eval_loss[0]]

        metrics = {}
        if len(eval_losses) > 0:
            metrics["loss"] = np.mean(eval_losses)

        return metrics
    
    def evaluate(
        self, eval_dataset: Optional[Dataset] = None, prediction_loss_only: Optional[bool] = None
    ) -> Dict[str, float]:
        """
        Run evaluation and return metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent.

        Args:
            eval_dataset: (Optional) Pass a dataset if you wish to override
            the one on the instance.
        Returns:
            A dict containing:
                - the eval loss
                - the potential metrics computed from the predictions
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        # update the torch model weights and delete the ort training model to free up GPU memory
        self.update_torch_model()
        del(self.ort_model)
        self.ort_model = None
        
        output = self._prediction_loop(eval_dataloader, description="Evaluation")
        return output.metrics
