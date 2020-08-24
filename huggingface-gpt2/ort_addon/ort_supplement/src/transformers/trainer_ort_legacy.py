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
from onnxruntime.capi.ort_trainer import ORTTrainer, IODescription, ModelDescription
from onnxruntime.capi.ort_trainer import LossScaler

from .data.data_collator import DataCollator, DefaultDataCollator
from .modeling_utils import PreTrainedModel
from .training_args import TrainingArguments
from .trainer import PredictionOutput, TrainOutput, EvalPrediction, set_seed

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

class linear_schedule_with_warmup():
    def __init__(self, num_warmup_steps, num_training_steps, last_epoch=-1):
        """ Create a schedule with a learning rate that decreases linearly after
        linearly increasing during a warmup period.
        """
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.last_epoch = last_epoch

    def lr_lambda(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        return max(
            0.0, float(self.num_training_steps - current_step) / float(max(1, self.num_training_steps - self.num_warmup_steps))
        )

    def get_lr_this_step(self, current_step, base_lr):

        return self.lr_lambda(current_step) * base_lr

class OrtTrainer:
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
        Trainer is a simple but feature-complete training and eval loop for PyTorch,
        optimized for Transformers.

        Args:
            prediction_loss_only:
                (Optional) in evaluation and prediction, only return the loss
        """
        self.model = model
        self.args = args
        if data_collator is not None:
            self.data_collator = data_collator
        else:
            self.data_collator = DefaultDataCollator()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.prediction_loss_only = prediction_loss_only

        if is_tensorboard_available() and self.is_world_master():
            self.tb_writer = SummaryWriter(log_dir=self.args.logging_dir)
        if not is_tensorboard_available():
            logger.warning(
                "You are instantiating a Trainer but Tensorboard is not installed. You should consider installing it."
            )
        set_seed(self.args.seed)
        onnxruntime.set_seed(self.args.seed)
        # Create output directory if needed
        if self.is_world_master():
            os.makedirs(self.args.output_dir, exist_ok=True)
        
        torch.cuda.set_device(self.args.local_rank)

        self.ort_model = self.to_ort_model(model, model.config, args)

    def update_torch_model(self,):
        if self.ort_model:
            logger.info(
                "Updating weights of torch model from ORT model."
            )
            ort_state_dict = self.ort_model.state_dict()
            self.model.load_state_dict(ort_state_dict, strict=False)
        else:
            logger.warning(
                "No ORT model found to update weights from, assuming torch model is up to date."
            )

    def gpt2_model_description(self,n_head, vocab_size, n_hidden, n_layer, n_ctx, batch_size):
    
        logger.info("****num of head is: {}".format(n_head))
        logger.info("****vocab size is: {}".format(vocab_size))
        logger.info("****num of hidden layer is: {}".format(n_hidden))
        logger.info("****num of layer is: {}".format(n_layer))
        logger.info("****seq length is: {}".format(n_ctx))

        input_ids_desc = IODescription('input_ids', [batch_size, n_ctx], torch.int64, num_classes = vocab_size)    
        labels_desc = IODescription('labels', [batch_size, n_ctx], torch.int64, num_classes = vocab_size)
        
        loss_desc = IODescription('loss', [], torch.float32)
        
        return ModelDescription([input_ids_desc, labels_desc],
                                [loss_desc])

    def ort_trainer_learning_rate_description(self):
        return IODescription('Learning_Rate', [1,], torch.float32)

    def to_ort_model(self,model, config, args):
        model_desc = self.gpt2_model_description(config.n_head, config.vocab_size, config.n_embd, config.n_layer, config.n_ctx, args.per_gpu_train_batch_size)
        learning_rate_description = self.ort_trainer_learning_rate_description()

        def map_optimizer_attributes(name):
            no_decay_keys = ["bias", "gamma", "beta", "LayerNorm"]
            no_decay = False
            for no_decay_key in no_decay_keys:
                if no_decay_key in name:
                    no_decay = True
                    break
            if no_decay:
                return {"alpha": 0.9, "beta": 0.999, "lambda": 0.0, "epsilon": args.adam_epsilon}
            else:
                return {"alpha": 0.9, "beta": 0.999, "lambda": args.weight_decay, "epsilon": args.adam_epsilon}

        from onnxruntime.capi._pybind_state import set_cuda_device_id, set_arena_extend_strategy, ArenaExtendStrategy
        set_arena_extend_strategy(ArenaExtendStrategy.kSameAsRequested)
        set_cuda_device_id(self.args.local_rank)

        model = ORTTrainer(model, None, model_desc, "AdamOptimizer",
            map_optimizer_attributes,
            learning_rate_description,
            args.device,  
            gradient_accumulation_steps=args.gradient_accumulation_steps, 
            world_rank = self.args.world_rank,
            world_size = self.args.world_size,
            use_mixed_precision =  self.args.fp16,
            allreduce_post_accumulation = True,
            _opset_version=12
            )
        
        logger.info("****************************Model converted to ORT")
        return model

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

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        return DataLoader(
            eval_dataset if eval_dataset is not None else self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator.collate_batch,
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
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

        scheduler = linear_schedule_with_warmup(num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        loss_scaler = LossScaler(self.ort_model.loss_scale_input_name, True, up_scale_window=2000, loss_scale=float(1 << 20)) if self.args.fp16 else 1

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
                    #skip incomplete batch
                    logger.info('Skipping incomplete batch...')
                    continue
                
                learning_rate = torch.tensor([scheduler.get_lr_this_step(global_step, base_lr = self.args.learning_rate)])
                loss, all_finite = self._training_step(model, inputs, learning_rate, loss_scaler)
                tr_loss += loss
                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                    ):

                    if self.args.fp16:
                        loss_scaler.update_loss_scale(all_finite.item())

                    global_step += 1
                    global_batch_train_duration = time.time() - global_batch_train_start
                    global_batch_train_start = time.time()

                    if self.args.local_rank in [-1, 0]:
                        if (self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0) or (
                            global_step == 1 and self.args.logging_first_step
                        ):
                            logs = {}
                            loss_avg = (tr_loss - logging_loss) / (self.args.logging_steps * self.args.gradient_accumulation_steps)
                            logs["learning_rate"] = learning_rate.item()
                            logs["loss"] = loss_avg
                            logs["global_step"] = global_step
                            logs["global_step_time"] = global_batch_train_duration
                            logging_loss = tr_loss

                            if self.tb_writer:
                                for k, v in logs.items():
                                    self.tb_writer.add_scalar(k, v, global_step)
                                    run.log(k,v)
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
                            # self._rotate_checkpoints()
                            
                if self.args.max_steps > 0 and global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break
            if self.args.max_steps > 0 and global_step > self.args.max_steps:
                train_iterator.close()
                break

        if self.tb_writer:
            self.tb_writer.close()
        self.update_torch_model()
        del(self.ort_model)
        self.ort_model = None
        
        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(global_step, tr_loss / global_step)

    def _training_step(
        self, model: nn.Module, inputs: Dict[str, torch.Tensor], learning_rate, loss_scaler
        ) -> float:
        model.train()

        if self.args.fp16:
            loss_scale = torch.tensor([loss_scaler.loss_scale_])
            result = model(inputs['input_ids'],inputs['labels'], learning_rate, loss_scale)
        else:
            result = model(inputs['input_ids'],inputs['labels'], learning_rate)
        
        all_finite = None
        if isinstance(result, (list, tuple)):
            loss = result[0]
            all_finite = result[-1]
        else:
            loss = result

        return loss.item(), all_finite

    def is_world_master(self) -> bool:
        """
        This will be True only in one process, even in distributed mode,
        even when training on multiple machines.
        """
        return self.args.local_rank == -1 or torch.distributed.get_rank() == 0

    def save_model(self, output_dir: Optional[str] = None):
        """
        Saving best-practices: if you use default names for the model,
        you can reload it using from_pretrained().

        Will only save from the master process.
        """
        if self.is_world_master():
            self._save(output_dir)

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

    def _sorted_checkpoints(self, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(self.args.output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)

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

        output = self._prediction_loop(eval_dataloader, description="Evaluation")
        return output.metrics

    def predict(self, test_dataset: Dataset) -> PredictionOutput:
        """
        Run prediction and return predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in evaluate().
        """
        test_dataloader = self.get_test_dataloader(test_dataset)
        return self._prediction_loop(test_dataloader, description="Prediction")

    def _prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
        ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only
        self.update_torch_model()
        # multi-gpu eval
        if self.args.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(self.model)
        else:
            model = self.model
        
        model.to(self.args.device)
        if self.is_world_master():
            logger.info("***** Running %s *****", description)
            logger.info("  Num examples = %d", len(dataloader.dataset))
            logger.info("  Batch size = %d", dataloader.batch_size)
        eval_losses: List[float] = []
        preds: np.ndarray = None
        label_ids: np.ndarray = None
        model.eval()

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(inputs.get(k) is not None for k in ["labels", "masked_lm_labels"])

            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]

            if not prediction_loss_only:
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                if inputs.get("labels") is not None:
                    if label_ids is None:
                        label_ids = inputs["labels"].detach().cpu().numpy()
                    else:
                        label_ids = np.append(label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics["loss"] = np.mean(eval_losses)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)
