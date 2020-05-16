## Finetuning changes required to run GLUE or SQuAD

Some changes maybe required in the model state dict to run evaluation or finetuning like GLUE or SQuAD on the pretrained model. Run the following script to convert your model checkpoint to nvidia bert format.

```bash
python convertCheckpoint.py /workspace/checkpoints/<ort_pretrained_checkpoint>   /workspace/checkpoints/<new checkpoint name>
```

Use the saved checkpoint created from above script for finetuning as described by [NVIDIA](https://github.com/NVIDIA/DeepLearningExamples/tree/96ff411ce84e679514947abe644d975a23867990/PyTorch/LanguageModeling/BERT#fine-tuning).
