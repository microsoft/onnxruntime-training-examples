## Finetuning changes required to run GLUE or SQuAD

Some changes are required in the model state dictonary to run evaluation or finetuning like GLUE or SQuAD on the pretrained model. Run the following script to convert the ORT model checkpoint to NVIDIA BERT format.

```bash
python convert_checkpoint.py /workspace/checkpoints/<ort_pretrained_checkpoint> /workspace/checkpoints/<new checkpoint name>
```

Use the saved checkpoint created from above script for finetuning as described by [NVIDIA](https://github.com/NVIDIA/DeepLearningExamples/tree/96ff411ce84e679514947abe644d975a23867990/PyTorch/LanguageModeling/BERT#fine-tuning).

Hyper-parameters and arguements to run glue scores:
```
--task_name: 'MNLI',
--do_train: '',
--train_batch_size: 2,
--do_eval: '',
--eval_batch_size: 8,
--do_lower_case: '',
--data_dir: '/workspace/bert/data/glue/MNLI',
--bert_model: 'bert-large-uncased',
--seed: 2,
--init_checkpoint:'checkpoints/DLE_BERT_FP16_PyT_LAMB_92_hard_scaling_node.pt',
--warmup_proportion: 0.1,
--max_seq_length: 128,
--learning_rate: 3e-5,
--num_train_epochs: 3,
--max_steps: -1.0,
--vocab_file : '/workspace/bert/data/uncased_L-24_H-1024_A-16/vocab.txt',
--config_file : '/workspace/bert/bert_config.json',
--output_dir: 'output/MNLI',
--fp16 : '',
```
