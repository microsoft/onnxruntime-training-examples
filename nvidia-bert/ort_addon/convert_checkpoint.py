# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import sys


# Pretraining with ORT introduces some prefixes in the model keys that need to be corrected for loading finetuning 
# classes like BertForQuestionAnswering. Some Moment states and prediction states are also removed since they are not needed for finetuning.
# Usage:
# convert_checkpoint.py <input checkpoint> <save location>
# Example : convert_checkpoint.py /workspace/checkpoints/bert_ort.pt /workspace/checkpoints/bert_ort_converted.pt 

inputFile=sys.argv[1]
outputFile=sys.argv[2]

stateDic = torch.load(inputFile)

from collections import OrderedDict
new_dict = OrderedDict((key.replace('wrapper_.', '').replace(
    'dense_act', 'dense'), value) for key, value in stateDic.get('model').items())
    
torch.save(new_dict,outputFile)

