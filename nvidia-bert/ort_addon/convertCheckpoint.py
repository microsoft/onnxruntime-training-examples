import torch
import sys

# Usage:
# convertCheckpoint.py <input checkpoint> <save location>
# Example : convertCheckpoint.py /workspace/checkpoints/bert_ort.pt /workspace/checkpoints/bert_ort_converted.pt 

inputFile=sys.argv[1]
outputFile=sys.argv[2]

stateDic = torch.load(inputFile)
 
from collections import OrderedDict
new_dict = OrderedDict((key.replace('Moment_1_model_', '').replace('model_.', ''), value) for key, value in stateDic.items() if not (("Moment" in key) or ("_fp16" in key) or ("predictions" in key) or ("seq_relationship" in key) or "Step" in key))

torch.save(new_dict,outputFile)

