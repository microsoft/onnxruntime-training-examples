from transformers import Wav2Vec2ForSequenceClassification, AutoConfig
import torch

# load config from the pretrained model
config = AutoConfig.from_pretrained("superb/wav2vec2-base-superb-sid")
model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-sid")

# modify last layer to output 2 classes
model.classifier = torch.nn.Linear(256, 2)


#export model to ONNX
dummy_input = torch.randn(1, 160000, requires_grad=True)
torch.onnx.export(model, dummy_input, "wav2vec.onnx",input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})

import onnx
import onnxruntime.training.onnxblock as onnxblock
from onnxruntime.training import artifacts

onnx_model = onnx.load("wav2vec.onnx")

requires_grad = ["classifier.weight", "classifier.bias"]
frozen_params = [
   param.name
   for param in onnx_model.graph.initializer
   if param.name not in requires_grad
]

# define custom loss function
class CustomCELoss(onnxblock.Block):
    def __init__(self):
        super().__init__()
        self.celoss = onnxblock.loss.CrossEntropyLoss()

    def build(self, logits, *args):
        return self.celoss(logits)


# Generate the training artifacts
artifacts.generate_artifacts(
    onnx_model,
    requires_grad=requires_grad,
    frozen_params=frozen_params,
    loss=CustomCELoss(),
    optimizer=artifacts.OptimType.AdamW,
    artifacts_dir="artifacts",
)


