import io
import os
import struct

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from onnxruntime.training import onnxblock


# Prepare training artifacts
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def model_export(pt_model, model_path, do_constant_folding, mode):
    dummy_input = torch.randn(4, 3, 32, 32)
    torch.onnx.export(
        pt_model,
        dummy_input,
        model_path,
        verbose=True,
        export_params=True,
        do_constant_folding=do_constant_folding,
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        training=mode,
    )


def prep_classifier(model_loc):
    class ImageClassifier(onnxblock.TrainingModel):
        def __init__(self):
            super(ImageClassifier, self).__init__()
            self._loss = onnxblock.loss.CrossEntropyLoss()

        def build(self, input_name):
            return self._loss(input_name)

    pt_model = SimpleNet()
    train_model_bytes = io.BytesIO()
    model_export(pt_model, train_model_bytes, False, torch.onnx.TrainingMode.TRAINING)
    model = onnx.load_model_from_string(train_model_bytes.getvalue())

    # Create Training Artifacts
    classifier = ImageClassifier()
    with onnxblock.onnx_model(model) as accessor:
        _ = classifier(model.graph.output[0].name)
        eval_model = accessor.eval_model

    # Create the model_loc directory
    if not os.path.exists(model_loc):
        os.makedirs(model_loc)

    # Save the generated onnx models
    output_model_path = os.path.join(model_loc, "classifier_training_model.onnx")
    output_eval_model_path = os.path.join(model_loc, "classifier_eval_model.onnx")
    onnx.save(model, output_model_path)
    onnx.save(eval_model, output_eval_model_path)

    optimizer = onnxblock.optim.AdamW(clip_grad=None)
    optimizer_model = None
    with onnxblock.onnx_model() as accessor:
        _ = optimizer(classifier.parameters())
        optimizer_model = accessor.model

    # save the optimizer model
    output_optimizer_model_path = os.path.join(model_loc, "adamw_optimizer.onnx")
    onnx.save(optimizer_model, output_optimizer_model_path)

    # prep checkpoint
    output_checkpoint_path = os.path.join(model_loc, "checkpoint")
    onnxblock.save_checkpoint(classifier.parameters(), output_checkpoint_path)


# Preprocess datasets
def save_data_as_bin(np_data_list, out_filename, data_type):
    arr = np.concatenate(np_data_list).ravel()
    with open(out_filename, "wb") as handle:
        if data_type == "float":
            handle.write(struct.pack("<%df" % len(arr), *arr))
        else:
            handle.write(struct.pack("<%di" % len(arr), *arr))


# Save batches of preprocessed data
# step_size: determines number of batches saved in a single file
# stop_at: total number of batches to pick
def process_and_save_data(data_loader, data_loc, step_size):
    inputs_pack, labels_pack = [], []
    for i, data in enumerate(data_loader, 0):
        # get the inputs; data is a tuple of (inputs, labels)
        inputs, labels = data
        inputs_pack.append(inputs.detach().numpy().flatten())
        labels_pack.append(labels.detach().numpy().flatten())

        if (i + 1) % step_size == 0:
            save_data_as_bin(inputs_pack, os.path.join(data_loc, "input_" + str(i // step_size) + ".bin"), "float")
            save_data_as_bin(labels_pack, os.path.join(data_loc, "labels_" + str(i // step_size) + ".bin"), "int")
            inputs_pack, labels_pack = [], []


def get_data(batch_size, data_loc, output_loc):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(
        root=os.path.join(data_loc, "data"), train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(
        root=os.path.join(data_loc, "data"), train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create the output_loc directory
    train_data_loc = os.path.join(output_loc, "train_data")
    if not os.path.exists(train_data_loc):
        os.makedirs(train_data_loc)

    test_data_loc = os.path.join(output_loc, "test_data")
    if not os.path.exists(test_data_loc):
        os.makedirs(test_data_loc)

    process_and_save_data(trainloader, train_data_loc, 250)
    process_and_save_data(testloader, test_data_loc, 250)


if __name__ == "__main__":
    # Prep training artifacts
    prep_classifier(model_loc=os.path.join(os.getcwd(), "assets"))

    # Preprocess data
    get_data(batch_size=4, data_loc=os.getcwd(), output_loc=os.path.join(os.getcwd(), "assets"))
