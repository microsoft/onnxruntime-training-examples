import os
import cv2
import json

import numpy as np

from style_mapper import StyleMapper
from style_generator import StyleGenerator

import onnxruntime

def save_image(fn, img, handles=None, targets=None):
    img = np.clip(img * 127.5 + 127.5, 0, 255).astype(np.uint8)
    img = np.transpose(np.squeeze(img), axes=[1, 2, 0])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(fn, img)


def main():

    # available models: afhqcat  afhqdog  afhqwild  brecahad  ffhq  metfaces
    model_name = "metfaces"

    current_script_folder = os.path.dirname(__file__)
    base_models_folder = os.path.join(current_script_folder, "checkpoints")
    model_folder = os.path.join(base_models_folder, model_name)
    
    onnx_folder = os.path.join(current_script_folder, "onnx", model_name)

    onnx_mapper_filename = os.path.join(onnx_folder, "stylegan_mapper.onnx")
    onnx_generator_filename = os.path.join(onnx_folder, "stylegan_generator.onnx")

    execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    print(f"Creating ONNX Mapper session for [{onnx_mapper_filename}]")
    so = onnxruntime.SessionOptions()
    mapper_session = onnxruntime.InferenceSession(
        onnx_mapper_filename, so, providers=execution_providers
    )

    print(f"Creating ONNX Generator session for [{onnx_generator_filename}]")
    so = onnxruntime.SessionOptions()
    generator_session = onnxruntime.InferenceSession(
        onnx_generator_filename, so, providers=execution_providers
    )

    ws_feats = 512

    seed = 71
    z = np.random.RandomState(seed=seed).randn(1, ws_feats)
    z = z / np.sqrt(np.mean(z**2, axis=1, keepdims=True) + 1e-8)
    z = z.astype(np.float32)

    results = mapper_session.run(
        None,
        {
            "input": z,
        },
    )

    ws = results[0]
    ws = np.tile(ws, 18).reshape([1, 18, -1])

    results = generator_session.run(
        None,
        {
            "input": ws,
        },
    )

    img = results[0]
    F = results[1]

    image_filename = f'out/stylegan_seed_{seed}.png'
    print(f'saving image to [{image_filename}]')
    save_image(image_filename, img)    

if __name__ == "__main__":
    main()
