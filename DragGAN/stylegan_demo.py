import os
import torch
import cv2
import json

import numpy as np

from style_mapper import StyleMapper
from style_generator import StyleGenerator


def save_image(fn, img, handles=None, targets=None):
    img = img.detach().cpu().numpy()
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
    
    onnx_output_folder = os.path.join(current_script_folder, "onnx", model_name)
    if (not os.path.isdir(onnx_output_folder)):
        os.makedirs(onnx_output_folder)

    params_path = os.path.join(model_folder, model_name + ".json")
    gen_dict_path = os.path.join(model_folder, model_name + "_style_generator.pt")
    map_dict_path = os.path.join(model_folder, model_name + "_style_mapper.pt")

    with open(params_path) as f:
        params = json.load(f)

    ws_feats = 512
    feat_list = params['genFeatureList'] 
    map_layers = params['mapLayers']

    map = StyleMapper(num_layers=map_layers, ws_feats=ws_feats).cuda().eval()
    map.load_state_dict(torch.load(map_dict_path, map_location='cuda'))
    gen = StyleGenerator(feat_list, ws_feats).cuda().eval()
    gen.load_state_dict(torch.load(gen_dict_path))

    seed = 71
    z = np.random.RandomState(seed=seed).randn(1, ws_feats)
    z = z / np.sqrt(np.mean(z**2, axis=1, keepdims=True) + 1e-8)

    ws = map(torch.from_numpy(z.astype(np.float32)).cuda())
    ws = ws.repeat(1, 18, 1)

    img, F = gen(ws)

    image_filename = f'out/stylegan_seed_{seed}.png'
    print(f'saving image to [{image_filename}]')
    save_image(image_filename, img)

if __name__ == "__main__":
    main()
