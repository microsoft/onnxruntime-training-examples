import torch
import numpy as np
import os
import json

import pickle
from collections import OrderedDict

torch.set_grad_enabled(False)

def get_conv0_weights(old_dict, new_dict, idx):
    resolution = 2 ** (idx+2)
    noise_strength = old_dict[f'synthesis.b{resolution}.conv0.noise_strength']
    noise_const    = old_dict[f'synthesis.b{resolution}.conv0.noise_const']
    mod_weight     = old_dict[f'synthesis.b{resolution}.conv0.affine.weight']
    blur_weight    = old_dict[f'synthesis.b{resolution}.conv0.resample_filter']

    # Pre-normalize, https://github.com/NVlabs/stylegan2-ada-pytorch/blob/d72cc7d041b42ec8e806021a205ed9349f87c6a4/training/networks.py#L102
    new_dict[f'block_list.{idx}.conv0.conv_mod.weight'] = mod_weight / np.sqrt(mod_weight.shape[1])
    new_dict[f'block_list.{idx}.conv0.conv_mod.bias']   = old_dict[f'synthesis.b{resolution}.conv0.affine.bias']

    # Pre-scale noise https://github.com/NVlabs/stylegan2-ada-pytorch/blob/d72cc7d041b42ec8e806021a205ed9349f87c6a4/training/networks.py#L296
    new_dict[f'block_list.{idx}.conv0.noise']           = (noise_strength * noise_const)[None, None, ...]

    # Pre-scale blur kernel, https://github.com/NVlabs/stylegan2-ada-pytorch/blob/d72cc7d041b42ec8e806021a205ed9349f87c6a4/torch_utils/ops/upfirdn2d.py#L343
    # x, y dimensions both double in size so scale by 2x2
    new_dict[f'block_list.{idx}.conv0.blur.weight']     = blur_weight[None, None, ...] * 4

    new_dict[f'block_list.{idx}.conv0.conv_weight']     = old_dict[f'synthesis.b{resolution}.conv0.weight']
    new_dict[f'block_list.{idx}.conv0.act_bias']        = old_dict[f'synthesis.b{resolution}.conv0.bias'][None, ..., None, None]
    return new_dict


def get_conv1_weights(old_dict, new_dict, idx):
    resolution = 2 ** (idx+2)
    noise_strength = old_dict[f'synthesis.b{resolution}.conv1.noise_strength']
    noise_const    = old_dict[f'synthesis.b{resolution}.conv1.noise_const']
    mod_weight     = old_dict[f'synthesis.b{resolution}.conv1.affine.weight']

    # Pre-normalize, https://github.com/NVlabs/stylegan2-ada-pytorch/blob/d72cc7d041b42ec8e806021a205ed9349f87c6a4/training/networks.py#L102
    new_dict[f'block_list.{idx}.conv1.conv_mod.weight'] = mod_weight / np.sqrt(mod_weight.shape[1])
    new_dict[f'block_list.{idx}.conv1.conv_mod.bias']   = old_dict[f'synthesis.b{resolution}.conv1.affine.bias']

    # Pre-scale noise https://github.com/NVlabs/stylegan2-ada-pytorch/blob/d72cc7d041b42ec8e806021a205ed9349f87c6a4/training/networks.py#L296
    new_dict[f'block_list.{idx}.conv1.noise']           = (noise_strength * noise_const)[None, None, ...]

    new_dict[f'block_list.{idx}.conv1.conv_weight']     = old_dict[f'synthesis.b{resolution}.conv1.weight']
    new_dict[f'block_list.{idx}.conv1.act_bias']        = old_dict[f'synthesis.b{resolution}.conv1.bias'][None, ..., None, None]
    return new_dict


def get_rgb_weights(old_dict, new_dict, idx):
    resolution  = 2 ** (idx+2)
    mod_weight  = old_dict[f'synthesis.b{resolution}.torgb.affine.weight']
    mod_bias    = old_dict[f'synthesis.b{resolution}.torgb.affine.bias']

    conv_weight = old_dict[f'synthesis.b{resolution}.torgb.weight']
    up_weight   = old_dict[f'synthesis.b{resolution}.resample_filter'][None, None, ...]

    # Two scaling factors are applied here
    # First one only applies to weight
    # https://github.com/NVlabs/stylegan2-ada-pytorch/blob/d72cc7d041b42ec8e806021a205ed9349f87c6a4/training/networks.py#L102
    # Second one applies to bias as well
    # https://github.com/NVlabs/stylegan2-ada-pytorch/blob/d72cc7d041b42ec8e806021a205ed9349f87c6a4/training/networks.py#L318
    mod_scale = np.sqrt(conv_weight.shape[1] * up_weight.shape[0] * up_weight.shape[1])
    new_dict[f'block_list.{idx}.rgb.conv_mod.weight'] = mod_weight / np.sqrt(mod_weight.shape[1]) / mod_scale
    new_dict[f'block_list.{idx}.rgb.conv_mod.bias']   = mod_bias   / mod_scale

    # Pre-scale blur kernel, https://github.com/NVlabs/stylegan2-ada-pytorch/blob/d72cc7d041b42ec8e806021a205ed9349f87c6a4/torch_utils/ops/upfirdn2d.py#L343
    # x, y dimensions both double in size so scale by 2x2
    new_dict[f'block_list.{idx}.rgb.upsample.weight'] = up_weight * 4

    new_dict[f'block_list.{idx}.rgb.conv_weight']     = conv_weight
    new_dict[f'block_list.{idx}.rgb.bias']            = old_dict[f'synthesis.b{resolution}.torgb.bias']
    return new_dict


# Resample filters are not used in the original implementation
def get_block_0_weights(old_dict):
    new_dict    = OrderedDict()
    new_dict    = get_conv1_weights(old_dict, new_dict, idx=0)
    mod_weight  = old_dict['synthesis.b4.torgb.affine.weight']
    conv_weight = old_dict['synthesis.b4.torgb.weight']

    # Pre-normalize, https://github.com/NVlabs/stylegan2-ada-pytorch/blob/d72cc7d041b42ec8e806021a205ed9349f87c6a4/training/networks.py#L102
    new_dict['block_list.0.conv_mod.weight'] = mod_weight / np.sqrt(mod_weight.shape[1])
    new_dict['block_list.0.conv_mod.bias']   = old_dict['synthesis.b4.torgb.affine.bias']

    # Bock 0 needs normalized convolution
    # https://github.com/NVlabs/stylegan2-ada-pytorch/blob/d72cc7d041b42ec8e806021a205ed9349f87c6a4/training/networks.py#L144
    new_dict['block_list.0.conv_weight']     = conv_weight / np.sqrt(conv_weight.shape[1] * conv_weight.shape[2] * conv_weight.shape[3])

    new_dict['block_list.0.x']               = old_dict['synthesis.b4.const'].unsqueeze(0)
    new_dict['block_list.0.bias']            = old_dict['synthesis.b4.torgb.bias']
    return new_dict


def count_layers(old_dict, layer_type):
    n_layers = -1
    max_n_layers = 50

    for idx in range(max_n_layers):
        resolution = 2 ** (idx+2)

        if layer_type == 'generator':
            # First block does not have conv0, so we count conv1 instead
            layer_name = f'synthesis.b{resolution}.conv1.'
        elif layer_type == 'mapper':
            layer_name = f'mapping.fc{idx}.weight'
        else:
            raise ValueError(f'Unknown type {layer_type}')

        if not any(layer_name in key for key in old_dict.keys()):
            n_layers = idx
            break

        if idx == max_n_layers -1:
            raise ValueError(f'Number of layer is larger than {max_n_layers}')

    return n_layers


def get_generator_checkpoint(old_dict):
    n_layers = count_layers(old_dict, 'generator')
    print(f'Number of layers in generator {n_layers}')

    # Calculate feature list size, feat_list's length is len(block_list) + 1, and its
    # contents are [b0_channel_in, b0_channel_out/b1_channel_in, b1_channel_out/b2_channel_in, ...]
    feat_list = [old_dict[f'synthesis.b4.conv1.weight'].shape[1]]
    for idx in range(n_layers):
        feat = old_dict[f'synthesis.b{2 ** (idx+2)}.conv1.weight'].shape[0]
        feat_list.append(feat)

    print(f'Feature list of this model {feat_list}')
    print('Please initialize stylegan with the above feature list')

    # Load block 0 weights
    new_dict = get_block_0_weights(old_dict)

    # Load weights for the rest blocks
    for idx in range(1, n_layers):
        new_dict = get_conv0_weights(old_dict, new_dict, idx)
        new_dict = get_conv1_weights(old_dict, new_dict, idx)
        new_dict = get_rgb_weights(old_dict, new_dict, idx)

    params = {
        'genLayers': n_layers,
        'genFeatureList': feat_list,
    }

    return new_dict, params


def get_mapper_checkpoint(old_dict):
    n_layers = count_layers(old_dict, 'mapper')
    print(f'Number of layers in mapper {n_layers}')

    new_dict = OrderedDict()
    for idx in range(n_layers):
        # Normalize both weights and bias
        # https://github.com/NVlabs/stylegan2-ada-pytorch/blob/d72cc7d041b42ec8e806021a205ed9349f87c6a4/training/networks.py#L208
        fc_weight = old_dict[f'mapping.fc{idx}.weight']
        new_dict[f'layer_list.{idx}.weight'] = fc_weight*0.01/(fc_weight.shape[1] ** 0.5)
        new_dict[f'layer_list.{idx}.bias'] = old_dict[f'mapping.fc{idx}.bias']*0.01

    params = {
        'mapLayers': n_layers,
    }

    return new_dict, params


def add_to_dict(old_dict, key_prefix, subdict):
    if subdict:
        for key, value in subdict.items():
            old_dict[f'{key_prefix}.{key}'] = value

    return old_dict


def recursive_add(old_dict, key_prefix, value):
    old_dict = add_to_dict(old_dict, key_prefix, value['state']['_parameters'])
    old_dict = add_to_dict(old_dict, key_prefix, value['state']['_buffers'])
    for module_key, module_value in value['state']['_modules'].items():
        old_dict = recursive_add(old_dict, f'{key_prefix}.{module_key}', module_value)

    return old_dict


def build_old_dict(ckpt):
    old_dict = {}
    # Load mapper 
    for key, value in ckpt['G_ema']['state']['_modules']['mapping']['state']['_modules'].items():
        fc_param = value['state']['_parameters']
        old_dict[f'mapping.{key}.weight'] = fc_param['weight']
        old_dict[f'mapping.{key}.bias'] = fc_param['bias']

    for key, value in ckpt['G_ema']['state']['_modules']['synthesis']['state']['_modules'].items():
        old_dict = recursive_add(old_dict, f'synthesis.{key}', value)

    return old_dict


if __name__ == "__main__":
    model_name = "metfaces"

    checkpoints_folder = os.path.join(os.path.dirname(__file__), "checkpoints")
    out_folder = os.path.join(checkpoints_folder, model_name)

    if (not os.path.isdir(out_folder)):
        os.makedirs(out_folder)

    old_dict_path = os.path.join(checkpoints_folder, model_name + ".pkl")
    gen_dict_path = os.path.join(out_folder, model_name + "_style_generator.pt")
    map_dict_path = os.path.join(out_folder, model_name + "_style_mapper.pt")

    # Another way to use the checkpoint converter is to download dnnlib and torch_utils from github
    # https://github.com/NVlabs/stylegan2-ada-pytorch
    # Then build the state dictionary with the following code
    # with open(old_dict_path,'rb') as ckpt_file:
    #     ckpt = pickle.load(ckpt_file)
    #     old_dict = ckpt['G_ema'].state_dict()

    with open(old_dict_path,'rb') as ckpt_file:
        ckpt = pickle.load(ckpt_file)

    old_dict = build_old_dict(ckpt)

    gen_dict, gen_params = get_generator_checkpoint(old_dict)
    map_dict, map_params = get_mapper_checkpoint(old_dict)

    gen_params.update(map_params)

    params_path = os.path.join(out_folder, model_name + ".json")
    print(f"saving model parameters [{params_path}]")
    with open(params_path, 'w') as f:
        json.dump(gen_params, f)
    
    print(f"saving generator weights [{gen_dict_path}]")
    torch.save(gen_dict, gen_dict_path)
    print(f"saving mapper weights  [{map_dict_path}]")
    torch.save(map_dict, map_dict_path)

    print("done")