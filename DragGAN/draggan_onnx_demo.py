import torch
import onnxruntime
import onnxruntime.training.api as orttraining
import numpy as np
import json
import os
import cv2
import copy
from numpy import linalg as LA
import subprocess

import time

# import sys
# sys.path.insert(0,'/home/ran/T/cudann/lib')
# sys.path.insert(0,'/home/ran/cuda-tkit/targets/x86_64-linux/lib')
# os.environ["CUDA_HOME"] = '/home/ran/cuda-tkit'
# os.environ["CUDNN_HOME"] = '/home/ran/T/cudann'
# os.environ["CUDACXX"] = '/home/ran/cuda-tkit/bin/nvcc'
# os.environ["LD_LIBRARY_PATH"] = '.:/home/ran/T/cudann/lib:/home/ran/cuda-tkit/targets/x86_64-linux/lib:' + os.environ["LD_LIBRARY_PATH"]

# export CUDA_HOME=/home/ran/cuda-tkit
# export CUDNN_HOME=/home/ran/T/cudann
# export CUDACXX=/home/ran/cuda-tkit/bin/nvcc
# export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH


from style_mapper import StyleMapper

def create_square_mask(
    height: int, width: int, center: list, radius: int
) -> np.ndarray:
    """Create a square mask tensor.

    Args:
        height (int): The height of the mask.
        width (int): The width of the mask.
        center (list): The center of the square mask as a list of two integers. Order [y,x]
        radius (int): The radius of the square mask.

    Returns:
        np.ndarray: The square mask tensor of shape (1, 1, height, width).

    Raises:
        ValueError: If the center or radius is invalid.
    """
    if not isinstance(center, list) or len(center) != 2:
        raise ValueError("center must be a list of two integers")
    if not isinstance(radius, int) or radius <= 0:
        raise ValueError("radius must be a positive integer")
    if (
        center[0] < radius
        or center[0] >= height - radius
        or center[1] < radius
        or center[1] >= width - radius
    ):
        raise ValueError(
            "center and radius must be within the bounds of the mask")

    mask = np.zeros((height, width), dtype=np.float32)
    x1 = int(center[1]) - radius
    x2 = int(center[1]) + radius
    y1 = int(center[0]) - radius
    y2 = int(center[0]) + radius
    mask[y1: y2 + 1, x1: x2 + 1] = 1.0
    return mask.astype(bool)


def point_tracking(
    F,
    F0,
    handle_points,  # [N, y, x]
    handle_points_0,  # [N, y, x]
    r2: int = 3
):
    """
    Tracks the movement of handle points in an image using feature matching.

    Args:
        F (np.ndarray): The feature maps tensor of shape [batch_size, num_channels, height, width].
        F0 (np.ndarray): The feature maps tensor of shape [batch_size, num_channels, height, width] for the initial image.
        handle_points (np.ndarray): The handle points tensor of shape [N, y, x].
        handle_points_0 (np.ndarray): The handle points tensor of shape [N, y, x] for the initial image.
        r2 (int): The radius of the patch around each handle point to use for feature matching.

    Returns:
        The new handle points tensor of shape [N, y, x].
    """
    n = handle_points.shape[0]  # Number of handle points
    new_handle_points = np.zeros_like(handle_points)

    for i in range(n):
        # Compute the patch around the handle point
        patch = create_square_mask(
            F.shape[2], F.shape[3], center=handle_points[i].tolist(), radius=r2
        )

        # Find indices where the patch is True
        patch_coordinates = np.nonzero(patch)  # shape [num_points, 2]
        patch_coordinates = np.stack(patch_coordinates, axis=1)

        # Extract features in the patch
        F_qi = F[
            :, :, patch_coordinates[:, 0], patch_coordinates[:, 1]
        ]
        # Extract feature of the initial handle point
        f_i = F0[:, :, int(handle_points_0[i][0]), int(handle_points_0[i][1])]

        # Compute the L1 distance between the patch features and the initial handle point feature
        distances = LA.norm(F_qi - f_i[:, :, None], 1, axis=1)

        # Find the new handle point as the one with minimum distance
        min_index = np.argmin(distances)
        new_handle_points[i] = patch_coordinates[min_index]

    return new_handle_points


def add_points_to_image(image, handle_points, target_points, size=5):
    h, w, = image.shape[:2]

    for x, y in target_points:
        x, y = int(x), int(y)
        image[max(0, x - size):min(x + size, h - 1), max(0, y - size)              :min(y + size, w), :] = [255, 0, 0]
    for x, y in handle_points:
        x, y = int(x), int(y)
        image[max(0, x - size):min(x + size, h - 1), max(0, y - size)              :min(y + size, w), :] = [0, 0, 255]

    return image


def save_image(fn, img, handles=None, targets=None):
    img = np.clip(img * 127.5 + 127.5, 0, 255).astype(np.uint8)
    img = np.transpose(np.squeeze(img), axes=[1, 2, 0])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if (handles != None):
        current_handle_points = [p for p in handles]
        img = add_points_to_image(img, current_handle_points, targets)

    cv2.imwrite(fn, img)

def ffmpeg_create_video(path):
    try:
        frames_pattern = os.path.join(path, 'iter_%3d.png')
        video_filename = os.path.join(path, '_training.mp4')
        command = f'ffmpeg -y -framerate 30 -i {frames_pattern} -c:v libx264 -pix_fmt yuv420p {video_filename}'
        if subprocess.run(command).returncode == 0:
            print("FFmpeg Script Ran Successfully")
        else:
            print("There was an error running your FFmpeg script")        
    except Exception as e:
        print(e)
    finally:
        pass

def generate_latent_vector(mapper_session, seed, ws_features):
    z = np.random.RandomState(seed=seed).randn(1, ws_features)
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
    return ws

# update the model with a new latent vector for optimization
def update_model(model, model_snapshot, latent):
    model.copy_buffer_to_parameters(model_snapshot, trainable_only=True)
    model._state.parameters["latent_trainable"].data = latent[:, :6, :]
    model._state.parameters["latent_untrainable"].data = latent[:, 6:, :]

def optimize(model, optimizer, handles, targets, output_path, n_iter=100):
    tolerance = 2
    step_size = 0.002
    r2 = 12

    # create output path folder if no exist
    if (not os.path.isdir(output_path)):
        os.makedirs(output_path)

    user_constraints = np.array(
        [[1.0, handles[0][0], handles[0][1], targets[0][0], targets[0][1]]])
    handle_points = [(p[1].item(), p[2].item()) for p in user_constraints]
    target_points = [(p[3].item(), p[4].item()) for p in user_constraints]
    handle_points = np.fliplr(np.array(handle_points, dtype=np.float32))
    target_points = np.fliplr(np.array(target_points, dtype=np.float32))

    model.eval()
    loss, img, F0 = model(handle_points, target_points)

    save_intermediate_images = True
    if (save_intermediate_images):
        save_image(os.path.join(output_path, "original_image.png"), img)

    handle_points_0 = copy.deepcopy(handle_points)

    model.train()
    optimizer.set_learning_rate(step_size)
    for iter in range(n_iter):
        start = time.perf_counter()

        # Check if the handle points have reached the target points
        if np.allclose(handle_points, target_points, atol=tolerance):
            break

        model.lazy_reset_grad()

        user_constraints = np.array(
            [[1.0, handles[0][0], handles[0][1], targets[0][0], targets[0][1]]])
        handle_points = [(p[1].item(), p[2].item())
                         for p in user_constraints]
        target_points = [(p[3].item(), p[4].item())
                         for p in user_constraints]
        handle_points = np.fliplr(np.array(handle_points, dtype=np.float32))
        target_points = np.fliplr(np.array(target_points, dtype=np.float32))

        loss, img, F = model(handle_points, target_points)
        optimizer.step()

        print(
            f"{iter}\tLoss: {loss.item():0.6f}\tTime: {(time.perf_counter() - start) * 1000:.0f}ms"
        )

        if (save_intermediate_images):
            save_image(os.path.join(output_path, f"Iter_{iter:03d}.png"), img,
                       handle_points.tolist(), target_points.tolist())

        # Update the handle points with point tracking
        new_handle_points = point_tracking(
            F,
            F0,
            handle_points,
            handle_points_0,
            r2
        )

        handles = [(p[1], p[0]) for p in new_handle_points]


def main():
    # available models: afhqcat  afhqdog  afhqwild  brecahad  ffhq  metfaces
    model_name = "metfaces"

    # base_models_folder = "checkpoints/stylegan2_ada"
    # mdoel_folder = os.path.join(base_models_folder, model_name)

    current_script_folder = os.path.dirname(__file__)
    base_models_folder = os.path.join(current_script_folder, "checkpoints")
    model_folder = os.path.join(base_models_folder, model_name)
    
    onnx_output_folder = os.path.join(current_script_folder, "onnx", model_name)
    if (not os.path.isdir(onnx_output_folder)):
        os.makedirs(onnx_output_folder)

    params_path = os.path.join(model_folder, model_name + ".json")

    with open(params_path) as f:
        params = json.load(f)

    ws_feats = 512
    feat_list  = params['genFeatureList'] 
    map_layers = params['mapLayers']
    gen_layers = params['genLayers']

    # creating the onnx mapper session
    onnx_mapper_filename = os.path.join(onnx_output_folder, "stylegan_mapper.onnx")
    execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    print(f"Creating ONNX Mapper session for [{onnx_mapper_filename}]")
    so = onnxruntime.SessionOptions()
    mapper_session = onnxruntime.InferenceSession(
        onnx_mapper_filename, so, providers=execution_providers
    )


    # loading the checkpoint data
    checkpoint_filename = f"{onnx_output_folder}/checkpoint"
    print(f"Creating ONNX Mapper session for [{checkpoint_filename}]")
    checkpoint_state = orttraining.CheckpointState.load_checkpoint(checkpoint_filename)

    # creating the training module
    print(f"Creating ONNX Training Module")
    model = orttraining.Module(
        f"{onnx_output_folder}/training_model.onnx",
        checkpoint_state,
        f"{onnx_output_folder}/eval_model.onnx",
        device="cuda",
    )

    optimizer_filename = f"{onnx_output_folder}/optimizer_model.onnx"
    print(f"Creating the ONNX Optimizer from [{checkpoint_filename}]")
    optimizer = orttraining.Optimizer(optimizer_filename, model)

    with open(os.path.join(current_script_folder, "handles.json")) as user_file:
        parsed_json = json.load(user_file)
    print(parsed_json)

    model_snapshot = model.get_contiguous_parameters(trainable_only=True)

    seed = 71
    print(f"running with seed [{seed}]")
    ws = generate_latent_vector(mapper_session, seed, ws_feats)
    update_model(model, model_snapshot, ws)

    scale = 1.0
    handles = [(p[1]*scale, p[0]*scale) for p in parsed_json['points']]
    targets = [(p[3]*scale, p[2]*scale) for p in parsed_json['points']]

    output_path = os.path.join(current_script_folder, "out", "orttraining1")
    optimize(model, optimizer, handles, targets, output_path)

    ffmpeg_create_video(output_path)


    seed = 85
    print(f"running with seed [{seed}]")
    ws = generate_latent_vector(mapper_session, seed, ws_feats)
    update_model(model, model_snapshot, ws)

    scale = 1.0
    handles = [(p[1]*scale, p[0]*scale) for p in parsed_json['points']]
    targets = [(p[3]*scale, p[2]*scale) for p in parsed_json['points']]

    output_path = os.path.join(current_script_folder, "out", "orttraining2")
    optimize(model, optimizer, handles, targets, output_path)

    ffmpeg_create_video(output_path)

    print("done")



if __name__ == "__main__":
    main()
