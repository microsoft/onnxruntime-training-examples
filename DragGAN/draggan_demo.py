import copy
import os
import torch
import cv2
import json
import time
from typing import List, Optional, Tuple

import numpy as np

from style_mapper import StyleMapper
from style_generator import StyleGenerator

from time import perf_counter
from contextlib import contextmanager

from onnxruntime.training import artifacts
import onnx

@contextmanager
def measure_time() -> float:
    start = perf_counter()
    yield lambda: (perf_counter() - start) * 1000.0


save_intermediate_images = True

def l1_loss(x, y):
    return torch.abs(x - y).mean()


def add_points_to_image(image, handle_points, target_points, size=5):
    h, w, = image.shape[:2]

    for x, y in target_points:
        image[max(0, x - size):min(x + size, h - 1), max(0, y - size):min(y + size, w), :] = [255, 0, 0]
    for x, y in handle_points:
        image[max(0, x - size):min(x + size, h - 1), max(0, y - size):min(y + size, w), :] = [0, 0, 255]

    return image


def save_image(fn, img, handles=None, targets=None):
    img = img.detach().cpu().numpy()
    img = np.clip(img * 127.5 + 127.5, 0, 255).astype(np.uint8)
    img = np.transpose(np.squeeze(img), axes=[1, 2, 0])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if (handles != None):
        current_handle_points = [p for p in handles]
        img = add_points_to_image(img, current_handle_points, targets)

    cv2.imwrite(fn, img)


def create_circular_mask(
    h: int,
    w: int,
    center: torch.Tensor = None,
    radius: Optional[int] = None,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """
    Create a circular mask tensor.

    Args:
        h (int): The height of the mask tensor.
        w (int): The width of the mask tensor.
        center (Optional[Tuple[int, int]]): The center of the circle as a tuple (y, x). If None, the middle of the image is used.
        radius (Optional[int]): The radius of the circle. If None, the smallest distance between the center and image walls is used.

    Returns:
        A boolean tensor of shape [h, w] representing the circular mask.
    """
    if center is None:  # use the middle of the image
        center = (int(h / 2), int(w / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], h - center[0], w - center[1])

    Y = torch.arange(h).float().to(device).unsqueeze(1)
    X = torch.arange(w).float().to(device).unsqueeze(0)
    dist_from_center = torch.sqrt((Y - center[0]) ** 2 + (X - center[1]) ** 2)

    mask = dist_from_center <= radius
    mask = mask.bool()
    return mask


def create_square_mask(
    height: int, width: int, center: list, radius: int
) -> torch.Tensor:
    """Create a square mask tensor.

    Args:
        height (int): The height of the mask.
        width (int): The width of the mask.
        center (list): The center of the square mask as a list of two integers. Order [y,x]
        radius (int): The radius of the square mask.

    Returns:
        torch.Tensor: The square mask tensor of shape (1, 1, height, width).

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

    mask = torch.zeros((height, width), dtype=torch.float32)
    x1 = int(center[1]) - radius
    x2 = int(center[1]) + radius
    y1 = int(center[0]) - radius
    y2 = int(center[0]) + radius
    mask[y1: y2 + 1, x1: x2 + 1] = 1.0
    return mask.bool()


def motion_supervision(
    F: torch.Tensor,
    F0: torch.Tensor,
    handle_points: torch.Tensor,
    target_points: torch.Tensor,
    r1: int = 3,
    lambda_: float = 20.0,
    device: torch.device = torch.device("cuda"),
    multiplier: float = 1.0,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Computes the motion supervision loss and the shifted coordinates for each handle point.

    Args:
        F (torch.Tensor): The feature map tensor of shape [batch_size, num_channels, height, width].
        F0 (torch.Tensor): The original feature map tensor of shape [batch_size, num_channels, height, width].
        handle_points (torch.Tensor): The handle points tensor of shape [num_handle_points, 2].
        target_points (torch.Tensor): The target points tensor of shape [num_handle_points, 2].
        r1 (int): The radius of the circular mask around each handle point.
        lambda_ (float): The weight of the reconstruction loss for the unmasked region.
        device (torch.device): The device to use for the computation.
        multiplier (float): The multiplier to use for the direction vector.

    Returns:
        A tuple containing the motion supervision loss tensor and a list of shifted coordinates
        for each handle point, where each element in the list is a tensor of shape [num_points, 2].
    """
    n = handle_points.shape[0]  # Number of handle points
    loss = 0.0
    all_shifted_coordinates = []  # List of shifted patches

    from grid_sample import bilinear_grid_sample

    for i in range(n):
        # Compute direction vector
        target2handle = target_points[i] - handle_points[i]
        d_i = target2handle / (torch.norm(target2handle) + 1e-7) * multiplier
        d_i_norm = torch.norm(d_i)
        target2handle_norm = torch.norm(target2handle)
        d_i = torch.where(d_i_norm > target2handle_norm, target2handle, d_i)

        # Compute the mask for the pixels within radius r1 of the handle point
        mask = create_circular_mask(
            F.shape[2], F.shape[3], center=handle_points[i], radius=r1
        ).to(device)

        # Find indices where mask is True
        coordinates = torch.nonzero(mask).float()  # shape [num_points, 2]

        # Shift the coordinates in the direction d_i
        shifted_coordinates = coordinates + d_i[None]
        all_shifted_coordinates.append(shifted_coordinates)

        h, w = F.shape[2], F.shape[3]

        # Extract features in the mask region and compute the loss
        F_qi = torch.squeeze(F).reshape((-1, h*w))  # shape: [C, H*W]
        F_qi = F_qi[:, mask.reshape((-1,))]  # shape: [C, H*W]

        # Sample shifted patch from F
        normalized_shifted_coordinates = shifted_coordinates.clone()
        normalized_shifted_coordinates[:, 0] = (
            2.0 * shifted_coordinates[:, 0] / (h - 1)
        ) - 1  # for height
        normalized_shifted_coordinates[:, 1] = (
            2.0 * shifted_coordinates[:, 1] / (w - 1)
        ) - 1  # for width
        # Add extra dimensions for batch and channels (required by grid_sample)
        normalized_shifted_coordinates = normalized_shifted_coordinates.unsqueeze(
            0
        ).unsqueeze(
            0
        )  # shape [1, 1, num_points, 2]
        normalized_shifted_coordinates = normalized_shifted_coordinates.flip(
            -1
        )  # grid_sample expects [x, y] instead of [y, x]
        normalized_shifted_coordinates = normalized_shifted_coordinates.clamp(
            -1, 1)

        # Use grid_sample to interpolate the feature map F at the shifted patch coordinates
        # print("Coordinates:", normalized_shifted_coordinates.shape)
        F_qi_plus_di = bilinear_grid_sample(
            F, normalized_shifted_coordinates, align_corners=True
        )
        # Output has shape [1, C, 1, num_points] so squeeze it
        F_qi_plus_di = F_qi_plus_di.squeeze(2)  # shape [1, C, num_points]

        loss += l1_loss(F_qi.detach(), F_qi_plus_di.squeeze())

    return loss, all_shifted_coordinates


def point_tracking(
    F: torch.Tensor,
    F0: torch.Tensor,
    handle_points: torch.Tensor,  # [N, y, x]
    handle_points_0: torch.Tensor,  # [N, y, x]
    r2: int = 3,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """
    Tracks the movement of handle points in an image using feature matching.

    Args:
        F (torch.Tensor): The feature maps tensor of shape [batch_size, num_channels, height, width].
        F0 (torch.Tensor): The feature maps tensor of shape [batch_size, num_channels, height, width] for the initial image.
        handle_points (torch.Tensor): The handle points tensor of shape [N, y, x].
        handle_points_0 (torch.Tensor): The handle points tensor of shape [N, y, x] for the initial image.
        r2 (int): The radius of the patch around each handle point to use for feature matching.
        device (torch.device): The device to use for the computation.

    Returns:
        The new handle points tensor of shape [N, y, x].
    """
    with torch.no_grad():
        n = handle_points.shape[0]  # Number of handle points
        new_handle_points = torch.zeros_like(handle_points)

        for i in range(n):
            # Compute the patch around the handle point
            patch = create_square_mask(
                F.shape[2], F.shape[3], center=handle_points[i].tolist(), radius=r2
            ).to(device)

            # Find indices where the patch is True
            patch_coordinates = torch.nonzero(patch)  # shape [num_points, 2]

            # Extract features in the patch
            F_qi = F[
                :, :, patch_coordinates[:, 0], patch_coordinates[:, 1]
            ]
            # Extract feature of the initial handle point
            f_i = F0[
                :, :, handle_points_0[i][0].long(), handle_points_0[i][1].long()
            ]

            # Compute the L1 distance between the patch features and the initial handle point feature
            distances = torch.norm(F_qi - f_i[:, :, None], p=1, dim=1)

            # Find the new handle point as the one with minimum distance
            min_index = torch.argmin(distances)
            new_handle_points[i] = patch_coordinates[min_index]

    return new_handle_points


class DragGANLearningForONNX(torch.nn.Module):
    def __init__(
        self,
        features_count,
        features_list,
        r1: int = 3,
        r2: int = 12,
        tolerance: int = 2,
        multiplier: float = 1.0,
        lambda_: float = 0.1,
        device: torch.device = torch.device("cuda"),
        output_path=".",
    ):
        super().__init__()

        self.device = device
        self.features_count = features_count
        self.features_list = features_list
        self.r1 = r1
        self.r2 = r2
        self.tolerance = tolerance
        self.multiplier = multiplier
        self.lambda_ = lambda_
        self.output_path = output_path

        self.generator = StyleGenerator(
            self.features_list, self.features_count).cuda().eval()

        z_dim = 512
        num_ws = 18
        # latent[:, :6, :].detach().clone().requires_grad_(True)
        self.latent_trainable = torch.nn.Parameter(torch.zeros(1, 6, z_dim))
        # latent[:, 6:, :].detach().clone().requires_grad_(False)
        self.latent_untrainable = torch.nn.Parameter(
            torch.zeros(1, num_ws - 6, z_dim))

    def load_weights(self, synth_weights, latent):
        self.generator.load_state_dict(synth_weights)

        if not isinstance(latent, torch.Tensor):
            latent = torch.from_numpy(latent).to(self.device).float()

        self.latent_trainable.data = latent[:, :6, :]
        self.latent_untrainable.data = latent[:, 6:, :]

    def pre_optimization(self, latent):

        if not isinstance(latent, torch.Tensor):
            latent = torch.from_numpy(latent).to(self.device).float()

        img, self.F0 = self.generator(latent)

        # if (save_intermediate_images):
        #     save_image(os.path.join(self.output_path, "original_image.png"), img)

        self.F0_resized = self.F0

    def forward(self, handle_points, target_points, F0_resized):
        # Detach only the unoptimized layers
        W_combined = torch.cat(
            [self.latent_trainable, self.latent_untrainable], dim=1)

        # Run the generator to get the image and feature maps
        # img, F = forward_G(G, W_combined, device)
        img, F = self.generator(W_combined)

        # Compute the motion supervision loss
        loss, _ = motion_supervision(
            F,
            F0_resized,
            handle_points,
            target_points,
            self.r1,
            self.lambda_,
            self.device,
            multiplier=self.multiplier,
        )

        return loss, img, F


def generate_artifacts(draggan_onnx_fn, output_folder):

    artifacts_directory = output_folder
    if (not os.path.isdir(artifacts_directory)):
        os.makedirs(artifacts_directory)

    model = onnx.load(draggan_onnx_fn)
    requires_grad = ["latent_trainable"]
    frozen_params = [
        init.name for init in model.graph.initializer if init.name not in requires_grad]

    artifacts.generate_artifacts(
        model, requires_grad=requires_grad, frozen_params=frozen_params, optimizer=artifacts.OptimType.AdamW, artifact_directory=artifacts_directory)


    # The original training script contains a tensor.detach call which essentially requires
    # the gradient to be detached from the computation graph.
    # This is not supported by ONNX Runtime Training offline utility out of the box.
    # So we manually remove the backward subgraph ending at the node which contains the detach call.
    training_model = onnx.load(os.path.join(
        artifacts_directory, "training_model.onnx"))
    for node in training_model.graph.node:
        if node.name == "/generator/Resize_Grad/ResizeGrad_0":
            node.input[0] = "F_grad_0"

    onnx.save(training_model, os.path.join(
        artifacts_directory, "training_model.onnx"))


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
    feat_list  = params['genFeatureList'] 
    map_layers = params['mapLayers']
    gen_layers = params['genLayers']

    map = StyleMapper(num_layers=map_layers, ws_feats=ws_feats).cuda().eval()
    map.load_state_dict(torch.load(map_dict_path, map_location='cuda'))
    gen = StyleGenerator(feat_list, ws_feats).cuda().eval()
    gen.load_state_dict(torch.load(gen_dict_path))

    seed = 71
    z = np.random.RandomState(seed=seed).randn(1, ws_feats)
    z = z / np.sqrt(np.mean(z**2, axis=1, keepdims=True) + 1e-8)

    ws = map(torch.from_numpy(z.astype(np.float32)).cuda())
    ws = ws.repeat(1,18,1)

    orig_img, orig_F = gen(ws)

    with open(os.path.join(current_script_folder, "handles.json")) as user_file:
        parsed_json = json.load(user_file)
    print(parsed_json)

    #
    #   Export the original stylegan mapper and generator
    #
    with torch.no_grad():
        torch.onnx.export(
            map, 
            (torch.tensor(z).cuda().to(torch.float32)), 
            os.path.join(onnx_output_folder, "stylegan_mapper.onnx"), 
            opset_version=16,
            input_names=['input'],
            output_names=['output']
        )

        torch.onnx.export(
            gen, 
            (ws), 
            os.path.join(onnx_output_folder, "stylegan_generator.onnx"), 
            opset_version=16,
            input_names=['input'],
            output_names=['img', 'F']
        )

    scale = 1.0
    handles = [(p[1]*scale, p[0]*scale) for p in parsed_json['points']]
    targets = [(p[3]*scale, p[2]*scale) for p in parsed_json['points']]

    ws = ws.cpu().detach().numpy()

    tolerance = 2
    n_iter = 100
    step_size = 0.002  # 0.001
    multiplier = 1.0
    display_every = 1
    device = torch.device("cuda")

    output_path = os.path.join(current_script_folder, "out", "training")
    if (not os.path.isdir(output_path)):
        os.makedirs(output_path)

    py_to_onnx_model = DragGANLearningForONNX(
        features_count=ws_feats,
        features_list=feat_list,
        output_path=output_path,
        tolerance=tolerance,
        multiplier=multiplier).to(device)
    py_to_onnx_model.load_weights(torch.load(gen_dict_path), ws)

    model = py_to_onnx_model

    user_constraints = torch.tensor(
        [[1.0, handles[0][0], handles[0][1], targets[0][0], targets[0][1]]]).to(device)
    handle_points = [(p[1].item(), p[2].item())
                        for p in user_constraints]
    target_points = [(p[3].item(), p[4].item())
                        for p in user_constraints]
    handle_points: torch.tensor = (
        torch.tensor(handle_points,
                        device=device).flip(-1).float()
    )
    # self.handle_points_0 = self.handle_points.clone()
    target_points: torch.tensor = (
        torch.tensor(target_points,
                        device=device).flip(-1).float()
    )

    model.pre_optimization(ws)

    with torch.no_grad():
        export_directory = "onnx"
        draggan_onnx_filename = os.path.join(onnx_output_folder, "draggan.onnx")
        torch.onnx.export(
            model,  # pytorch model to export
            (copy.deepcopy(handle_points.to(device)),
                copy.deepcopy(target_points.to(device)),
                model.F0.to(device)),  # sample inputs to the model
            # exported onnx model
            draggan_onnx_filename,
            opset_version=16,  # opset version
            training=torch.onnx.TrainingMode.TRAINING,  # training mode
            # avoid doing constant folding ,otherwise the trainable weights might be optimized away
            do_constant_folding=False,
            # model input names
            input_names=['handle_points', 'target_points', 'F0'],
            output_names=['loss', 'img', 'F'])  # model output names
        

        #
        # generate training artifacts
        # 
        generate_artifacts(draggan_onnx_filename, onnx_output_folder)

        #
        # save the info file
        # 
        info_filename = os.path.join(onnx_output_folder, "info.json")
        info = {
            'ModelName': model_name,
            'ImageSize': orig_img.shape[2:],
            'FFeatureSize': orig_F.shape[1],
        }
        with open(info_filename, "w") as fp:
            json.dump(info , fp)            


    # run the training model once just to see that it works
    loss, img, F0 = model(handle_points, target_points, model.F0_resized)
    if (save_intermediate_images):
        save_image(os.path.join(output_path, "original_image.png"), img)

    handle_points_0 = handle_points.clone()

    # Freeze the first layer.
    requires_grad = ["latent_trainable"]
    for name, param in model.named_parameters():
        if name not in requires_grad:
            param.requires_grad = False
        else:
            param.requires_grad = True

    #
    # optimization loop
    #
    optimizer = torch.optim.Adam(model.parameters(), lr=step_size)
    for iter in range(n_iter):
        start = time.perf_counter()

        # Check if the handle points have reached the target points
        if torch.allclose(handle_points, target_points, atol=model.tolerance):
            break

        optimizer.zero_grad()

        user_constraints = torch.tensor(
            [[1.0, handles[0][0], handles[0][1], targets[0][0], targets[0][1]]]).to(device)

        handle_points = [(p[1].item(), p[2].item())
                            for p in user_constraints]
        target_points = [(p[3].item(), p[4].item())
                            for p in user_constraints]
        handle_points: torch.tensor = (
            torch.tensor(handle_points,
                            device=device).flip(-1).float()
        )
        # self.handle_points_0 = self.handle_points.clone()
        target_points: torch.tensor = (
            torch.tensor(target_points,
                            device=device).flip(-1).float()
        )

        # Run the generator to get the image and feature maps
        loss, img, F = model(
            handle_points, target_points, model.F0_resized)

        loss.backward()
        optimizer.step()

        print(
            f"{iter}\tLoss: {loss.item():0.6f}\tTime: {(time.perf_counter() - start) * 1000:.0f}ms"
        )

        if (save_intermediate_images):
            save_image(os.path.join(output_path, f"Iter_{iter:03d}.png"), img, handle_points.cpu(
            ).long().numpy().tolist(), target_points.cpu().long().numpy().tolist())

        # Update the handle points with point tracking
        new_handle_points = point_tracking(
            F,
            F0,
            handle_points,
            handle_points_0,
            model.r2,
            model.device,
        )

        handles = [(p[1], p[0])
                    for p in new_handle_points.cpu().detach().numpy()]



if __name__ == "__main__":
    main()
