"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import random
from typing import Optional

import numpy as np
import torch
from datasets.utils import Rays, namedtuple_map

from nerfacc import OccGridEstimator

#from nerfacc import OccGridEstimator, sampling, rendering

from PIL import Image
import cv2
import torchvision.transforms as T

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def render_image(
    # scene
    radiance_field: torch.nn.Module,
    occupancy_grid: OccGridEstimator,
    rays: Rays,
    scene_aabb: torch.Tensor,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            return radiance_field.query_density(positions, t)
        return radiance_field.query_density(positions)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            return radiance_field(positions, t, t_dirs)
        return radiance_field(positions, t_dirs)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        ray_indices, t_starts, t_ends = ray_marching(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            scene_aabb=scene_aabb,
            grid=occupancy_grid,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        rgb, opacity, depth = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=chunk_rays.origins.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        chunk_results = [rgb, opacity, depth, len(t_starts)]
        results.append(chunk_results)
    colors, opacities, depths, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
    )

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def load_ims_to_tensorboard(writer, tag, input_images, step, resize=True):
    # input images is a list of tensors with shape (H,W,3) or (H,W,1)
    # all values are expected to be in the interval [0, 1]
    for idx, im in enumerate(input_images):
        im = im.cpu().numpy()
        if im.shape[2] == 1:
            im = normalize_im(im)
            input_images[idx] = np.dstack([im, im, im])
        else:
            input_images[idx] = im
    stack = np.clip(np.hstack(input_images), 0, 1)
    if resize:
        resize_factor = 400 / stack.shape[0]
        new_shape = (int(stack.shape[1]*resize_factor), int(stack.shape[0]*resize_factor))
        stack = Image.fromarray((stack*255.).astype(np.uint8)).resize(new_shape)
        stack = np.clip(np.array(stack)/255., 0, 1)
    writer.add_image(tag, stack, global_step=step, dataformats="HWC")

def normalize_im(x, uint8=False):
    # x shape is (H,W,1) or (H,W)
    mi, ma = np.min(x), np.max(x)
    x_ = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x_ = np.clip(x_, 0, 1)
    if uint8:
        x_ = (255*x_).astype(np.uint8)
        x_ = np.clip(x_, 0, 255)
    return x_

def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W) or (H,W,1)
    """
    depth = depth.cpu().numpy()
    x = np.nan_to_num(depth) # change nan to 0
    x = normalize_im(x, uint8=True)
    x = np.array(Image.fromarray(cv2.applyColorMap(x, cmap)))/255.
    x = torch.from_numpy(x)
    return x

