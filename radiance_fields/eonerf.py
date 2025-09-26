"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import functools
import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from radiance_fields.mlp import SinusoidalEncoder, MLP, DenseLayer

from nerfacc.volrend import render_weight_from_density, accumulate_along_rays, render_transmittance_from_density

def cast_solar_rays(rays_o, rays_d, sun_d, far, N_samples, depth, m=0.01):
    #m = torch.clip(m, 0, 0.1)
    # (1) we will take the surface point of each ray as the origin point for solar correction rays
    sc_rays_o = rays_o + torch.vstack([depth - m, depth - m, depth - m]).T * rays_d
    # (2) compute the far and depth steps of each solar correction ray. All must be inside the [-1, 1] cube
    #sc_rays_end = torch.clip(sc_rays_o - far * sun_d, -1, 1)
    sc_rays_end = sc_rays_o - far * sun_d
    sc_scale = torch.max(torch.abs(sc_rays_o - far * sun_d))
    sc_rays_end = sc_rays_end / torch.maximum(sc_scale, torch.ones_like(sc_scale))
    #z = torch.clip(sc_rays_end[:, -1], -0.98, 0.98)
    #sc_rays_end[:, -1] = z
    #sc_rays_end = torch.clip(sc_rays_o - far * sun_d, -0.9, 0.9)
    sc_far = torch.linalg.norm(sc_rays_end - sc_rays_o, axis=1).unsqueeze(1)
    sc_near = torch.zeros(sc_far.shape).to(rays_o.device)
    sc_z_steps = torch.linspace(0, 1, N_samples, device=rays_o.device)
    sc_z_vals = sc_near * (1 - sc_z_steps) + sc_far * sc_z_steps
    # (3) discretize solar correction rays
    sc_xyz_coarse = sc_rays_o.unsqueeze(1) - sun_d.unsqueeze(1) * sc_z_vals.unsqueeze(2)
    return sc_xyz_coarse, sc_z_vals

def weights_from_sigma(z_vals, sigmas, test=False):
    # define deltas, i.e. the length between the points in which the ray is discretized
    deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples-1)
    delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # (N_rays, 1) the last delta is infinity
    deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples)

    # compute alpha as in the formula (3) of the nerf paper
    noise = torch.randn(sigmas.shape, device=sigmas.device) * 0
    alphas = 1 - torch.exp(-deltas * torch.relu(sigmas if test else sigmas + noise))

    # compute transmittance and weights
    alphas_shifted = \
        torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1)  # [1, a1, a2, ...]
    transmittance = torch.cumprod(alphas_shifted, -1)[:, :-1]  # T in the paper
    weights = alphas * transmittance  # (N_rays, N_samples)
    # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

    return weights, transmittance, alphas

def entropy_from_alphas(alphas, ray_indices):
    # alphas has shape (N_samples)
    nrays = torch.unique(ray_indices).size(0)
    ray_prob = torch.zeros((nrays,), dtype=alphas.dtype).to(alphas.device) + 1e-10
    ray_prob.index_add_(0, ray_indices, alphas)
    probs = alphas / ray_prob[ray_indices]
    #probs = p(r_i) in InfoNeRF Equation (4)
    entropy_per_sample = -1 * probs * torch.log10(probs + 1e-10)
    entropy_per_ray = torch.zeros((nrays,), dtype=alphas.dtype).to(alphas.device)
    entropy_per_ray.index_add_(0, ray_indices, entropy_per_sample)
    entropy_per_ray = entropy_per_ray.unsqueeze(-1)
    return entropy_per_ray

class EONerfMLP(nn.Module):
    def __init__(
        self,
        n_input_images: int,
        net_depth: int = 8,  # The depth of the MLP
        net_width: int = 256,  # The width of the MLP
        skip_layer: int = 4,  # The layer to add skip layers to
        radiometric_normalization: bool = False,
    ):
        super().__init__()

        self.pos_enc_L = 10
        self.view_enc_L = 4

        # (1) define positional encoders
        self.posi_encoder = SinusoidalEncoder(3, 0, self.pos_enc_L, True)
        self.view_encoder = SinusoidalEncoder(3, 0, self.view_enc_L, True)
        self.transient_encoder = nn.Embedding(n_input_images, 4) # second position is the embedding dimensionality
        self.beta_min = 0.05

        # radiometric normalization module
        self.radiometric_normalization = radiometric_normalization
        if self.radiometric_normalization:
            ones, zeros = torch.ones(n_input_images, 3), torch.zeros(n_input_images, 3)
            init_radiometricT = torch.cat([ones, zeros, zeros], dim=1)
            self.radiometricT_enc = torch.nn.Embedding.from_pretrained(init_radiometricT, freeze=False)

        # (2) define main MLP + outputs
        self.base_mlp = MLP(
            input_dim=self.posi_encoder.latent_dim,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            output_enabled=False,
        )

        hidden_features = self.base_mlp.output_dim
        self.sigma_layer = DenseLayer(hidden_features, 1, output_activation=nn.Softplus()) # equivalent of sigma_from_xyz
        self.bottleneck_layer = DenseLayer(hidden_features, net_width) #equivalent of feats_from_xyz

        # (3) define albedo color MLP
        self.albedo_mlp = MLP(
            input_dim=net_width,
            output_dim=3,
            net_depth=1,
            net_width=net_width // 2,
            skip_layer=None,
            output_activation=nn.Sigmoid(),
        )

        # (4) define transient MLP + outputs
        self.transient_mlp = MLP(
            input_dim=net_width + self.transient_encoder.embedding_dim,
            net_depth=4,
            net_width=net_width // 2,
            skip_layer=None,
            output_enabled=False,
        )
        hidden_features_transient = self.transient_mlp.output_dim
        self.transient_scalar = DenseLayer(hidden_features_transient, 1, output_activation=nn.Sigmoid())
        self.transient_beta = DenseLayer(hidden_features_transient, 1, output_activation=nn.Softplus())

        # (5) define ambient color MLP
        self.ambient_mlp = MLP(
            input_dim=self.view_encoder.latent_dim,
            output_dim=3,
            net_depth=1,
            net_width=net_width // 2,
            skip_layer=None,
            output_activation=nn.Sigmoid(),
        )

    def query_density(self, x):
        x = self.posi_encoder(x)
        x = self.base_mlp(x)
        sigma = self.sigma_layer(x)
        return sigma

    def query_opacity(self, x, step_size):
        density = self.query_density(x)
        # if the density is small enough those two are the same.
        # opacity = 1.0 - torch.exp(-density * step_size)
        opacity = density * step_size
        return opacity

    def forward(self, x, sun_dirs=None, img_indices=None):
        # multi-view consistent outputs
        x = self.posi_encoder(x)
        x = self.base_mlp(x)
        sigma = self.sigma_layer(x)
        bottleneck_feats = self.bottleneck_layer(x)
        albedo = self.albedo_mlp(bottleneck_feats)

        # view-specific outputs
        sun_dirs = self.view_encoder(sun_dirs)
        ambient = self.ambient_mlp(sun_dirs)
        transient_emb = self.transient_encoder(img_indices.squeeze())
        transient_mlp_input = torch.cat([bottleneck_feats, transient_emb], dim=-1)
        transient_mlp_feats = self.transient_mlp(transient_mlp_input)
        transient_scalar = self.transient_scalar(transient_mlp_feats)
        transient_beta = self.transient_beta(transient_mlp_feats)
        return sigma, albedo, ambient, transient_scalar, transient_beta

    def render_depth(self, chunk_rays, t_starts, t_ends, ray_indices):
        # t_starts shape: (n_pts,)
        # t_ends shape: (n_pts,)
        # ray_indices shape: (n_pts,)

        n_rays = chunk_rays.origins.shape[0]
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        z_vals = (t_starts + t_ends)[:, None] / 2.0
        positions = t_origins + t_dirs * z_vals
        _, counts = torch.unique(ray_indices, return_counts=True)
        idx_of_last_pt_in_each_ray = torch.cumsum(counts, 0) - 1
        t_ends[idx_of_last_pt_in_each_ray] = 1e10
        sigma = self.query_density(positions).squeeze()
        weights, trans, alphas = render_weight_from_density(
            t_starts,
            t_ends,
            sigma,
            ray_indices=ray_indices,
            n_rays=n_rays,
        ) # weights.shape = (n_pts,)
        depth_ = accumulate_along_rays(weights, values=z_vals, ray_indices=ray_indices, n_rays=n_rays)
        return depth_

    def rendering(self, chunk_rays, t_starts, t_ends, ray_indices, epoch_idx=100):
        # t_starts shape: (n_pts,)
        # t_ends shape: (n_pts,)
        # ray_indices shape: (n_pts,)

        n_rays = chunk_rays.origins.shape[0]
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        t_sundirs = chunk_rays.sundirs[ray_indices]
        t_imgidx = chunk_rays.img_idx[ray_indices]
        z_vals = (t_starts + t_ends)[:, None] / 2.0
        positions = t_origins + t_dirs * z_vals

        #if positions.flatten().max() > 1.:
        #    print(f"exception 1: found a point xyz outside [-1, 1], max is {positions.flatten().max()}")
        #if positions.flatten().min() < -1.:
        #    print(f"exception 2: found a point xyz outside [-1, 1], min is {positions.flatten().min()}")

        # set the ending point of each ray to infinite --> forces last position to be non-empty
        # nerfacc has some differences with respect to the old point sampling in pytorch
        # mainly the number of points per ray may be different, and the last position is not at infinite
        # instead of using a tensor of (Nrays, Npts), nerfacc uses a tensor of (Nrays*Npts) along with ray_indices
        _, counts = torch.unique(ray_indices, return_counts=True)
        idx_of_last_pt_in_each_ray = torch.cumsum(counts, 0) - 1
        t_ends[idx_of_last_pt_in_each_ray] = 1e10

        # assert n_rays == len(torch.unique(ray_indices)) # sanity check
        #print("n_pts:", ray_indices.shape[0])
        sigma, albedo_rgb, ambient_rgb, transient_scalar, transient_beta = self.forward(positions, t_sundirs, t_imgidx)
        sigma = sigma.squeeze()
        #slope = 10*((epoch_idx+1)/2)
        #sigma = torch.sigmoid(slope*(sigma - 0.5))

        weights, trans, alphas = render_weight_from_density(
            t_starts,
            t_ends,
            sigma,
            ray_indices=ray_indices,
            n_rays=n_rays,
        ) # weights.shape = (n_pts,)

        # render depth, albedo + transient outputs
        depth_ = accumulate_along_rays(weights, values=z_vals, ray_indices=ray_indices, n_rays=n_rays)
        albedo_rgb_ = accumulate_along_rays(weights, values=albedo_rgb, ray_indices=ray_indices, n_rays=n_rays)
        ambient_rgb_ = accumulate_along_rays(weights, values=ambient_rgb, ray_indices=ray_indices, n_rays=n_rays)
        transient_scalar_ = accumulate_along_rays(weights, values=transient_scalar, ray_indices=ray_indices, n_rays=n_rays)
        transient_beta_ = accumulate_along_rays(weights, values=transient_beta, ray_indices=ray_indices, n_rays=n_rays)
        transient_beta_ += self.beta_min

        #entropy_ = entropy_from_alphas(alphas, ray_indices)
        entropy_ = torch.ones_like(depth_)

        return albedo_rgb_, depth_, transient_beta_, transient_scalar_, ambient_rgb_, entropy_


    def old_rendering(self, chunk_rays, z_vals, epoch_idx=100):

        n_rays, n_samples = z_vals.shape[0], z_vals.shape[1]
        rays_o, rays_d, sun_d, rays_t = chunk_rays.origins, chunk_rays.viewdirs, chunk_rays.sundirs, chunk_rays.img_idx
        rays_xyz = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)
        ray_indices = torch.arange(n_rays).to(rays_o.device)
        ray_indices = torch.repeat_interleave(ray_indices, repeats=n_samples, dim=0)
        sun_d_ = torch.repeat_interleave(sun_d, repeats=n_samples, dim=0)
        rays_t_ = torch.repeat_interleave(rays_t, repeats=n_samples, dim=0)
        positions = rays_xyz.view(-1, 3)

        # assert n_rays == len(torch.unique(ray_indices)) # sanity check
        sigma, albedo_rgb, ambient_rgb, transient_scalar, transient_beta = self.forward(positions, sun_d_, rays_t_)
        weights, _, _ = weights_from_sigma(z_vals,  sigma.view(n_rays, n_samples))
        weights = weights.view(-1, 1)

        # render depth, albedo + transient outputs
        depth_ = accumulate_along_rays(weights, ray_indices, values=z_vals.view(-1, 1), n_rays=n_rays)
        albedo_rgb_ = accumulate_along_rays(weights, ray_indices, values=albedo_rgb, n_rays=n_rays)
        ambient_rgb_ = accumulate_along_rays(weights, ray_indices, values=ambient_rgb, n_rays=n_rays)
        transient_scalar_ = accumulate_along_rays(weights, ray_indices, values=transient_scalar, n_rays=n_rays)
        transient_beta_ = accumulate_along_rays(weights, ray_indices, values=transient_beta, n_rays=n_rays)
        transient_beta_ += self.beta_min

        # compute geometric shadows à l'ancienne
        if epoch_idx < 2:
            geo_shadows_ = torch.ones_like(transient_scalar_).to(transient_scalar_.device)
            s_ = geo_shadows_
        else:
            rays_o, rays_d, sun_dirs = chunk_rays.origins, chunk_rays.viewdirs, chunk_rays.sundirs
            sc_samples = 128
            sc_far = 3. * torch.ones_like(depth_).to(depth_.device)
            sc_xyz, sc_z_vals = cast_solar_rays(rays_o, rays_d, sun_dirs, sc_far, sc_samples, depth_.squeeze())
            sc_sigma = self.query_density(sc_xyz.view(-1, 3))
            _, sc_transmittance, _ = weights_from_sigma(sc_z_vals, sc_sigma.view(n_rays, sc_samples))
            sun_visibility = sc_transmittance.unsqueeze(-1)  # (N_rays, N_samples, 1)
            geo_shadows_ = sun_visibility[:, -1]  # (N_rays, 1)
            s_ = geo_shadows_ * transient_scalar_

        # add geometric shadows to the rendered image
        rgb_ = albedo_rgb_ * (s_ + (1 - s_) * ambient_rgb_)
        rgb_ = torch.clip(rgb_, 0, 1)

        return rgb_, depth_, transient_beta_, geo_shadows_
