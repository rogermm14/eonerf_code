import torch
from nerfacc import OccGridEstimator, rendering, render_transmittance_from_density, accumulate_along_rays

from typing import Optional

from datasets.satellite import SatRays
from datasets.utils import namedtuple_map
from sys import getsizeof

def count_number_of_pts_per_nerfacc_ray(rays, ray_indices):
    # nerfacc with occupancy_grid can return rays without points
    # in satellite scenes it is impossible that a ray contains no points
    unique_indices, counts = torch.unique(ray_indices, return_counts=True)
    pts_per_ray = torch.zeros_like(torch.zeros_like(rays.origins[:, 0])).to(rays.origins.device)
    pts_per_ray[unique_indices] = counts.type(pts_per_ray.dtype)
    return pts_per_ray

def filter_pts_outside_cube(xyz):
    # xyz has shape (Npts, 3)
    mask = torch.sum(torch.abs(xyz) >= 1, dim=1) == 0
    all_pts_ok = torch.sum(mask) == xyz.shape[0]
    return xyz[mask], mask, all_pts_ok

def nerfacc_sampling(origins, viewdirs, occupancy_grid, sampling_args):
    # inputs:
    #     origins.shape = (Nrays, 3)
    #     viewdiers.shape = (Nrays, 3)
    # outputs:
    #    ray_indices.shape = (Npts) --> associates each point/interval with a ray index
    #    t_starts.shape = (Npts) --> starting distance t of each point interval (from the formula r(t) = o + t * d)
    #    t_ends.shape = (Npts) --> end distance t of each point interval (from the formula r(t) = o + t * d)
    ray_indices, t_starts, t_ends = occupancy_grid.sampling(
        origins,
        viewdirs,
        near_plane=0.0,
        far_plane=4.0,
        sigma_fn=sampling_args["sigma_fn"],
        render_step_size=sampling_args["render_step_size"],
        stratified=sampling_args["stratified"],
        cone_angle=sampling_args["cone_angle"],
        alpha_thre=sampling_args["alpha_thre"],
        early_stop_eps=sampling_args["early_stop_eps"],
    )
    return ray_indices, t_starts, t_ends

def perturb_z_vals(z_vals, perturb):
    z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
    # get intervals between samples
    upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
    lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

    perturb_rand = perturb * torch.rand_like(z_vals)
    z_vals = lower + (upper - lower) * perturb_rand
    return z_vals

def satnerf_sampling(origins, viewdirs, sampling_args, near=None, far=None, perturb=True):
    # info: this is the exact same sampler as satnerf, but ready to be plugged in the nerfacc logic

    # set near and far boundaries. every ray will be discretized into n_samples
    if near is None:
        near = torch.zeros_like(origins[: , 0:1], device=origins.device)
    if far is None:
        far = near + 2
    n_samples =  int(2/sampling_args["render_step_size"]) # length of the [-1, 1] cube, i.e. 2, divided the render step

    # use linear sampling in depth space
    z_steps = torch.linspace(0, 1, n_samples, device=origins.device)
    z_vals = near * (1 - z_steps) + far * z_steps
    if perturb:  # perturb the depth values (z_vals) so that they are not always fixed at the same uniform steps
        z_vals = perturb_z_vals(z_vals, perturb)

    # rewrite everything in the nerfacc fashion
    n_rays = origins.shape[0]
    t_ends = (z_vals[:, :-1] + (z_vals[:, 1:] - z_vals[:, :-1])).flatten()
    t_starts = z_vals[:, :-1].flatten()
    ray_indices = torch.arange(start=0, end=n_rays, device=origins.device).repeat_interleave(n_samples-1)

    # remove points outside [-1, 1] cube
    z_vals = (t_starts + t_ends)[:, None] / 2.0
    xyz = origins[ray_indices] + viewdirs[ray_indices] * z_vals
    _, mask, _ = filter_pts_outside_cube(xyz)
    ray_indices, t_starts, t_ends = ray_indices[mask], t_starts[mask], t_ends[mask]

    return ray_indices, t_starts, t_ends


def compute_geometric_shadows(chunk_rays, depth, radiance_field, occupancy_grid, sampling_args):
    n_rays = chunk_rays.origins.shape[0]
    m = 0
    sc_origins = chunk_rays.origins + torch.hstack([depth - m, depth - m, depth - m]) * chunk_rays.viewdirs
    sc_viewdirs = -1.0 * chunk_rays.sundirs
    #sc_ray_indices, sc_t_starts, sc_t_ends = nerfacc_sampling(sc_origins, sc_viewdirs, occupancy_grid, sampling_args)
    sc_ray_indices, sc_t_starts, sc_t_ends = satnerf_sampling(sc_origins, sc_viewdirs, sampling_args)
    # sc_ray_indices, sc_t_starts, sc_t_ends = nerfacc_sampling_for_sc(sc_origins, sc_viewdirs, occupancy_grid, sampling_args)

    sc_pts_per_ray = count_number_of_pts_per_nerfacc_ray(chunk_rays, sc_ray_indices)

    sc_z_vals = (sc_t_starts + sc_t_ends)[:, None] / 2.0
    sc_positions = sc_origins[sc_ray_indices] + sc_viewdirs[sc_ray_indices] * sc_z_vals
    sc_sigma = radiance_field.query_density(sc_positions)
    sc_sigma = sc_sigma.squeeze()

    # slope = 10 * ((epoch_idx + 1) / 2)
    # sc_sigma = torch.sigmoid(slope * (sc_sigma - 0.5))

    sc_transmittance, _ = render_transmittance_from_density(sc_t_starts,
                                                            sc_t_ends,
                                                            sc_sigma,
                                                            ray_indices=sc_ray_indices,
                                                            n_rays=n_rays)
    sc_transmittance = sc_transmittance.view(-1, 1)
    _, counts = torch.unique(sc_ray_indices, return_counts=True)
    # sc_ray_indices = sc_ray_indices.type(torch.int)
    idx_of_last_pt_in_each_ray = torch.cumsum(counts, 0) - 1
    geo_shadow = torch.ones((n_rays, 1)).to(chunk_rays.origins.device)
    geo_shadow[torch.unique(sc_ray_indices).squeeze()] = sc_transmittance[idx_of_last_pt_in_each_ray]

    return geo_shadow, sc_pts_per_ray

def compute_nadir_rays(chunk_rays, depth, radiance_field, occupancy_grid, sampling_args):
    n_rays = chunk_rays.origins.shape[0]
    nadir_origins = chunk_rays.origins + torch.hstack([depth, depth, depth]) * chunk_rays.viewdirs
    #nadir_origins[:, -1] = 1.0
    nadir_viewdirs = torch.zeros_like(nadir_origins).to(chunk_rays.origins.device)
    nadir_viewdirs[:, -1] = -1.0
    ray_indices, t_starts, t_ends = satnerf_sampling(nadir_origins, nadir_viewdirs, sampling_args)

    pts_per_ray = count_number_of_pts_per_nerfacc_ray(chunk_rays, ray_indices)

    z_vals = (t_starts + t_ends)[:, None] / 2.0
    nadir_positions = nadir_origins[ray_indices] + nadir_viewdirs[ray_indices] * z_vals
    sigma = radiance_field.query_density(nadir_positions)
    sigma = sigma.squeeze()
    trans, alphas = render_transmittance_from_density(t_starts,
                                                         t_ends,
                                                         sigma,
                                                         ray_indices=ray_indices,
                                                         n_rays=n_rays)
    weights = torch.ones_like(alphas).to(alphas.device)/pts_per_ray[ray_indices]
    opacity_after_surface = accumulate_along_rays(weights, values=alphas.reshape(-1, 1), ray_indices=ray_indices, n_rays=n_rays)

    opacity_after_surface = torch.cat([opacity_after_surface, opacity_after_surface], 1)

    return opacity_after_surface

def compute_nadir_rays_v2(chunk_rays, depth, radiance_field, occupancy_grid, sampling_args):
    n_rays = chunk_rays.origins.shape[0]
    nadir_origins = chunk_rays.origins + torch.hstack([depth, depth, depth]) * chunk_rays.viewdirs
    #nadir_origins[:, -1] = 1.0
    nadir_viewdirs = torch.zeros_like(nadir_origins).to(chunk_rays.origins.device)

    opacity_after_surface = []
    for dir in [-1.0, 1.0]:
        # -1 dir is after surface, + 1 dir is before surface
        nadir_viewdirs[:, -1] = dir
        ray_indices, t_starts, t_ends = satnerf_sampling(nadir_origins, nadir_viewdirs, sampling_args)

        pts_per_ray = count_number_of_pts_per_nerfacc_ray(chunk_rays, ray_indices)

        z_vals = (t_starts + t_ends)[:, None] / 2.0
        nadir_positions = nadir_origins[ray_indices] + nadir_viewdirs[ray_indices] * z_vals
        sigma = radiance_field.query_density(nadir_positions)
        sigma = sigma.squeeze()
        trans, alphas = render_transmittance_from_density(t_starts,
                                                             t_ends,
                                                             sigma,
                                                             ray_indices=ray_indices,
                                                             n_rays=n_rays)
        weights = torch.ones_like(alphas).to(alphas.device)/pts_per_ray[ray_indices]
        opacity_after_surface.append(accumulate_along_rays(weights, values=alphas.reshape(-1, 1), ray_indices=ray_indices, n_rays=n_rays))

    opacity_after_surface = torch.cat([opacity_after_surface[0], opacity_after_surface[1]], 1)

    return opacity_after_surface

def render_image(
    # scene
    radiance_field: torch.nn.Module,
    occupancy_grid: OccGridEstimator,
    rays: SatRays,
    scene_aabb: torch.Tensor,
    args,
    epoch_idx: Optional[int] = None,
    chunk: int = 5120,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    early_stop_eps: float = 0.0,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
    # in case you only wanna query depth
    only_depth: bool = False,
    eval: bool = False,
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
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        sigmas = radiance_field.query_density(positions)
        return sigmas.squeeze()

    sampling_args = {
        "sigma_fn": sigma_fn,
        "render_step_size": render_step_size,
        "stratified": radiance_field.training,
        "cone_angle": cone_angle,
        "alpha_thre": alpha_thre,
        "early_stop_eps": early_stop_eps,
    }

    if only_depth:
        results = []
        for i in range(0, num_rays, chunk):
            chunk_rays = namedtuple_map(lambda r: r[i: i + chunk], rays)
            near, far = chunk_rays.t_near, chunk_rays.t_far

            # compute outputs related to camera rays
            #ray_indices, t_starts, t_ends = nerfacc_sampling(chunk_rays.origins, chunk_rays.viewdirs, occupancy_grid, sampling_args)
            ray_indices, t_starts, t_ends = satnerf_sampling(chunk_rays.origins, chunk_rays.viewdirs, sampling_args, near=near)
            pts_per_ray = count_number_of_pts_per_nerfacc_ray(chunk_rays, ray_indices)
            if torch.sum(pts_per_ray == 0):
                #print("warning: certain rays without points were detected !")
                ray_indices, t_starts, t_ends = satnerf_sampling(chunk_rays.origins, chunk_rays.viewdirs, sampling_args)

            depth = radiance_field.render_depth(chunk_rays, t_starts, t_ends, ray_indices)
            chunk_rendering_samples = len(t_starts)
            results.append([depth, chunk_rendering_samples])
        out, n_rendering_samples = [
            torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
            for r in zip(*results)
        ]
        n_rendering_samples = sum(n_rendering_samples)
        results = {"depth": out.view((*rays_shape[:-1], -1))}
    else:
        results = []
        for i in range(0, num_rays, chunk):
            chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
            near, far = chunk_rays.t_near, chunk_rays.t_far

            # compute outputs related to camera rays
            #ray_indices, t_starts, t_ends = nerfacc_sampling(chunk_rays.origins, chunk_rays.viewdirs, occupancy_grid, sampling_args)
            ray_indices, t_starts, t_ends = satnerf_sampling(chunk_rays.origins, chunk_rays.viewdirs, sampling_args, near=near)
            pts_per_ray = count_number_of_pts_per_nerfacc_ray(chunk_rays, ray_indices)
            if torch.sum(pts_per_ray == 0):
                #print("warning: certain rays without points were detected !")
                ray_indices, t_starts, t_ends = satnerf_sampling(chunk_rays.origins, chunk_rays.viewdirs, sampling_args)

            albedo_rgb, depth, beta, transient_s, ambient_rgb, entropy = radiance_field.rendering(chunk_rays, t_starts, t_ends, ray_indices, epoch_idx)
            ambient_rgb *= 0.2 # prevents amient_rgb = [1, 1, 1] so that shadows cannot be ignored

            # compute outputs related to solar rays
            n_rays = chunk_rays.origins.shape[0]
            if epoch_idx < 2:
                geo_shadow = torch.ones((n_rays, 1)).to(ray_indices.device)
                s = geo_shadow
                sc_pts_per_ray = torch.ones_like(pts_per_ray).to(ray_indices.device)
            else:
                geo_shadow, sc_pts_per_ray = compute_geometric_shadows(chunk_rays, depth, radiance_field, occupancy_grid, sampling_args)
                # add geometric shadows to the rendered image
                s = geo_shadow * transient_s


            use_opacity_after_surface = False
            if use_opacity_after_surface:
                opacity_after_surface = compute_nadir_rays_v2(chunk_rays, depth, radiance_field, occupancy_grid, sampling_args)
            else:
                opacity_after_surface = torch.ones(transient_s.shape[0], 2).to(transient_s.device)

            # compute rgb using the s-nerf irradiance model
            #rgb = albedo_rgb * (s + (1 - s) * ambient_rgb)

            if eval:
                img_indices_notpruned = torch.ones_like(albedo_rgb[:, 0]).long() * chunk_rays.img_idx[0]
            else:
                img_indices_notpruned = chunk_rays.img_idx.squeeze()
            ambient_bias = torch.abs(radiance_field.radiometricT_enc(img_indices_notpruned)[:, 6:])

            rgb = albedo_rgb * s + (1 - s) * (ambient_rgb *albedo_rgb) # + ambient_bias)

            # optional radiometric normalization
            if radiance_field.radiometric_normalization:
                A = radiance_field.radiometricT_enc(img_indices_notpruned)[:, :3]
                b = radiance_field.radiometricT_enc(img_indices_notpruned)[:, 3:6]
            else:
                A = torch.ones_like(rgb).to(rgb.device)
                b = torch.zeros_like(rgb).to(rgb.device)

            rgb = A * rgb + b
            rgb = torch.clip(rgb, 0, 1)
            shadowless_rgb = A * albedo_rgb + b
            #rgb = torch.clip(rgb ** (1/2.2), 0, 1) # gamma correction

            pts_per_ray = pts_per_ray.unsqueeze(-1)
            sc_pts_per_ray = sc_pts_per_ray.unsqueeze(-1)
            chunk_results = torch.cat([rgb, depth, albedo_rgb, ambient_rgb, geo_shadow, transient_s, beta,
                                       entropy, pts_per_ray, sc_pts_per_ray, opacity_after_surface, shadowless_rgb], dim=1)
            chunk_rendering_samples = len(t_starts)
            results.append([chunk_results, chunk_rendering_samples])

        out, n_rendering_samples = [
            torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
            for r in zip(*results)
        ]
        n_rendering_samples = sum(n_rendering_samples)

        results = {"rgb": out[:, :3].view((*rays_shape[:-1], -1)),
                   "depth": out[:, 3:4].view((*rays_shape[:-1], -1)),
                   "albedo_rgb": out[:, 4:7].view((*rays_shape[:-1], -1)),
                   "ambient_rgb": out[:, 7:10].view((*rays_shape[:-1], -1)),
                   "geo_shadows": out[:, 10:11].view((*rays_shape[:-1], -1)),
                   "transient_s": out[:, 11:12].view((*rays_shape[:-1], -1)),
                   "beta": out[:, 12:13].view((*rays_shape[:-1], -1)),
                   "entropy": out[:, 13:14].view((*rays_shape[:-1], -1)),
                   "pts_per_ray": out[:, 14:15].view((*rays_shape[:-1], -1)),
                   "sc_pts_per_ray": out[:, 15:16].view((*rays_shape[:-1], -1)),
                   "opacity_after_surface": out[:, 16:18].view((*rays_shape[:-1], -1)),
                   "shadowless_rgb": out[:, 18:21].view((*rays_shape[:-1], -1))
                   }
    return results, n_rendering_samples


def render_image_old( # scene
    radiance_field: torch.nn.Module,
    occupancy_grid: OccGridEstimator,
    rays: SatRays,
    scene_aabb: torch.Tensor,
    epoch_idx: Optional[int] = None,
    chunk: int = 5120,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
):

    rays_o, rays_d, sun_d, rays_t = rays.origins, rays.viewdirs, rays.sundirs, rays.img_idx
    near = torch.zeros_like(rays_o[: , 0:1])
    far = near + 2
    N_samples = 128

    # sample depths for coarse model
    z_steps = torch.linspace(0, 1, N_samples, device=rays_o.device)
    # use linear sampling in depth space
    z_vals = near * (1 - z_steps) + far * z_steps

    perturb = 1.0
    if perturb > 0:  # perturb sampling depths (z_vals)
        z_vals = perturb_z_vals(z_vals, perturb)

    num_rays = rays_o.shape[0]

    # run model
    results = []
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i: i + chunk], rays)
        rgb, depth, beta, geo_shadow = radiance_field.old_rendering(chunk_rays, z_vals[i:i+chunk], epoch_idx)
        chunk_results = [rgb, depth, beta, geo_shadow, rgb.shape[0]]
        results.append(chunk_results)
    colors, depths, betas, geo_shadows, n_rendering_samples = [
            torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
            for r in zip(*results)
        ]
    rays_shape = rays.origins.shape
    n_rendering_samples = sum(n_rendering_samples)

    results = {"rgb": colors.view((*rays_shape[:-1], -1)),
               "depth": depths.view((*rays_shape[:-1], -1)),
               "geo_shadows": geo_shadows.view((*rays_shape[:-1], -1)),
               "beta": betas.view((*rays_shape[:-1], -1)),
               }
    return results, n_rendering_samples
