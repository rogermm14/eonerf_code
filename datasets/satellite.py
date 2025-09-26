"""
This script defines the dataloader for a dataset of multi-view satellite images
"""

import numpy as np
import os
import time

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T

import rasterio
import rpcm
import glob
import sat_utils

import collections

SatRays = collections.namedtuple("Rays", ("origins", "viewdirs", "sundirs", "img_idx", "t_near", "t_far"))

def define_satrays_from_tensors(rays, ts):
    origins, viewdirs, sundirs = rays[:, :3], rays[:, 3:6], rays[:, 8:11]
    t_near, t_far = rays[:, 6:7], rays[:, 7:8]
    return SatRays(origins=origins, viewdirs=viewdirs, sundirs=sundirs, img_idx=ts, t_near=t_near, t_far=t_far)

def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))


def load_dataset_from_args(args, split):
    d = SatelliteDataset(root_dir=args.root_dir,
                     img_dir=args.img_dir if args.img_dir is not None else args.root_dir,
                     split=split,
                     cache_dir=args.cache_dir,
                     img_downscale=args.img_downscale,
                     utm=not args.ecef,
                     prior_dsm_path=args.init_dsm_path,
                     prior_conf_path=args.init_conf_path,
                     shadow_masks_dir=args.shadow_masks_dir,
                     subset=args.subset_Nviews)
    return d

def count_training_images(root_dir):
    with open(os.path.join(root_dir, "train.txt"), "r") as f:
        json_files = f.read().split("\n")
    json_files = [p for p in json_files if len(p) > 1]
    return len(json_files)

def read_rpc_from_json(json_path):
    d = sat_utils.read_dict_from_json(json_path)
    rpc = rpcm.RPCModel(d["rpc"], dict_format="rpcm")
    return rpc

def get_dir_vec_from_el_az(elevation_deg, azimuth_deg):
    #convention: elevation is 0 degrees at nadir, 90 at frontal view
    el = np.radians(90-elevation_deg)
    az = np.radians(azimuth_deg)
    dir_vec = -1.0 * np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
    #dir_vec = dir_vec / np.linalg.norm(dir_vec)
    return dir_vec

def get_rays(cols, rows, rpc, min_alt, max_alt, utm=True):
    """
            Draw a set of rays from a satellite image
            Each ray is defined by an origin 3d point + a direction vector
            First the bounds of each ray are found by localizing each pixel at min and max altitude
            Then the corresponding direction vector is found by the difference between such bounds
            Args:
                cols: 1d array with image column coordinates
                rows: 1d array with image row coordinates
                rpc: RPC model with the localization function associated to the satellite image
                min_alt: float, the minimum altitude observed in the image
                max_alt: float, the maximum altitude observed in the image
            Returns:
                rays: (h*w, 8) tensor of floats encoding h*w rays
                      columns 0,1,2 correspond to the rays origin
                      columns 3,4,5 correspond to the direction vector
                      columns 6,7 correspond to the distance of the ray bounds with respect to the camera
            """

    min_alts = float(min_alt) * np.ones(cols.shape)
    max_alts = float(max_alt) * np.ones(cols.shape)

    # assume the points of maximum altitude are those closest to the camera
    if utm:
        lons, lats = rpc.localization(cols, rows, max_alts)
        easts, norths = sat_utils.utm_from_latlon(lats, lons)
        xyz_near = np.vstack([easts, norths, max_alts]).T
        # similarly, the points of minimum altitude are the furthest away from the camera
        lons, lats = rpc.localization(cols, rows, min_alts)
        easts, norths = sat_utils.utm_from_latlon(lats, lons)
        xyz_far = np.vstack([easts, norths, min_alts]).T
    else:
        print("using ecef to build the rays")
        lons, lats = rpc.localization(cols, rows, max_alts)
        x_near, y_near, z_near = sat_utils.latlon_to_ecef_custom(lats, lons, max_alts)
        xyz_near = np.vstack([x_near, y_near, z_near]).T
        # similarly, the points of minimum altitude are the furthest away from the camera
        lons, lats = rpc.localization(cols, rows, min_alts)
        x_far, y_far, z_far = sat_utils.latlon_to_ecef_custom(lats, lons, min_alts)
        xyz_far = np.vstack([x_far, y_far, z_far]).T

    # define the rays origin as the nearest point coordinates
    rays_o = xyz_near

    # define the unit direction vector
    d = xyz_far - xyz_near
    rays_d = d / np.linalg.norm(d, axis=1)[:, np.newaxis]

    # assume the nearest points are at distance 0 from the camera
    # the furthest points are at distance Euclidean distance(far - near)
    fars = np.linalg.norm(d, axis=1)
    nears = float(0) * np.ones(fars.shape)

    # create a stack with the rays origin, direction vector and near-far bounds
    rays = torch.from_numpy(np.hstack([rays_o, rays_d, nears[:, np.newaxis], fars[:, np.newaxis]]))
    rays = rays.type(torch.FloatTensor)
    return rays


def normalize_rays(rays, scene_offset, scene_scale, solar_dir=True):
    rays_o = rays[:, :3]
    rays_e = rays[:, :3] + rays[:, 3:6] * rays[:, 7:8]
    rays_o_n = (rays_o - scene_offset) / scene_scale
    rays_e_n = (rays_e - scene_offset) / scene_scale
    d = rays_e_n - rays_o_n
    rays_d = d / np.linalg.norm(d, axis=1)[:, np.newaxis]
    fars = np.linalg.norm(d, axis=1)
    nears = float(0) * np.ones(fars.shape)
    rays_n = np.hstack([rays_o_n, rays_d, nears[:, np.newaxis], fars[:, np.newaxis]])
    if solar_dir and rays.shape[1] == 11:
        sun_d = rays[:, 8:11]
        sun_d /= scene_scale
        sun_d /= np.linalg.norm(sun_d, axis=1)[:, np.newaxis]
        rays_n = np.hstack([rays_n, sun_d])
    return rays_n

def old_normalize_rays(rays, scene_offset, scene_scale, solar_dir=True):
    rays[:, 0] -= scene_offset[0]
    rays[:, 1] -= scene_offset[1]
    rays[:, 2] -= scene_offset[2]
    rays[:, 0] /= scene_scale
    rays[:, 1] /= scene_scale
    rays[:, 2] /= scene_scale
    rays[:, 6] /= scene_scale
    rays[:, 7] /= scene_scale
    return rays

def load_rgb_geotiff(img_path, downscale_factor=1, imethod=Image.BICUBIC):
    if ".tif" in img_path:
        with rasterio.open(img_path, 'r') as f:
            img = np.transpose(f.read(), (1, 2, 0))
    elif ".png" in img_path:
        img = np.array(Image.open(img_path))[:, :, np.newaxis] # (h, w, 1)
        img = np.tile(img, (1, 1, 3))  # (h, w, 3)
    else:
        print(f"error! unknown file extension of {img_path}")
        exit()
    img = img / 255. if img.max() > 1.1 else img
    img = np.clip(img, 0, 1)
    h, w = img.shape[:2]
    if downscale_factor > 1:
        w = int(w // downscale_factor)
        h = int(h // downscale_factor)
        img = np.transpose(img, (2, 0, 1))
        img = T.Resize(size=(h, w), interpolation=imethod, antialias=True)(torch.Tensor(img))
        img = np.transpose(img.numpy(), (1, 2, 0))
    img = np.clip(img, 0, 1)
    return img # (h, w, 3)

def save_output_image(input, output_path, source_path):
    """
    input: (D, H, W) where D is the number of channels (3 for rgb, 1 for grayscale)
           can be a pytorch tensor or a numpy array
    """
    # convert input to numpy array float32
    if torch.is_tensor(input):
        im_np = input.type(torch.FloatTensor).cpu().numpy()
    else:
        im_np = input.astype(np.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(source_path, 'r') as src:
        profile = src.profile
        profile["dtype"] = rasterio.float32
        profile["height"] = im_np.shape[1]
        profile["width"] = im_np.shape[2]
        profile["count"] = im_np.shape[0]
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(im_np)

def save_outputs_to_images(dataset, sample, results, out_dir, suffix="", downsampling_factor=1):

    rays = sample["rays"].squeeze()
    rgbs = sample["rgbs"].squeeze()
    src_id = sample["src_id"]
    src_path = os.path.join(dataset.img_dir, src_id + ".tif")
    W, H = sample["w"], sample["h"]
    if downsampling_factor > 1:
        W, H = W//2, H//2

    # save 1-channel image outputs
    for k in ["geo_shadows", "transient_s", "beta"]:
        if k in results:
            img = results[f"{k}"].view(1, H, W).repeat((3, 1, 1)).cpu()
            out_path = "{}/{}/{}{}.tif".format(out_dir, k, src_id, suffix)
            save_output_image(img, out_path, src_path)
    # save 3-channel image outputs
    for k in ["rgb", "ambient_rgb", "albedo_rgb"]:
        if k in results:
            img = results[f"{k}"].view(H, W, 3).permute(2, 0, 1).cpu()
            out_path = "{}/{}/{}{}.tif".format(out_dir, k, src_id, suffix)
            save_output_image(img, out_path, src_path)
    # save gt rgb image
    img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    out_path = "{}/gt_rgb/{}{}.tif".format(out_dir, src_id, suffix)
    save_output_image(img_gt, out_path, src_path)
    # save altitude and DSM
    if "depth" in results:
        _, _, alts = dataset.get_utmalt_from_nerf_prediction(rays.cpu(), results["depth"].cpu())
        out_path = "{}/depth/{}{}.tif".format(out_dir, src_id, suffix)
        save_output_image(alts.reshape(1, H, W), out_path, src_path)
        # save dsm
        out_path = "{}/dsm/{}{}.tif".format(out_dir, src_id, suffix)
        dsm_res = 0.5 if "JAX" in src_id else 0.3
        _ = dataset.get_dsm_from_nerf_prediction(rays.cpu(), results["depth"].cpu(), dsm_path=out_path, resolution=dsm_res)
    """
    if "pts_per_ray" in results:
        pts_per_ray = results["pts_per_ray"].cpu().reshape(1, H, W)
        out_path = "{}/pts_per_ray/{}_{}.tif".format(out_dir, src_id, suffix)
        save_output_image(pts_per_ray, out_path, src_path)
    if "sc_pts_per_ray" in results:
        sc_pts_per_ray = results["sc_pts_per_ray"].cpu().reshape(1, H, W)
        out_path = "{}/sc_pts_per_ray/{}_{}.tif".format(out_dir, src_id, suffix)
        save_output_image(sc_pts_per_ray, out_path, src_path)
    """

def save_depth_priors_img(dataset, sample, external_dsm_path, out_dir, external_conf_path=None, suffix=""):
    rays = sample["rays"].squeeze()
    src_id = sample["src_id"]
    src_path = os.path.join(dataset.img_dir, src_id + ".tif")
    W, H = sample["w"], sample["h"]
    json_path = os.path.join(dataset.json_dir, src_id + ".json")

    # save altitude
    depth_prior, conf_prior = dataset.load_depth_priors_from_dsm(external_dsm_path, external_conf_path, [json_path])
    depth_prior = depth_prior[0]
    _, _, alts = dataset.get_utmalt_from_nerf_prediction(rays.cpu(), depth_prior)
    out_path = "{}/depth_prior/{}{}.tif".format(out_dir, src_id, suffix)
    alts[depth_prior.cpu().numpy() < 0.] = np.nan
    save_output_image(alts.reshape(1, H, W), out_path, src_path)
    # save dsm
    out_path = "{}/dsm_prior/{}{}.tif".format(out_dir, src_id, suffix)
    dsm_res = 0.5 if "JAX" in src_id else 0.3
    _ = dataset.get_dsm_from_nerf_prediction(rays.cpu(), depth_prior.cpu(), dsm_path=out_path, resolution=dsm_res)
    # save conf prior
    if external_conf_path is not None:
        conf_prior = conf_prior[0]
        out_path = "{}/conf_prior/{}{}.tif".format(out_dir, src_id, suffix)
        conf_prior[conf_prior < 0.] = np.nan
        save_output_image(conf_prior.reshape(1, H, W), out_path, src_path)

def sort_from_more_shadows_to_less_shadows(shadow_mask_vectors):
    # Compute the number of 0s in each vector
    zero_counts = [np.sum(vec == 0) for vec in shadow_mask_vectors]
    # Get the sorted indices in decreasing order of zero count
    sorted_indices = np.argsort(zero_counts)[::-1]
    return sorted_indices.tolist()

class SatelliteDataset(Dataset):
    def __init__(self, root_dir, img_dir, split="train", img_downscale=1.0, utm=True, cache_dir=None,
                 prior_dsm_path=None, prior_conf_path=None, shadow_masks_dir=None, subset=None):
        """
        NeRF Satellite Dataset
        Args:
            root_dir: string, directory containing the json files with all relevant metadata per image
            img_dir: string, directory containing all the satellite images (may be different from root_dir)
            split: string, either 'train' or 'val'
            img_downscale: float, image downscale factor
            cache_dir: string, directory containing precomputed rays
        """
        self.json_dir = root_dir
        self.img_dir = img_dir
        self.cache_dir = cache_dir
        self.train = split == "train"
        self.img_downscale = float(img_downscale)
        self.white_back = False
        self.utm_sampling = utm
        self.subset = subset
        self.shadow_masks_dir = shadow_masks_dir

        assert os.path.exists(root_dir), f"root_dir {root_dir} does not exist"
        assert os.path.exists(img_dir), f"img_dir {img_dir} does not exist"

        # load scaling params
        loc_path = os.path.join(self.json_dir, "scene.loc_{}".format("utm" if self.utm_sampling else "ecef"))
        if not os.path.exists(loc_path):
            self.init_scaling_params()
        d = sat_utils.read_dict_from_json(loc_path)
        self.scene_offset = torch.tensor([float(d["X_offset"]), float(d["Y_offset"]), float(d["Z_offset"])])
        self.set_utm_zonestring()

        if self.utm_sampling:
            self.scene_scale = torch.tensor([float(d["X_scale"]), float(d["Y_scale"]), float(d["Z_scale"])])
        else:
            self.scene_scale = torch.tensor([float(d["X_scale"]), float(d["Y_scale"]), float(d["Z_scale"])]).max()

        # load dataset split
        if self.train:
            self.load_train_split()
        else:
            self.load_val_split()

        # load depth priors
        if prior_dsm_path is None:
            self.prior_depths, self.prior_confs = None, None
        else:
            assert os.path.exists(prior_dsm_path)
            self.prior_depths, self.prior_confs = self.load_depth_priors_from_dsm(prior_dsm_path, prior_conf_path)
        # load shadow priors
        if shadow_masks_dir is None:
            self.prior_shadows = None
        else:
            self.prior_shadows = self.load_shadow_masks(shadow_masks_dir)
            if self.train:
                self.n_shadow_rays = int(torch.sum(self.prior_shadows).item())
                self.shadow_rays_indices = torch.arange(self.all_rays.shape[0])[self.prior_shadows.long()]
                self.n_nonshadow_rays = int(torch.sum(1 - self.prior_shadows).item())
                self.nonshadow_rays_indices = torch.arange(self.all_rays.shape[0])[(1 - self.prior_shadows).long()]

    def set_utm_zonestring(self):
        with open(os.path.join(self.json_dir, "train.txt"), "r") as f:
            json_files = f.read().split("\n")
        d = sat_utils.read_dict_from_json(os.path.join(self.json_dir, json_files[0]))
        lat_offset, lon_offset = d["rpc"]["lat_offset"], d["rpc"]["lon_offset"]
        self.utm_zonestring = sat_utils.utm_zonstring_from_lonlat(lon_offset, lat_offset)

    def load_train_split(self):
        with open(os.path.join(self.json_dir, "train.txt"), "r") as f:
            json_files = f.read().split("\n")
        json_files = [j for j in json_files if ".json" in j]
        if self.subset is not None and self.subset > 1:
            total_ims = len(json_files) 
            if self.shadow_masks_dir is None:
                json_files = np.array(json_files)[:self.subset].tolist() #select first N views
            else:
                priority_imgs_with_more_shadows = False # not used for s-eo dataset, but could be interesting...
                if priority_imgs_with_more_shadows:
                    json_files = [os.path.join(self.json_dir, json_p) for json_p in json_files]
                    shadow_mask_vecs = self.load_shadow_masks(self.shadow_masks_dir, json_files=json_files, train=False)
                    decreasing_shadows_indices = sort_from_more_shadows_to_less_shadows(shadow_mask_vecs)
                    json_files = np.array(json_files)[decreasing_shadows_indices][:self.subset].tolist() #select the N views with more shadows
                    json_files = [os.path.basename(json_p) for json_p in json_files]
                else:
                    json_files = np.array(json_files)[:self.subset].tolist() #select first N views
            print(f"\nIMPORTANT! --subset_Nviews {self.subset} is active. Using only {self.subset} training images (out of {total_ims})\n")
        self.json_files = [os.path.join(self.json_dir, json_p) for json_p in json_files]
        self.all_rays, self.all_rgbs, self.all_ids_img, self.all_img_shapes, self.all_rpcs = self.load_data(self.json_files, verbose=True)

    def load_val_split(self):
        with open(os.path.join(self.json_dir, "test.txt"), "r") as f:
            json_files = f.read().split("\n")
        json_files = [j for j in json_files if ".json" in j]
        self.json_files = [os.path.join(self.json_dir, json_p) for json_p in json_files]
        # add an extra image from the training set to the validation set (for debugging purposes)
        with open(os.path.join(self.json_dir, "train.txt"), "r") as f:
            json_files = f.read().split("\n")
        json_files = [j for j in json_files if ".json" in j]
        n_train_ims = len(json_files)
        self.all_ids_img = [i + n_train_ims for i, j in enumerate(self.json_files)]
        self.json_files = [os.path.join(self.json_dir, json_files[0])] + self.json_files
        self.all_ids_img = [0] + self.all_ids_img

    def init_scaling_params(self):
        print("Could not find a scene.loc file in the root directory, creating one...")
        print("Warning: this can take some minutes")
        all_json = glob.glob("{}/*.json".format(self.json_dir))
        all_rays = []
        for json_p in all_json:
            d = sat_utils.read_dict_from_json(json_p)
            h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
            rpc = sat_utils.rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.img_downscale)
            min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])
            #min_alt, max_alt = min_alt - 5, max_alt +5
            #cols, rows = np.meshgrid(np.arange(w), np.arange(h))
            #cols, rows = cols.flatten(), rows.flatten()
            cols = np.array(2*[0, w-1, w-1, 0])
            rows = np.array(2*[0, 0, h-1, h-1])
            rays = get_rays(cols, rows, rpc, min_alt, max_alt, utm=self.utm_sampling)
            all_rays += [rays]
        all_rays = torch.cat(all_rays, 0)
        near_points = all_rays[:, :3]
        far_points = all_rays[:, :3] + all_rays[:, 7:8] * all_rays[:, 3:6]
        all_points = torch.cat([near_points, far_points], 0)

        d = {}
        d["X_scale"], d["X_offset"] = sat_utils.rpc_scaling_params(all_points[:, 0])
        d["Y_scale"], d["Y_offset"] = sat_utils.rpc_scaling_params(all_points[:, 1])
        d["Z_scale"], d["Z_offset"] = sat_utils.rpc_scaling_params(all_points[:, 2])
        sat_utils.write_dict_to_json(d, "{}/scene.loc_{}".format(self.json_dir, "utm" if self.utm_sampling else "ecef"))
        print("... done !")

    def load_data(self, json_files, verbose=False):
        """
        Load all relevant information from a set of json files
        Args:
            json_files: list containing the path to the input json files
        Returns:
            all_rays: (N, 11) tensor of floats encoding all ray-related parameters corresponding to N rays
                      columns 0,1,2 correspond to the rays origin
                      columns 3,4,5 correspond to the direction vector
                      columns 6,7 correspond to the distance of the ray bounds with respect to the camera
                      columns 8,9,10 correspond to the sun direction vectors
            all_rgbs: (N, 3) tensor of floats encoding all the rgb colors corresponding to N rays
            all_ids_pixel : (N) tensor of integers encoding the index i*w+j corresponding to the pixel position on the image corresponding to N rays
            all_img_shapes: (N, 2) tensor of integers encoding the width and the height of the image corresponding to N rays

        """
        all_rgbs, all_rays, all_sun_dirs, all_ids_img, all_img_shapes, all_rpcs = [], [], [], [], [], []
        for t, json_p in enumerate(json_files):

            # read json, image path and id
            d = sat_utils.read_dict_from_json(json_p)
            img_p = os.path.join(self.img_dir, d["img"])
            img_id = sat_utils.get_file_id(d["img"])

            # get rgb colors and image size
            rgbs = load_rgb_geotiff(img_p, self.img_downscale)
            rgbs = rgbs.reshape((-1, 3)) # (h*w, 3)
            h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)

            # get rpc
            rpc = sat_utils.rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.img_downscale)
            all_rpcs.append(rpc)

            # get rays
            recompute = True
            cache_path = "{}/{}.data".format(self.cache_dir, img_id)
            if self.cache_dir is not None and os.path.exists(cache_path):
                rays = torch.load(cache_path).cpu().numpy()
                if rays.shape[1] == 11:
                    recompute = False
            else:
                min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])
                #min_alt, max_alt = min_alt -5, max_alt +5
                cols, rows = np.meshgrid(np.arange(w), np.arange(h))
                rays = get_rays(cols.flatten(), rows.flatten(), rpc, min_alt, max_alt, utm=self.utm_sampling)
                if self.cache_dir is not None:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    torch.save(rays, cache_path)

            # get sun direction
            if recompute:
                sun_dirs = self.get_sun_dirs(90-float(d["sun_elevation"]), float(d["sun_azimuth"]), rays.shape[0])
                rays = np.hstack([rays, sun_dirs])

            all_ids_img += [t * np.ones((rays.shape[0], 1))]
            all_rgbs += [rgbs]
            all_rays += [rays]
            all_img_shapes += [[h, w]]

            if verbose:
                print("Image {} loaded ( {} / {} )".format(img_id, t + 1, len(json_files)))

        all_ids_img = torch.from_numpy(np.concatenate(all_ids_img, 0))
        all_rgbs = torch.from_numpy(np.concatenate(all_rgbs, 0))  # (len(json_files)*h*w, 3)
        all_rgbs = all_rgbs.type(torch.FloatTensor)
        all_rays = np.concatenate(all_rays, 0)
        if recompute:
            if self.utm_sampling:
                all_rays = self.normalize_rays(all_rays) # (len(json_files)*h*w, 11)
            else:
                all_rays = old_normalize_rays(all_rays, self.scene_offset.cpu().numpy(), self.scene_scale.cpu().numpy())
        all_rays = torch.from_numpy(all_rays)
        all_rays = all_rays.type(torch.FloatTensor)
        all_img_shapes = torch.from_numpy(np.array(all_img_shapes))

        return all_rays, all_rgbs, all_ids_img, all_img_shapes, all_rpcs

    def normalize_rays(self, rays):
        return normalize_rays(rays, self.scene_offset.cpu().numpy(), self.scene_scale.cpu().numpy())

    def get_sun_dirs(self, sun_elevation_deg, sun_azimuth_deg, n_rays):
        """
        Get sun direction vectors
        Args:
            sun_elevation_deg: float, sun elevation in  degrees
            sun_azimuth_deg: float, sun azimuth in degrees
            n_rays: number of rays affected by the same sun direction
        Returns:
            sun_dirs: (n_rays, 3) 3-valued unit vector encoding the sun direction, repeated n_rays times
        """
        sun_d = get_dir_vec_from_el_az(sun_elevation_deg, sun_azimuth_deg)
        if not self.utm_sampling:
            sun_d *= -1.
        sun_dirs = np.tile(sun_d, (n_rays, 1))
        return sun_dirs

    def get_utmalt_from_nerf_prediction(self, rays, depth, double=True):
        """
        Compute an image of altitudes from a NeRF depth prediction output
        Args:
            rays: (h*w, 11) tensor of input rays
            depth: (h*w, 1) tensor with nerf depth prediction
        Returns:
            lats: numpy vector of length h*w with the latitudes of the predicted points
            lons: numpy vector of length h*w with the longitude of the predicted points
            alts: numpy vector of length h*w with the altitudes of the predicted points
        """

        # convert inputs to double (avoids loss of resolution later when the tensors are converted to numpy)
        if double:
            rays = rays.double()
            depth = depth.double()

        # use input rays + predicted sigma to construct a point cloud
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]
        xyz_n = rays_o + rays_d * depth.view(-1, 1)

        # denormalize prediction to obtain utm-alt coordinates
        xyz = (xyz_n * self.scene_scale.to(rays.device)) + self.scene_offset.to(rays.device)

        # convert to easts-norths-alts
        #xyz = xyz.data.numpy()
        if self.utm_sampling:
            easts, norths, alts = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        else:
            lats, lons, alts = sat_utils.ecef_to_latlon_custom(xyz[:, 0], xyz[:, 1], xyz[:, 2])
            easts, norths = sat_utils.utm_from_latlon(lats, lons)
        return easts, norths, alts

    def get_lonlatalt_from_nerf_prediction(self, rays, depth, differentiable=True):
        easts, norths, alts = self.get_utmalt_from_nerf_prediction(rays, depth, double=False)
        if differentiable:
            zone = int(self.utm_zonestring[:-1])
            lons, lats = sat_utils.utm_to_lonlat_differentiable(easts, norths, zone)
        else:
            easts, norths, alts = easts.detach().cpu().numpy(), norths.detach().cpu().numpy(), alts.detach().cpu().numpy()
            lons, lats = sat_utils.lonlat_from_utm(easts, norths, self.utm_zonestring)
        return lons, lats, alts

    def get_dsm_from_nerf_prediction(self, rays, depth, dsm_path=None, roi_txt=None, resolution=0.5):
        """
        Compute a DSM from a NeRF depth prediction output
        Args:
            rays: (h*w, 11) tensor of input rays
            depth: (h*w, 1) tensor with nerf depth prediction
            dsm_path (optional): string, path to output DSM, in case you want to write it to disk
            roi_txt (optional): compute the DSM only within the bounds of the region of interest of the txt
        Returns:
            dsm: (h, w) numpy array with the output dsm
        """

        # get point cloud from nerf depth prediction
        easts, norths, alts = self.get_utmalt_from_nerf_prediction(rays, depth)
        cloud = np.vstack([easts, norths, alts]).T
        cloud[cloud[:, 1] < 0, 1] += 10e6
        # negative depths are not allowed
        cloud = cloud[depth.cpu().squeeze().flatten().numpy() >= 0., :]

        # (optional) read region of interest, where lidar GT is available
        if roi_txt is not None:
            gt_roi_metadata = np.loadtxt(roi_txt)
            xoff, yoff = gt_roi_metadata[0], gt_roi_metadata[1]
            xsize, ysize = int(gt_roi_metadata[2]), int(gt_roi_metadata[2])
            resolution = gt_roi_metadata[3]
            yoff += ysize * resolution  # weird but seems necessary ?
        else:
            xmin, xmax = cloud[:, 0].min(), cloud[:, 0].max()
            ymin, ymax = cloud[:, 1].min(), cloud[:, 1].max()
            xoff = np.floor(xmin / resolution) * resolution
            xsize = int(1 + np.floor((xmax - xoff) / resolution))
            yoff = np.ceil(ymax / resolution) * resolution
            ysize = int(1 - np.floor((ymin - yoff) / resolution))


        from plyflatten import plyflatten
        from plyflatten.utils import rasterio_crs, crs_proj
        import utm
        import affine
        import rasterio

        # run plyflatten
        dsm = plyflatten(cloud, xoff, yoff, resolution, xsize, ysize, radius=1, sigma=float("inf"))

        d = sat_utils.read_dict_from_json(self.json_files[0])
        lat_offset, lon_offset = d["rpc"]["lat_offset"], d["rpc"]["lon_offset"]
        n = utm.latlon_to_zone_number(lat_offset, lon_offset)
        l = utm.latitude_to_zone_letter(lat_offset)
        crs_proj = rasterio_crs(crs_proj("{}{}".format(n, l), crs_type="UTM"))

        # (optional) write dsm to disk
        if dsm_path is not None:
            os.makedirs(os.path.dirname(dsm_path), exist_ok=True)
            profile = {}
            profile["dtype"] = dsm.dtype
            profile["height"] = dsm.shape[0]
            profile["width"] = dsm.shape[1]
            profile["count"] = 1
            profile["driver"] = "GTiff"
            profile["nodata"] = float("nan")
            profile["crs"] = crs_proj
            profile["transform"] = affine.Affine(resolution, 0.0, xoff, 0.0, -resolution, yoff)
            with rasterio.open(dsm_path, "w", **profile) as f:
                f.write(dsm[:, :, 0], 1)

        return dsm

    def get_rgb_img_as_HWC(self, img_idx):
        h, w = self.all_img_shapes[img_idx]
        img_len = torch.prod(self.all_img_shapes[img_idx])
        first_ray_idx_of_img = torch.cumsum(torch.prod(self.all_img_shapes, dim=1), dim=0)[:-1]
        first_ray_idx_of_img = torch.cat([torch.Tensor([0], device=first_ray_idx_of_img.device), first_ray_idx_of_img])
        first_ray_idx_of_img = first_ray_idx_of_img[img_idx].long()
        return self.all_rgbs[first_ray_idx_of_img:first_ray_idx_of_img+img_len].reshape(h,w,3)

    def load_depth_priors_from_dsm(self, prior_dsm_path, prior_conf_path=None, json_files=None, verbose=False):

        assert os.path.exists(prior_dsm_path)
        all_depths, all_conf = [], []
        json_files_ = self.json_files if json_files is None else json_files

        t0 = time.time()
        if verbose:
            print(f"Generating depth priors from {prior_dsm_path}")

        for i, json_path in enumerate(json_files_):

            d = sat_utils.read_dict_from_json(json_path)
            img_id = sat_utils.get_file_id(d["img"])

            cache_path = "{}/{}.depth".format(self.cache_dir, img_id)
            if self.cache_dir is not None and os.path.exists(cache_path):
                depth = torch.load(cache_path)
            else:
                h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
                rpc = sat_utils.rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.img_downscale)
                dsm_alts_np = sat_utils.reproject_dsm_alt_to_satellite_image(prior_dsm_path, h, w, rpc)
                dsm_alts = torch.from_numpy(dsm_alts_np.ravel())

                # convert dsm altitude to depth
                rays, _, _, _, _ = self.load_data([json_path], verbose=False)
                rays = rays.double()
                dsm_alts = dsm_alts.double()
                dsm_alts_ = (dsm_alts - self.scene_offset[-1]) / self.scene_scale[-1]
                rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]
                depth = (dsm_alts_ - rays_o[:, -1])/rays_d[:, -1]

                # set nans to negative numbers (valid depth values are always positive)
                depth[torch.isnan(depth)] = -1.0

            """
            h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
            depth_ = depth.view(h,w).cpu().numpy()
            output_path = f"exp/depth_reprojected_{i}.tif"
            dst_profile = {"count": 1, "height": h, "width": w, "driver": "GTiff", "dtype": np.float32}
            with rasterio.open(output_path, "w", **dst_profile) as dst:
                dst.write(depth_, 1)
            """

            if self.cache_dir is not None:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save(depth, cache_path)
            all_depths.append(depth)

            # (optional) get sgm confidence
            if prior_conf_path is not None and os.path.exists(prior_conf_path):
                cache_path2 = "{}/{}.conf".format(self.cache_dir, img_id)
                if self.cache_dir is not None and os.path.exists(cache_path2):
                    conf = torch.load(cache_path2)
                else:
                    h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
                    rpc = sat_utils.rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.img_downscale)
                    conf = sat_utils.reproject_dsm_alt_to_satellite_image(prior_dsm_path, h, w, rpc, other_val_path=prior_conf_path)
                    conf = torch.from_numpy(conf.ravel())
                    conf[np.isnan(conf)] = -1.0

                """
                h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
                depth_ = conf.view(h,w).cpu().numpy()
                output_path = f"exp/conf_reprojected_{i}.tif"
                dst_profile = {"count": 1, "height": h, "width": w, "driver": "GTiff", "dtype": np.float32}
                with rasterio.open(output_path, "w", **dst_profile) as dst:
                    dst.write(depth_, 1)
                """

                if self.cache_dir is not None:
                    torch.save(conf, cache_path2)
                all_conf.append(conf)
            else:
                all_conf = None

            if verbose:
                print(f"depth prior {i+1}/{len(json_files_)} done")

        if self.train:
            all_depths = torch.from_numpy(np.concatenate(all_depths, 0))
            all_depths = all_depths.type(torch.FloatTensor)
            if prior_conf_path is not None and os.path.exists(prior_conf_path):
                all_conf = torch.from_numpy(np.concatenate(all_conf, 0))
                all_conf = all_conf.type(torch.FloatTensor)

        if verbose:
            print(f"done in {int(time.time() - t0)} seconds!")

        return all_depths, all_conf

    def get_first_ray_idx_of_img_idx(self, img_idx):
        first_ray_idx_of_img = torch.cumsum(torch.prod(self.all_img_shapes, dim=1), dim=0)[:-1]
        first_ray_idx_of_img = torch.cat([torch.Tensor([0], device=first_ray_idx_of_img.device), first_ray_idx_of_img])
        return first_ray_idx_of_img[img_idx].long()

    def get_ray_index_from_colrowimg(self, cols, rows, img_idx):
        # index 1d = row * width + col
        img_idx = img_idx.long()
        pixel_indices1d = (rows.flatten() * self.all_img_shapes[img_idx, 1] + cols.flatten()).long()  # 1d index within image
        ray_indices1d = self.get_first_ray_idx_of_img_idx(img_idx) + pixel_indices1d
        ray_indices1d = ray_indices1d.long()
        return ray_indices1d

    def get_colrowimg_from_ray_index(self, ray_idx):
        img_idx = self.all_ids_img[ray_idx].flatten().long().to(ray_idx.device)
        pixel_idx1d = ray_idx - self.get_first_ray_idx_of_img_idx(img_idx)
        rows = pixel_idx1d // self.all_img_shapes[img_idx, 1]
        cols = pixel_idx1d % self.all_img_shapes[img_idx, 1]
        return cols, rows, img_idx

    def get_patch_from_index(self, idx, patch_size=0):
        """
        Args:
            idx: index of the ray
            patch_size: size of the patch around the ray (2*patch_size+1 * 2*patch_size+1)
        Returns:
            indices: indices of the rays belonging to the patch
        """
        if patch_size==0:
            return idx
        else:
            #import timeit
            #t0 = timeit.default_timer()
            idx_img = self.all_ids_img[idx].long()
            idx_pixel = (idx - self.get_first_ray_idx_of_img_idx(idx_img)).item()
            h = self.all_img_shapes[idx_img, 0].squeeze()
            w = self.all_img_shapes[idx_img, 1].squeeze()
            #[[h, w]] = self.all_img_shapes[idx_img]
            i, j = idx_pixel//w, idx_pixel%w # i = row, j = col
            #print(i, j)
            j_min, j_max = j-patch_size//2, j+patch_size//2 + patch_size % 2
            i_min, i_max = i-patch_size//2, i+patch_size//2 + patch_size % 2
            j_min = torch.clamp(j_min, min=0, max=w-1)
            j_max = torch.clamp(j_max, min=0, max=w-1)
            i_min = torch.clamp(i_min, min=0, max=h-1)
            i_max = torch.clamp(i_max, min=0, max=h-1)
            step_i = (i_max - i_min)/patch_size
            step_j = (j_max - j_min)/patch_size
            patch_rows, patch_cols = torch.meshgrid(torch.arange(start=i_min, end=i_max, step=step_i),
                                                    torch.arange(start=j_min, end=j_max, step=step_j))
            patch_rows = patch_rows.flatten().long()
            patch_cols = patch_cols.flatten().long()
            patch_ray_indices1d = self.get_ray_index_from_colrowimg(patch_cols, patch_rows, torch.ones_like(patch_rows) * idx_img)
            #print(f"done in {timeit.default_timer() - t0} seconds")
        return patch_ray_indices1d

    def load_shadow_masks(self, shadow_masks_dir, json_files=None, train=None):

        assert os.path.exists(shadow_masks_dir)
        json_files_ = self.json_files if json_files is None else json_files
        train_ = self.train if train is None else train

        # shadows in the shadow mask are black (0), the rest is white (1)

        shadow_masks = []
        for json_p in json_files_:
            d = sat_utils.read_dict_from_json(json_p)
            img_p = os.path.join(shadow_masks_dir, d["img"])
            if not os.path.exists(img_p):
                img_p = img_p.replace(".tif", ".png")
            


            smask = load_rgb_geotiff(img_p, self.img_downscale)[:, :, 0]
            shadow_threshold = 0.3
            smask[smask > shadow_threshold] = 1.
            smask[smask <= shadow_threshold] = 0.
            smask = smask.reshape((-1, 1))  # (h*w, 3)
            shadow_masks.append(smask)

        if train_:
            shadow_masks = torch.from_numpy(np.concatenate(shadow_masks, 0))
            shadow_masks = shadow_masks.type(torch.FloatTensor)
            shadow_masks = shadow_masks.squeeze()

        return shadow_masks


    def __len__(self):
        # compute length of dataset
        if self.train:
            return self.all_rays.shape[0]
        else:
            return len(self.json_files)

    def __getitem__(self, idx, patch_size=0):
        # take a batch from the dataset
        if self.train:
            patch_indices = self.get_patch_from_index(idx, patch_size=patch_size)
            sample = {"rays": self.all_rays[patch_indices], "rgbs": self.all_rgbs[patch_indices],
                      "ts": self.all_ids_img[patch_indices].long(), "idx": patch_indices}
        else:
            rays, rgbs, _, _, _ = self.load_data([self.json_files[idx]])
            ts = self.all_ids_img[idx] * torch.ones(rays.shape[0], 1)
            d = sat_utils.read_dict_from_json(self.json_files[idx])
            img_id = sat_utils.get_file_id(d["img"])
            h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
            sample = {"rays": rays, "rgbs": rgbs, "ts": ts.long(), "src_id": img_id, "h": h, "w": w, "idx": idx}
        return sample
