import numpy as np
import argparse
import json
import os
import torch
import shutil

from datasets import satellite
from sat_rendering import render_image
import sat_utils
import metrics
import rasterio
from PIL import Image
import glob

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

def load_opts(opts_path, root_dir=None, img_dir=None, gt_dir=None):
    assert os.path.exists(opts_path)
    with open(opts_path, 'r') as f:
        args = argparse.Namespace(**json.load(f))
    args.sc_lambda = 0
    #args.root_dir = "/mnt/cdisk/roger/Datasets" + args.root_dir.split("Datasets")[-1]
    #args.img_dir = "/mnt/cdisk/roger/Datasets" + args.img_dir.split("Datasets")[-1]
    #args.cache_dir = "/mnt/cdisk/roger/Datasets" + args.cache_dir.split("Datasets")[-1]
    #args.gt_dir = "/mnt/cdisk/roger/Datasets" + args.gt_dir.split("Datasets")[-1]
    #args.chunk = 512
    if args.model == "eo-nerf":
        args.radiometric_normalization = True
    if gt_dir is not None:
        assert os.path.isdir(gt_dir)
        args.gt_dir = gt_dir
    if img_dir is not None:
        assert os.path.isdir(img_dir)
        args.img_dir = img_dir
    if root_dir is not None:
        assert os.path.isdir(root_dir)
        args.root_dir = root_dir
    if not os.path.isdir(args.cache_dir):
        args.cache_dir = None
    return args


def load_eonerf_from_ckpt(args, ckpt_path, train=False, device="cuda:0"):
    from radiance_fields.eonerf import EONerfMLP
    from nerfacc import OccGridEstimator

    assert os.path.exists(ckpt_path)
    checkpoint = torch.load(ckpt_path)

    n_input_images = satellite.count_training_images(args.root_dir)
    if "radiometricT_enc.weight" in checkpoint['model_state_dict']:
        n_images_in_embdict = checkpoint['model_state_dict']["radiometricT_enc.weight"].shape[0]
        if n_input_images != n_images_in_embdict:
            print("warning: number of input is inconsistent with the shape of the embedding dictionary")
            n_input_images = n_images_in_embdict

    model = EONerfMLP(n_input_images, radiometric_normalization=args.radiometric_normalization)
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if train:
        model.train()
    else:
        model.eval()

    if "occ_grid_state_dict" in checkpoint:
        roi_aabb = [-1., -1., -1., 1., 1., 1.]
        grid_resolution = args.n_grid
        occ_grid = OccGridEstimator(roi_aabb=roi_aabb, resolution=grid_resolution, levels=1).to(device)
        occ_grid.to(device)
        occ_grid.load_state_dict(checkpoint["occ_grid_state_dict"])
    else:
        occ_grid = None

    return model, occ_grid


def create_rays_from_nadir(dataset, h, w, sun_el_deg, sun_az_deg):
    from datasets.satellite import get_dir_vec_from_el_az
    radius = 2
    #radius *= 9
    el_deg, az_deg = 0, 0
    scale = dataset.scene_scale.cpu().numpy()
    h = int(h // dataset.img_downscale)
    w = int(w // dataset.img_downscale)
    focal = max(h, w) // dataset.img_downscale
    near = max(0, radius - 2)
    far = near + 2.5
    rays = generate_rays_from_virtual_pinhole(w, h, focal, radius, el_deg, az_deg, near, far, scene_scale=scale, verbose=False)
    sun_d = get_dir_vec_from_el_az(sun_el_deg, sun_az_deg)
    sun_dirs = torch.from_numpy(np.tile(sun_d, (rays.shape[0], 1)))
    sun_dirs /= dataset.scene_scale
    sun_dirs /= np.linalg.norm(sun_dirs, axis=1)[:, np.newaxis]
    rays = torch.hstack([rays, sun_dirs.type(torch.FloatTensor)])
    return rays

def pose_spherical(theta, phi, radius, extra_transform=np.eye(4)):
    # Create the camera to world coordinate transform matrix
    # theta = azimuth
    # phi = elevation
    # radius = distance to scene

    def get_translation_matrix(t):
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, t],
                         [0, 0, 0, 1]])

    def get_rotation_matrix_phi(phi):
        # Rotation Matrix for movement in phi
        return np.array([[1, 0, 0, 0],
                         [0, np.cos(phi), np.sin(phi), 0],
                         [0, -np.sin(phi), np.cos(phi), 0],
                         [0, 0, 0, 1]])

    def get_rotation_matrix_theta(theta):
        # Rotation Matrix for movement in theta
        return np.array([[np.cos(theta), 0, -np.sin(theta), 0],
                         [0, 1, 0, 0],
                         [np.sin(theta), 0, np.cos(theta), 0],
                         [0, 0, 0, 1]])

    camera_to_world_transform = get_translation_matrix(radius)
    camera_to_world_transform = get_rotation_matrix_phi(np.radians(phi)) @ camera_to_world_transform
    camera_to_world_transform = get_rotation_matrix_theta(np.radians(theta)) @ camera_to_world_transform
    camera_to_world_transform = extra_transform @ camera_to_world_transform
    return camera_to_world_transform


def generate_rays_from_virtual_pinhole(w, h, focal, radius, elevation, azimuth, near, far,
                                       scene_scale=np.ones(3), pixel_center=0.5, verbose=False):

    from datasets.satellite import get_dir_vec_from_el_az

    if verbose:
        print("Inputs of generate_rays_from_virtual_pinhole:")
        print("     - w: {}".format(w))
        print("     - h: {}".format(h))
        print("     - focal: {:.3f}".format(focal))
        print("     - radius: {:.3f}".format(radius))
        print("     - elevation: {:.3f}".format(elevation))
        print("     - azimuth: {:.3f}".format(azimuth))
        print("     - near: {:.3f}".format(near))
        print("     - far: {:.3f}".format(far))
        print("     - scene_scale: {}".format(scene_scale))

    pinhole = False

    # get exterior parameters or pose matrix
    # camtoworld.shape = (4, 4)
    camtoworld = pose_spherical(azimuth, elevation, radius)

    # create pixel coordinates
    x, y = np.meshgrid(
        np.arange(w, dtype=np.float32) + pixel_center,  # x-axis (columns)
        np.arange(h, dtype=np.float32) + pixel_center,  # y-axis (rows)
        indexing="xy")

    # compute the camera viewing direction vector for each pixel (in camera coordinate frame)
    # camera_dirs.shape = (w, h, 3)
    camera_dirs = np.stack([(x - w * 0.5) / focal, -(y - h * 0.5) / focal, -np.ones_like(x)], axis=-1)

    # convert camera direction vectors to world coordinate frame using the pose matrix
    # directions.shape = (w, h, 3)
    if pinhole:
        directions = ((camera_dirs[Ellipsis, None, :] * camtoworld[None, None, :3, :3]).sum(axis=-1))
    else:
        dir_vec = get_dir_vec_from_el_az(elevation, azimuth)
        dir_vec /= scene_scale
        dir_vec /= np.linalg.norm(dir_vec)
        directions = np.tile(dir_vec, (h, w, 1))

    # pick the camera location given by the pose matrix as the origin (there is only one camera center for all rays)
    if pinhole:
        origins = np.broadcast_to(camtoworld[None, None, :3, -1], directions.shape)
    else:
        # TODO: Improve

        d = directions[0, 0, :]
        pt_o = np.array([0, 0, -1]) # pt_o = scene origin (center of the bottom face of the cube volume within [-1, 1])
        pt_a = pt_o - radius * d # pt_a = middle point of the output image
        # the point pt_a and the direction vector d define the image plane
        # d is perpendicular to the plane

        # find the vectors u and v defining the plane perpendicular to d
        #t = (pt_a[0]*d[0] + pt_a[1]*d[1] + pt_a[2]*d[2] + d[2]) / d[2]
        #pt_b = pt_o + np.array([0, 0, 1]) * t
        #u = pt_b - pt_a
        #v = np.cross(u, d)

        # now retreive all pixel coordinates in the image plane using the plane equation
        #x = (np.arange(w) - w * 0.5) / (1.5*w/radius) + pt_a[0]
        #y = - (np.arange(h) - h * 0.5) / (1.5*h/radius) + pt_a[1]
        #print(x)
        #exit()
        x = (np.arange(w) - w * 0.5) / (1*w/radius) + pt_a[0]
        y = - (np.arange(h) - h * 0.5) / (1*h/radius) + pt_a[1]
        X,Y = np.meshgrid(x,y)
        Z = ((- d[0]*(X-pt_a[0]) - d[1]*(Y-pt_a[1])) / d[2]) + pt_a[2]
        origins = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T


        #x = - (np.arange(w) - w * 0.5) / (w/2)
        #y = (np.arange(h) - h * 0.5) / (h/2)
        #X,Y = np.meshgrid(x,y)
        #Z= np.ones_like(X)
        #origins = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T


        """
        # define a set of origins in the upper plane of the cube
            x = (np.arange(w) - w * 0.5) / (w/2)
            y = -(np.arange(h) - h * 0.5) / (h/2)
            X,Y = np.meshgrid(x,y)
            Z = np.ones_like(X)
            origins = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    
        # for each ray, compute the intersection with the lower plane of the cube
        # line: o + t * d
        # plane = 0x + 0y + 1z = -1
        # (o[-1] + t * d[-1]) = -1 --> t = (-1 -o[-1])/d[-1]
        t = (-1. - 1. * origins[:, -1]) / d[-1]
        end_pts = origins + np.tile(t[:, np.newaxis], (1, 3))*np.tile(d, (t.shape[0], 1))
        m = 0.73 # approx. everything that falls outisde [-0.8, 0.8] cannot be trusted
        valid_ends = (end_pts[:, 0] > -m) & (end_pts[:, 0] < m) & (end_pts[:, 1] > -m) & (end_pts[:, 1] < m)
        vorigins = origins[valid_ends].copy()
        maxx, minx, maxy, miny = vorigins[:, 0].max(), vorigins[:, 0].min(), vorigins[:, 1].max(), vorigins[:, 1].min()
        #print(minx, miny, maxx, maxy)

        x = np.linspace(minx, maxx, w)
        y = np.linspace(maxy, miny, h)
        X,Y = np.meshgrid(x,y)
        Z = np.ones_like(X)
        origins = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        """

    # normalize the viewing direction vectors
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    rays_o = origins.reshape((-1, 3))
    rays_d = viewdirs.reshape((-1, 3))

    ones = np.ones_like(rays_o[Ellipsis, :1])

    rays = torch.from_numpy(np.hstack([rays_o, rays_d, near*ones, far*ones]))
    rays = rays.type(torch.FloatTensor)
    # rays.shape = (width*height, 8)

    return rays

def eval_eonerf(run_id, logs_dir, output_dir, epoch_nb=None, root_dir=None, img_dir=None, gt_dir=None, dsm=False):

    device = "cuda:0"

    # (1) load pre-trained eonerf
    opts_path = os.path.join(logs_dir, f"{run_id}/opts.json")
    args = load_opts(opts_path, root_dir=root_dir, img_dir=img_dir, gt_dir=gt_dir)
    
    if epoch_nb is None:
    	ckpt_paths = sorted(glob.glob(f"{logs_dir}/{run_id}/ckpts/*.ckpt"))
    	epoch_numbers = [int(p.split("=")[-1].split(".")[0]) for p in ckpt_paths]
    	epoch_nb = max(epoch_numbers) + 1

    ckpt_path = os.path.join(logs_dir, f"{run_id}/ckpts/epoch={epoch_nb - 1}.ckpt")
    radiance_field, occupancy_grid = load_eonerf_from_ckpt(args, ckpt_path, device=device)

    # (2) load dataset
    dataset = satellite.load_dataset_from_args(args, split="val")
    with open(os.path.join(args.root_dir, "train.txt"), "r") as f:
        json_files = f.read().split("\n")
    if os.path.exists(os.path.join(args.root_dir, "test.txt")):
        with open(os.path.join(args.root_dir, "test.txt"), "r") as f:
            json_files.extend(f.read().split("\n"))
    json_files = [j for j in json_files if ".json" in j]
    dataset.json_files = [os.path.join(args.root_dir, json_p) for json_p in json_files]
    dataset.all_ids_img = [i for i, p in enumerate(dataset.json_files)]

    # (3) evaluate
    n_ims_to_eval = len(dataset)
    for i in range(n_ims_to_eval):

        # (3.1) prepare input rays
        if dsm:
            # evaluate a single image, observed from the nadir, to get the output dsm
            json_path_closest_to_nadir = sat_utils.sort_by_increasing_view_incidence_angle(dataset.json_dir)[0]
            with open(json_path_closest_to_nadir, 'r') as f:
                d = json.load(f)
            src_id = sat_utils.get_file_id(json_path_closest_to_nadir)
            sun_el_deg = 90. - float(d["sun_elevation"])
            sun_az_deg = float(d["sun_azimuth"])
            h, w = int(d["height"]), int(d["width"])
            rays = create_rays_from_nadir(dataset, h, w, sun_el_deg, sun_az_deg).to(device)
            ts = torch.zeros(h * w, 1).long().squeeze().to(device)
            pixels = torch.ones((rays.shape[0], 3)).to(device)
            data = {"rays": rays, "rgbs": pixels, "ts": ts, "h": h, "w": w, "src_id": src_id}
            origins, viewdirs, sundirs = rays[:, :3], rays[:, 3:6], rays[:, 8:11]
            satrays = satellite.define_satrays_from_tensors(rays, ts)
        else:
            # evaluate all input images
            data = dataset[i]
            rays = data["rays"].to(device)
            ts = torch.zeros_like(rays[:, 0:1]).long().to(device)
            pixels = data["rgbs"].to(device)
            satrays = satellite.define_satrays_from_tensors(rays, ts)

        # (3.2) run model
        roi_aabb = [-1., -1., -1., 1., 1., 1.]
        scene_aabb = torch.tensor(roi_aabb, dtype=torch.float32, device=occupancy_grid.device)
        render_step_size = ((scene_aabb[3:] - scene_aabb[:3]).max()/ args.n_samples).item()
        with torch.no_grad():
            results, n_rendering_samples = render_image(
                radiance_field,
                occupancy_grid,
                satrays,
                scene_aabb,
                args,
                # rendering options
                epoch_idx=epoch_nb,
                chunk=args.chunk,
                near_plane=None,
                far_plane=None,
                render_step_size=render_step_size,
                eval = True
            )

        out_dir = os.path.join(output_dir, run_id)
        satellite.save_outputs_to_images(dataset, data, results, out_dir, downsampling_factor=args.img_downscale)

        if dsm:
            dsm_path = os.path.join(out_dir, f"dsm/{src_id}.tif")
            assert os.path.exists(dsm_path)
            print("Path to output EO-NeRF DSM: {}".format(dsm_path))

            if args.gt_dir is not None:

                if "JAX" in src_id:
                    aoi_id = src_id[:7]
                else:
                    aoi_id = args.root_dir.split("/")[-1].replace("_new", "")

                # evaluate NeRF generated DSM
                mae = sat_utils.compute_mae_and_save_dsm_diff(dsm_path, src_id, args.gt_dir, out_dir, epoch_nb, aoi_id)
                rdsm_tmp_path = os.path.join(out_dir, "{}_rdsm_epoch{}.tif".format(src_id, epoch_nb))
                rdsm_path = rdsm_tmp_path.replace(".tif", "_{:.3f}.tif".format(mae))
                shutil.copyfile(rdsm_tmp_path, rdsm_path)
                os.remove(rdsm_tmp_path)
                print("\nAltitude MAE: {:.2f}".format(np.nanmean(mae)))
                print(f"Path to GT-aligned DSM: {rdsm_path}")

                # save tmp gt DSM
                gt_dsm_path = os.path.join(args.gt_dir, "{}_DSM.tif".format(aoi_id))
                tmp_gt_path = os.path.join(output_dir, run_id, "tmp_gt.tif")
                if aoi_id in ["JAX_004", "JAX_260"]:
                    gt_seg_path = os.path.join(args.gt_dir, "{}_CLS_v2.tif".format(aoi_id))
                else:
                    gt_seg_path = os.path.join(args.gt_dir, "{}_CLS.tif".format(aoi_id))

                # apply water mask
                with rasterio.open(gt_seg_path, "r") as f:
                    mask = f.read()[0, :, :]
                    water_mask = mask.copy()
                    water_mask[mask != 9] = 0
                    water_mask[mask == 9] = 1
                if ("CLS.tif" in gt_seg_path) and (os.path.exists(gt_seg_path.replace("CLS.tif", "WATER.png"))):
                    print("found water mask!")
                    mask = np.array(Image.open(gt_seg_path.replace("CLS.tif", "WATER.png")))
                    water_mask = mask == 0
                with rasterio.open(rdsm_path, "r") as f:
                    profile = f.profile
                with rasterio.open(gt_dsm_path, "r") as f:
                    gt_dsm = f.read()[0, :, :]
                with rasterio.open(tmp_gt_path, 'w', **profile) as dst:
                    water_mask_ = np.zeros_like(gt_dsm)
                    water_mask_[:water_mask.shape[0], :water_mask.shape[1]] = water_mask
                    gt_dsm[water_mask_.astype(bool)] = np.nan
                    dst.write(gt_dsm, 1)
            return mae
        else:
            loss, loss_dict = metrics.uncertainty_aware_loss(pixels, results["rgb"], results["beta"])
            psnr_ = metrics.psnr(results["rgb"], pixels)
            print(f"({i + 1}/{n_ims_to_eval}) {data['src_id']} | loss={loss:.2f} | psnr={psnr_:.2f}")

if __name__ == '__main__':
    import fire
    fire.Fire(eval_eonerf)

