"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import math
import pathlib
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from radiance_fields.eonerf import EONerfMLP
from utils import set_random_seed, load_ims_to_tensorboard, get_learning_rate, visualize_depth

from nerfacc import OccGridEstimator

from torch.utils.tensorboard import SummaryWriter
import metrics
import datetime
import os

from opt import get_opts
from datasets.satellite import load_dataset_from_args, count_training_images, save_outputs_to_images
from datasets.satellite import save_depth_priors_img, define_satrays_from_tensors
from sat_rendering import render_image, render_image_old
from sat_utils import compute_mae_and_save_dsm_diff

from torch.utils.data import DataLoader
#from osgeo import gdal

os.environ["CUDA_LAUNCH_BLOCKING"] = "0, 1"




if __name__ == "__main__":

    torch.cuda.empty_cache()
    set_random_seed(42)

    args = get_opts()
    device = f"cuda:{args.gpu_id}"

    render_n_samples = args.n_samples #1024 (default)

    # set the scene bounding box.
    # roi_aabb is a shape (6,) tensor in the format of {minx, miny, minz, maxx, maxy, maxz}
    roi_aabb = [-1., -1., -1., 1., 1., 1.]
    scene_aabb = torch.tensor(roi_aabb, dtype=torch.float32, device=device)
    near_plane = None
    far_plane = None
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max()
        / render_n_samples
    ).item()
    grid_resolution = args.n_grid

    # set the radiance field we want to train.
    max_steps = args.max_train_steps
    grad_scaler = torch.cuda.amp.GradScaler(1)
    n_input_images = count_training_images(args.root_dir)
    radiance_field = EONerfMLP(n_input_images,
                               radiometric_normalization=args.radiometric_normalization).to(device)

    optimizer = torch.optim.Adam(radiance_field.parameters(), lr=float(args.lr))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)


    # set the dataset
    train_dataset = load_dataset_from_args(args, split="train")
    val_dataset = load_dataset_from_args(args, split="val")
    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=1, batch_size=args.batch_size, pin_memory=False)
    val_loader = DataLoader(val_dataset, shuffle=True, num_workers=1, batch_size=args.batch_size, pin_memory=False)
    print("datasets successfully loaded")

    occupancy_grid = OccGridEstimator(roi_aabb=roi_aabb, resolution=grid_resolution, levels=1).to(device)
    print("occupancy grid is ready")
    #occupancy_grid = None

    # training
    log_dir = os.path.join(args.logs_dir, args.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print("tensorboard log is ready")

    # do we have prior info available ?
    depth_prior_available = train_dataset.prior_depths is not None
    if depth_prior_available:
        print(f"Using depth priors from {args.init_dsm_path} !")
    conf_prior_available = train_dataset.prior_confs is not None
    if conf_prior_available:
        print(f"Using confidence priors from {args.init_conf_path} !")
    shadow_prior_available = train_dataset.prior_shadows is not None
    if shadow_prior_available:
        print(f"Using shadow priors from {args.shadow_masks_dir} !")
    w_depth = 100.0

    step = 0
    tic = time.time()
    for epoch in range(10000000):
        for i, data in enumerate(train_loader):

            radiance_field.train()

            # get individual rays
            rays = data["rays"].to(device)
            ts = data["ts"].to(device)
            pixels = data["rgbs"].to(device)
            ray_indices = data["idx"]

            satrays = define_satrays_from_tensors(rays, ts)

            # update occupancy grid
            occupancy_grid.update_every_n_steps(
                step=step,
                occ_eval_fn=lambda x: radiance_field.query_opacity(
                    x, render_step_size
                ),
                n=50,
                occ_thre=1e-2,
            )

            # render
            results, n_rendering_samples = render_image(
                radiance_field,
                occupancy_grid,
                satrays,
                scene_aabb,
                args,
                # rendering options
                epoch_idx=epoch,
                chunk=args.chunk,
                near_plane=near_plane,
                far_plane=far_plane,
                render_step_size=render_step_size,
            )
            if n_rendering_samples == 0:
                continue

            # compute loss
            if epoch < 2:
                loss = F.mse_loss(results["rgb"], pixels)
                loss_dict = {"loss": loss, "coarse_color": loss}
            else:
                loss, loss_dict = metrics.uncertainty_aware_loss(pixels, results["rgb"], results["beta"])

            if depth_prior_available:
                depth_prior = train_dataset.prior_depths[ray_indices].to(device)
                conf_prior = train_dataset.prior_confs[ray_indices].to(device) if conf_prior_available else None
                aux_loss, aux_dict = metrics.depth_loss_L2(depth_prior, results["depth"].squeeze(), conf_prior, w_depth)
                loss, loss_dict = metrics.update_loss_with_aux_term(loss, loss_dict, aux_loss, aux_dict, epoch)

            if shadow_prior_available:
                shadow_prior = train_dataset.prior_shadows[ray_indices].to(device)
                aux_loss, aux_dict = metrics.shadow_loss_L2(shadow_prior, results["geo_shadows"].squeeze(), epoch=epoch)
                aux_dict["shadows_term1"] = aux_loss
                loss, loss_dict = metrics.update_loss_with_aux_term(loss, loss_dict, aux_loss, aux_dict, epoch, start_epoch=2)


            optimizer.zero_grad()
            # do not unscale it because we are using Adam.
            grad_scaler.scale(loss).backward()
            optimizer.step()

            with torch.no_grad():
                psnr_ = metrics.psnr(results["rgb"], pixels)

            # log metrics to tensorboard
            for k in loss_dict.keys():
                writer.add_scalar(f'train/{k}', loss_dict[k], step)
            writer.add_scalar('lr', get_learning_rate(optimizer), step)
            writer.add_scalar('epoch', epoch, step)
            writer.add_scalar('train/psnr', psnr_, step)

            if step % 1000 == 0:
                elapsed_time = time.time() - tic
                print(
                    f"epoch={epoch} | elapsed_time={elapsed_time:.2f}s | step={step} | loss={loss:.5f} | "
                    f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | psnr={psnr_:.2f}"
                )

            val_freq = (len(train_dataset)//args.batch_size)//1
            save_freq = val_freq*4
            if step > 0 and step % save_freq == 0:
                ckpt_path = os.path.join(log_dir, f"ckpts/epoch={epoch}.ckpt")
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'occ_grid_state_dict': occupancy_grid.state_dict(),
                    'model_state_dict': radiance_field.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, ckpt_path)

            #val_freq = 20
            if step > 0 and step % val_freq == 0:

                # evaluation
                radiance_field.eval()

                d_metrics = {"loss": [], "coarse_color": [], "coarse_logbeta": [], "psnr": [], "mae": []}
                n_ims_to_eval = min(5, len(val_dataset))
                with torch.no_grad():
                    for i in tqdm.tqdm(range(n_ims_to_eval)):

                        data = val_dataset[i]

                        rays = data["rays"].to(device)
                        ts = torch.zeros_like(rays[:, 0:1]).long().to(device)
                        pixels = data["rgbs"].to(device)
                        satrays = define_satrays_from_tensors(rays, ts)

                        # rendering
                        # test options
                        #test_chunk_size = 512,  # min(args.batch_size, args.chunk),
                        results, n_rendering_samples = render_image(
                            radiance_field,
                            occupancy_grid,
                            satrays,
                            scene_aabb,
                            args,
                            # rendering options
                            epoch_idx=epoch,
                            chunk=args.chunk,
                            near_plane=near_plane,
                            far_plane=far_plane,
                            render_step_size=render_step_size,
                        )

                        # compute metrics
                        loss, loss_dict = metrics.uncertainty_aware_loss(pixels, results["rgb"], results["beta"])
                        psnr_ = metrics.psnr(results["rgb"], pixels)

                        # log to tensorboard
                        h, w = data["h"], data["w"]
                        depth_ = results["depth"].view(h, w, 1)
                        tb_ims = [pixels.view(h, w, 3), results["rgb"].view(h, w, 3), results["albedo_rgb"].view(h, w, 3)]
                        if epoch < 0:
                            tb_ims += [visualize_depth(depth_)]
                        else:
                            tb_ims += [results["geo_shadows"].view(h, w, 1)]
                            if shadow_prior_available:
                                shadow_prior = torch.from_numpy(val_dataset.prior_shadows[i])
                                vals_to_penalize = metrics.differentiable_thresholding(results["geo_shadows"], 0.2) * (
                                1 - metrics.differentiable_thresholding(shadow_prior.to(results["geo_shadows"].device), 0.5))
                                tb_ims += [shadow_prior.view(h, w, 1), vals_to_penalize.view(h, w, 1)]
                            tb_ims += [visualize_depth(depth_)]
                        if i == 0:
                            load_ims_to_tensorboard(writer, f"train_{i:d}/gt_pred_depth", tb_ims, step)
                        if i == 1:
                            load_ims_to_tensorboard(writer, f"val_{i-1:d}/gt_pred_depth", tb_ims, step)

                        # save outputs to disk
                        if step > 0 and step % 2*save_freq == 0:
                            if i in [0, 1]:
                                out_dir = os.path.join(log_dir, "val") if i > 0 else os.path.join(log_dir, "train")
                                save_outputs_to_images(val_dataset, data, results, out_dir, suffix=epoch)
                                if depth_prior_available:
                                    save_depth_priors_img(val_dataset, data, args.init_dsm_path, out_dir, external_conf_path=args.init_conf_path, suffix=epoch)

                        if i != 0 and args.gt_dir is not None:
                            # compute MAE
                            #try:
                            if "IARPA" in args.root_dir:
                                res = 0.3
                                aoi_id = args.root_dir.split("/")[-1].replace("_new", "")
                            elif "JAX" in args.root_dir:
                                res = 0.5
                                aoi_id = data["src_id"][:7]
                            else:
                                res = 0.3
                                aoi_id = args.root_dir.split("/")[-1].replace("_new", "")


                            unique_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            val_im_dir = os.path.join(log_dir, "val")
                            out_path = os.path.join(val_im_dir, f"dsm/tmp_pred_dsm_{unique_id}.tif")
                            _ = val_dataset.get_dsm_from_nerf_prediction(rays.cpu(), depth_.cpu(),
                                                                        dsm_path=out_path, resolution=res)
                            assert os.path.exists(out_path)
                            mae_ = compute_mae_and_save_dsm_diff(out_path, data["src_id"],
                                                                args.gt_dir, val_im_dir, 0, aoi_id,
                                                                save=False)
                            os.remove(out_path)
                            #except:
                            #    print("warning: MAE computation failed!")
                            #    mae_ = np.nan
                            for k in loss_dict.keys():
                                d_metrics[k].append(loss_dict[k])
                            d_metrics["psnr"].append(psnr_)
                            d_metrics["mae"].append(torch.Tensor([mae_]))

                    for k in d_metrics.keys():
                        if len(d_metrics[k]) > 0:
                            mean_val = torch.mean(torch.stack(d_metrics[k]))
                            writer.add_scalar(f"val/{k}", mean_val, step)

                train_dataset.training = True

            if step == max_steps:
                print("training stops")
                exit()

            step += 1

        scheduler.step()
        if depth_prior_available:
            w_depth *= 0.8
