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
from radiance_fields.mlp import VanillaNeRFRadianceField
from utils import set_random_seed, load_ims_to_tensorboard, get_learning_rate
from utils2 import render_image_with_occgrid
from datasets.nerf_synthetic import SubjectLoader

from nerfacc.estimators.occ_grid import OccGridEstimator

from torch.utils.tensorboard import SummaryWriter
import metrics
import datetime
import os

if __name__ == "__main__":

    device = "cuda:0"
    set_random_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="/mnt/cdisk/roger/Datasets/nerf_synthetic",
        help="the root dir of the dataset",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="trainval",
        choices=["train", "trainval"],
        help="which train split to use",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="lego",
        choices=[
            # nerf synthetic
            "chair",
            "drums",
            "ficus",
            "hotdog",
            "lego",
            "materials",
            "mic",
            "ship",
            # mipnerf360 unbounded
            "garden",
        ],
        help="which scene to use",
    )
    parser.add_argument(
        "--test_chunk_size",
        type=int,
        default=1024,
    )

    parser.add_argument("--cone_angle", type=float, default=0.0)
    parser.add_argument("--logs_dir", type=str, default="/mnt/cdisk/roger/eonerfacc_logs",
                        help="output directory to save experiment logs")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="experiment name")
    parser.add_argument("--model", type=str, default="eo-nerf", choices=['nerf', 's-nerf', 'sat-nerf', 'eo-nerf'],
                        help="which NeRF to use")
    args = parser.parse_args()
    exp_id = args.model if args.exp_name is None else args.exp_name
    args.exp_name = "{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), exp_id)

    # training parameters
    max_steps = 50000
    init_batch_size = 1024
    target_sample_batch_size = 1 << 16
    # scene parameters
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
    near_plane = 0.0
    far_plane = 1.0e10
    # model parameters
    grid_resolution = 64
    grid_nlvl = 1
    # render parameters
    render_step_size = 5e-3

    # setup the radiance field we want to train.
    max_steps = 50000
    radiance_field = VanillaNeRFRadianceField().to(device)
    optimizer = torch.optim.Adam(radiance_field.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            max_steps // 2,
            max_steps * 3 // 4,
            max_steps * 5 // 6,
            max_steps * 9 // 10,
        ],
        gamma=0.33,
    )

    # setup the dataset
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}
    train_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=args.data_root,
        split=args.train_split,
        num_rays=init_batch_size,
        **train_dataset_kwargs,
    )

    test_dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=args.data_root,
        split="test",
        num_rays=None,
        **test_dataset_kwargs,
    )

    estimator = OccGridEstimator(
    roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl).to(device)

    # training
    log_dir = os.path.join(args.logs_dir, args.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    step = 0
    tic = time.time()
    for epoch in range(10000000):
        for i in range(len(train_dataset)):
            radiance_field.train()
            estimator.train()

            data = train_dataset[i]

            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            pixels = data["pixels"]

            def occ_eval_fn(x):
                density = radiance_field.query_density(x)
                return density * render_step_size

            # update occupancy grid
            estimator.update_every_n_steps(
                step=step,
                occ_eval_fn=occ_eval_fn,
                occ_thre=1e-2,
            )

            # render
            rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
                radiance_field,
                estimator,
                rays,
                # rendering options
                near_plane=near_plane,
                render_step_size=render_step_size,
                render_bkgd=render_bkgd
            )
            if n_rendering_samples == 0:
                continue

            if target_sample_batch_size > 0:
                # dynamic batch size for rays to keep sample batch size constant.
                num_rays = len(pixels)
                num_rays = int(
                    num_rays * (target_sample_batch_size / float(n_rendering_samples))
                )
                train_dataset.update_num_rays(num_rays)

            # compute loss
            loss = F.smooth_l1_loss(rgb, pixels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                psnr_ = metrics.psnr(rgb, pixels)
            writer.add_scalar('train/loss', loss, step)
            writer.add_scalar('train/psnr', psnr_, step)
            writer.add_scalar('lr', get_learning_rate(optimizer), step)

            if step % 1000 == 0:
                elapsed_time = time.time() - tic
                loss = F.mse_loss(rgb, pixels)
                print(
                    f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                    f"loss={loss:.5f} | "
                    f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | psnr={psnr_:.2f}"
                )


            val_freq = 5000
            if step > 0 and step % val_freq == 0:
                # evaluation
                radiance_field.eval()
                estimator.eval()

                psnrs = []
                n_ims_to_eval = min(5, len(test_dataset))
                with torch.no_grad():
                    for i in tqdm.tqdm(range(n_ims_to_eval)):

                        data = test_dataset[i]
                        render_bkgd = data["color_bkgd"]
                        rays = data["rays"]
                        pixels = data["pixels"]

                        # rendering
                        rgb, acc, depth, _ = render_image_with_occgrid(
                            radiance_field,
                            estimator,
                            rays,
                            # rendering options
                            near_plane=near_plane,
                            render_step_size=render_step_size,
                            render_bkgd=render_bkgd,
                            # test options
                            test_chunk_size=args.test_chunk_size,
                        )
                        mse = F.mse_loss(rgb, pixels)
                        psnr = -10.0 * torch.log(mse) / np.log(10.0)
                        psnrs.append(psnr.item())

                        if i == 0:
                            load_ims_to_tensorboard(writer, f"train_{i:d}/gt_pred_depth", [rgb, acc, depth], step)
                        if i == 1:
                            load_ims_to_tensorboard(writer, f"val_{i:d}/gt_pred_depth", [rgb, acc, depth], step)


                        # imageio.imwrite(
                        #     "acc_binary_test.png",
                        #     ((acc > 0).float().cpu().numpy() * 255).astype(np.uint8),
                        # )
                        # imageio.imwrite(
                        #     "rgb_test.png",
                        #     (rgb.cpu().numpy() * 255).astype(np.uint8),
                        # )
                        # break
                psnr_avg = sum(psnrs) / len(psnrs)
                print(f"evaluation: psnr_avg={psnr_avg}")
                train_dataset.training = True


            if step == max_steps:
                print("training stops")
                exit()

            step += 1
