"""
This script defines the input parameters that can be customized from the command line
"""

import argparse
import datetime
import json
import os

def get_opts():
    parser = argparse.ArgumentParser()

    # input paths
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of the input dataset')
    parser.add_argument('--img_dir', type=str, default=None,
                        help='Directory where the images are located (if different than root_dir)')
    parser.add_argument("--ckpts_dir", type=str, default="ckpts",
                        help="output directory to save trained models")
    parser.add_argument("--logs_dir", type=str, default="logs",
                        help="output directory to save experiment logs")
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='directory where the ground truth DSM is located (if available)')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='directory where cache for the current dataset is found')
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="pretrained checkpoint path to load")

    # other basic stuff and dataset options
    parser.add_argument("--exp_name", type=str, default=None,
                        help="experiment name")
    parser.add_argument('--data', type=str, default='sat', choices=['sat', 'blender'],
                        help='type of dataset')
    parser.add_argument("--model", type=str, default="sat-nerf", choices=['nerf', 's-nerf', 'sat-nerf', 'eo-nerf'],
                        help="which NeRF to use")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU that will be used")

    # training and network configuration
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size (number of input rays per iteration)')
    parser.add_argument('--img_downscale', type=float, default=1.0,
                        help='downscale factor for the input images')
    parser.add_argument('--max_train_steps', type=int, default=300000,
                        help='number of training iterations')
    parser.add_argument('--save_every_n_epochs', type=int, default=4,
                        help="save checkpoints and debug files every n epochs")
    parser.add_argument('--fc_units', type=int, default=512,
                        help='number of fully connected units in the main block of layers')
    parser.add_argument('--fc_layers', type=int, default=8,
                        help='number of fully connected layers in the main block of layers')
    parser.add_argument('--n_samples', type=int, default=128,
                        help='number of coarse scale discrete points per input ray')
    parser.add_argument('--n_importance', type=int, default=0,
                        help='number of fine scale discrete points per input ray')
    parser.add_argument('--noise_std', type=float, default=0.0,
                        help='standard deviation of noise added to sigma to regularize')
    parser.add_argument('--chunk', type=int, default=1024,
                        help='maximum number of rays that can be processed at once without memory issues')

    # other sat-nerf specific stuff
    parser.add_argument('--sc_lambda', type=float, default=0.,
                        help='float that multiplies the solar correction auxiliary loss')
    parser.add_argument('--ds_lambda', type=float, default=0.,
                        help='float that multiplies the depth supervision auxiliary loss')
    parser.add_argument('--ds_drop', type=float, default=0.25,
                        help='portion of training steps at which the depth supervision loss will be dropped')
    parser.add_argument('--ds_noweights', action='store_true',
                        help='do not use reprojection errors to weight depth supervision loss')
    parser.add_argument('--first_beta_epoch', type=int, default=2,
                        help='portion of training steps at which the depth supervision loss will be dropped')
    parser.add_argument('--t_embbeding_tau', type=int, default=4,
                        help='portion of training steps at which the depth supervision loss will be dropped')
    parser.add_argument('--t_embbeding_vocab', type=int, default=30,
                        help='portion of training steps at which the depth supervision loss will be dropped')
    parser.add_argument('--geometric_shadows', action='store_true',
                        help='use geometry to represent shadows')
    parser.add_argument('--rpc_correction', action='store_true',
                        help='run internal bundle adjustment of RPC camera models using offset correction')
    parser.add_argument('--radiometric_normalization', action='store_true',
                        help='run internal radiometric normalization')
    parser.add_argument('--ecef', action='store_true',
                        help='use ecef as in the first satnerf prototype')
    parser.add_argument('--n_grid', type=int, default=128,
                        help='N voxels per dimension for the occupancy grid (N*N*N in total)')
    parser.add_argument('--init_dsm_path', type=str, default=None,
                        help="Path to external .tif DSM for initializing the NeRF")
    parser.add_argument('--init_conf_path', type=str, default=None,
                        help="Path to external .tif SGM confidence associated with the DSM in init_dsm_path")
    parser.add_argument('--shadow_masks_dir', type=str, default=None,
                        help="Path to a directory containing a predefined shadow mask for each image in the dataset")
    parser.add_argument('--subset_Nviews', type=int, help='use N views instead of all views', default=None)

    args = parser.parse_args()

    if args.model == "eo-nerf":
        args.radiometric_normalization = True

    exp_id = args.model if args.exp_name is None else args.exp_name
    #args.exp_name = "{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), exp_id)
    print("\nRunning {} - Using gpu {}\n".format(args.exp_name, args.gpu_id))

    os.makedirs("{}/{}".format(args.logs_dir, args.exp_name), exist_ok=True)
    with open("{}/{}/opts.json".format(args.logs_dir, args.exp_name), "w") as f:
        json.dump(vars(args), f, indent=2)

    return args
