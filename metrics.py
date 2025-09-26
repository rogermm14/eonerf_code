"""
This script defines the evaluation metrics and the loss functions
"""

import torch
import torch.nn.functional as F
import numpy as np

def update_loss_with_aux_term(loss, loss_dict, aux_loss, aux_dict, epoch, start_epoch=0, end_epoch=torch.inf):
    # the auxiliary term will only contribute between start_epoch and end_epoch
    if (epoch >= start_epoch) and (epoch < end_epoch):
        loss += aux_loss
    for k in aux_dict.keys():
        loss_dict[k] = aux_dict[k]
    return loss, loss_dict

def uncertainty_aware_loss(gt_rgb, pred_rgb, pred_beta):
    color_term = ((pred_rgb - gt_rgb) ** 2 / (2 * pred_beta ** 2)).mean()
    beta_term = (3 + torch.log(pred_beta).mean()) / 2  # +3 to make c_b positive since beta_min = 0.05
    loss = color_term + beta_term
    loss_dict = {'loss': loss, 'coarse_color': color_term, 'coarse_logbeta': beta_term}
    return loss, loss_dict

def depth_loss_L2(gt_depth, pred_depth, gt_conf=None, w=100):
    valid_vals = gt_depth >= 0
    if gt_conf is not None:
        valid_vals = valid_vals & (gt_conf >= 4)
    depth_l2_term = ((pred_depth[valid_vals] - gt_depth[valid_vals]) ** 2).mean()
    depth_l2_term *= w
    loss_dict = {'depth_l2': depth_l2_term, "depth_weight": w}
    return depth_l2_term, loss_dict

def differentiable_thresholding(x, thr=0.5):
    return torch.sigmoid(100*(x - thr))

def shadow_loss_L2(smask, geo_shadows, epoch=None):
    # smask --> prior shadow masks
    # geo_shadows ----> the shadow masks learned with EO-NeRF

    vals_to_penalize_old = (geo_shadows > 0.2) & (smask < 0.5) 
    vals_to_penalize_percent = sum(vals_to_penalize_old)/sum(torch.ones_like(smask))
    # this is an orientative measure for debugging
    # (amount of pixels that are shadows in the prior masks but not in the rendered shadows)

    #diff_where_noshadows = (smask > 0.5) * (geo_shadows - smask) ** 2
    #mean_diff_where_noshadows = torch.sum(diff_where_noshadows) / (torch.sum(smask > 0.5) + 1e-6)
    diff_where_shadows = (smask <= 0.5) * (geo_shadows - smask) ** 2
    mean_diff_where_shadows = torch.sum(diff_where_shadows) / (torch.sum(smask <= 0.5) + 1e-6)
    #shadows_term1 = mean_diff_where_shadows + 0.5*mean_diff_where_noshadows
    percentage_of_gt_shadows = torch.sum(smask <= 0.5)  / torch.sum(smask >= 0)
    shadows_term1 = percentage_of_gt_shadows * mean_diff_where_shadows

    # alternative version using cross_entropy
    # shadows_term1 = percentage_of_gt_shadows * F.binary_cross_entropy(geo_shadows, smask)

    loss_dict = {'shadows_term1': shadows_term1, 'shadow_vals_to_penalize': vals_to_penalize_percent}

    return shadows_term1, loss_dict

def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value

def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

