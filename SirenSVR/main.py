import argparse
import glob
import os
import torch
import numpy as np
import pandas as pd
import random
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from dataprocessing.data_processor import DataProcessor
from models.inr_rec import INR_Rec
from training import fit_model
from utils.utils import loss_functions, load_model, load_checkpoint
import json
import time
parser = argparse.ArgumentParser(description='SirenSVR')
# ------------------------------ Task setup ------------------------------
parser.add_argument('--rec_grad', default=False, type=bool, help='Reconstruct gradient and laplacian, too. If false, reconstruct image only.')
parser.add_argument('--slice_thickness', default=None, type=int, help='Specify slice thickness. If none, spacing of last dimension is used.')
parser.add_argument('--load_meta_learned_model', default="", type=str, help='...')
parser.add_argument('--intensity_preproc', default='normalization', type=str, help='...')
parser.add_argument('--bias_field_correction', default=False, type=bool, help='apply bias field correction')
parser.add_argument('--denoise', default='none', type=str, help='Apply denoising for pre-processing: rician, gaussian, none')
parser.add_argument('--stack_pre_align', default='global', type=str, help='Pre-align stack to reference: Options are: "individual", "global", "none"')
parser.add_argument('--affine_stack2atlas', default=None, help='...')
parser.add_argument('--rec_post_processing', default='none', type=str, help='Align reconstruction with provided reference. Options are: "none", "mask", "reg", "mask+reg"')
parser.add_argument('--freeze_rec_epochs', default=0, type=int, help='num epochs to freeze reconstruction')
parser.add_argument('--unfreeze_patience', default=0, type=int, help='num epochs to freeze reconstruction')
parser.add_argument('--sample_halo_for_num_epochs', default=1000, type=int, help='num epochs a halo of background voxels is sampled around the brain')
# ----------------------------- Network setup -----------------------------
parser.add_argument('--hash_grid', default=False, type=bool, help='hash grid for SR instead of SIREN')
parser.add_argument('--normalize_siren', default=False, type=bool, help='apply layer norm to siren')
parser.add_argument('--siren_motion_correction', default=True, type=bool, help='use sine activation for motion correction mlp')
parser.add_argument('--sr_hidden_size', default=330, type=int, help='size of hidden layers of sr model')
parser.add_argument('--sr_num_layers', default=6, type=int, help='number hidden layers of model')
parser.add_argument('--tf_hidden_size', default=64, type=int, help='size of hidden layers of transformation model')
parser.add_argument('--tf_num_layers', default=2, type=int, help='number hidden layers of transformation model')
parser.add_argument('--slice_emb_dim', default=2, type=int, help='dim of unproj. slice embedding([slice_id, stack_id])')
parser.add_argument('--psf_k_size', default=0, type=int, help='...')
parser.add_argument('--psf_k_size_inf', default=0, type=int, help='...')
parser.add_argument('--psf_k_cap', default=128, type=int, help='max number of psf samples drawn')
parser.add_argument('--slice_weighting', default=True, type=bool, help='slice wise weighting for outlier rejection')
parser.add_argument('--voxel_weighting', default=0, type=int, help='voxel wise weighting for outlier rejection')
parser.add_argument('--slice_scaling', default=True, type=bool, help='voxel wise weighting for outlier rejection')
parser.add_argument('--tf_parameter_learning', default=False, type=bool, help='use learnable parameters for transformation model instead of siren')
# ----------------------------- Training setup
parser.add_argument('--device', default='cuda:0', type=str, help='...')
parser.add_argument('--amp', default=True, type=bool, help='...')
parser.add_argument('--max_epochs', default=6000, type=int, help='max number of training epochs')
parser.add_argument('--val_every', default=6000, type=int, help='val training by sampling entire img every x epochs')
parser.add_argument('--n_samples_total', default=40000, type=int, help='total amount of voxels/coordinates to sample per batch. This is graduially reduced to a minimum of 4000 while the PSF kernel size increases. ')
parser.add_argument('--loss_metric_sr', default="L1", type=str, help='loss metric for reconstruction')
parser.add_argument('--optim_lr_sr', default=3e-5, type=float, help='optimization learning rate SR model')
parser.add_argument('--optim_lr_tf', default=5e-4, type=float, help='optimization learning rate MC model')
parser.add_argument('--tf_reg_weight', default=0.0, type=float, help='regularization weight for rigid transforms')
# parser.add_argument('--lrschedule', default='cosine_anneal', type=str, help='type of learning rate scheduler')
# parser.add_argument('--eta_min', default=1e-5, type=float, help='min lr for cosine annealing')
parser.add_argument('--lrschedule', default='StepLR', type=str, help='type of learning rate scheduler')
parser.add_argument('--lr_gamma', default=0.85, type=float, help='min stepsize for lr scheduler')
# parser.add_argument('--lrschedule', default='none', type=str, help='type of learning rate scheduler')
parser.add_argument('--psf_scheduler', default='quadratic', type=str, help='type of psf size scheduler, "quadratic", "linear", or "none"')
parser.add_argument('--inf_res', default=0.8, type=float, help='isotropic resolution of inference')
# ----------------------------- Data setup -----------------------------
parser.add_argument('--path_tmp', default='/users/MetaSVR/tmp/', type=str, help='path to tmp directory for pre-processing and stack-alignment via flirt')
parser.add_argument('--path_flirt', default='/path/to/flirt', type=str, help='path to flirt used for stack alignment')
parser.add_argument('--path_stack_img', default=[''], type=str, action='append', help='paths to img stack files')
parser.add_argument('--path_stack_mask', default=None, type=str, action='append', help='paths to img stack files')
parser.add_argument('--path_img_ref', default='', type=str, help='path to reference img file')
parser.add_argument('--path_save', default='', type=str, help='path to save results')
parser.add_argument('--path_save_model', default='', type=str, help='path to save results')
parser.add_argument('--load_meta_learned_model', default='', type=str, help='path to meta-learned model')


def main():
    # set paths to input stacks
    # define reference image (e.g. atlas)
    # define pre-processing steps (bias-field recommended, denoising not recommended)
    # define stack alignment (global, individual, none) global recommended, individual recommended for large displacements between stacks
    # define reconstruction post-processing (none, mask, reg, mask+reg) mask+reg recommended

    args = parser.parse_args()
    data_processor = DataProcessor(args)
    loss_fns = loss_functions(args)
    model = INR_Rec(args, data_processor.get_bbox(), *data_processor.get_psf_stds()).to(args.device)
    params = [{"name": "network_sr",
               "params": model.sr_net.parameters(),
               "weight_decay": 0,
               "lr": args.optim_lr_sr},
              {"name": "network_tf",
               "params": model.tf_net.parameters(),
               "weight_decay": 0,
               "lr": args.optim_lr_tf}]
    if args.tf_parameter_learning:
        params.append({"name": "tf_parameters",
                       "params": data_processor.slice_tf_params,
                       "lr": 5e-3})
    if args.slice_weighting:
        params.append({"name": "slice_lat_vecs",
                       "params": data_processor.slice_weights,
                       "lr": 1e-3})
    if args.voxel_weighting:
        params.append({"name": "voxel_weights",
                       "params": data_processor.get_all_voxel_weights(),
                       "lr": 1e-3})
    if args.slice_scaling:
        params.append({"name": "slice_scalings",
                       "params": data_processor.slice_scalings,
                       "lr": 1e-3})
    optimizer = torch.optim.Adam(params=params)
    if args.load_meta_learned_model:
        load_checkpoint(args, model, filename=args.load_meta_learned_model, sr_prior=True, tf_prior=True)

    if args.lrschedule == 'cosine_anneal':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.eta_min, verbose=False)
    elif args.lrschedule == 'StepLR':
        scheduler = StepLR(optimizer, step_size=1000, gamma=args.lr_gamma)
    else:
        scheduler = None

    fit_model(args, model, optimizer, loss_fns, data_processor, scheduler=scheduler, start_epoch=0)


if __name__ == "__main__":
    main()
