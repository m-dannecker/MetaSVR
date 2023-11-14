import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from utils.utils import *
from utils.transform_utils import generate_img_grid
from models.inr_rec import *
from training import fit_model
import copy
import torch
import random
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import numpy as np
import ants
import os
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


def run_reptile_meta_learning(args, meta_model, meta_optim, loss_fns, dps_train, dps_val,
                              scheduler_meta, file_dicts_train, file_dicts_val, sr_only=False, start_epoch=0):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    metric_tracker = MetricTracker(save_dir=args.path_save_model) if args.save_metrics else None
    dp_train_idcs = list(range(len(dps_train)))
    for epoch in range(start_epoch, args.meta_epochs):
        metrics_dict_train = []
        metrics_dict_val = []
        print("------------  Starting with META EPOCH %d ------------" % epoch)
        if not epoch % args.val_meta_every:
            print("\n------------------------  VALIDATION  ------------------------")
            max_epochs_save = args.max_epochs
            psf_save = args.psf_k_size
            args.max_epochs = args.val_max_epochs
            args.val_every = args.max_epochs
            args.psf_k_size = args.val_psf_k_size
            for j, dp_val in enumerate(dps_val):
                inner_model, inner_optim, inner_scheduler = init_inner_model(args, meta_model, dp_val,
                                                                             file_dicts_val[j])
                print("Subject: ", args.rec_file_name)
                metrics_dict_val.append(fit_model(args, inner_model, inner_optim, loss_fns, dp_val, inner_scheduler, meta_epoch=epoch)[0])
            args.max_epochs = max_epochs_save
            args.val_every = args.max_epochs
            args.psf_k_size = psf_save
            if metric_tracker is not None:
                mean_metrics_val = {"PSNR": np.mean([m["PSNR"] for m in metrics_dict_val]),
                                    "SSIM": np.mean([m["SSIM"] for m in metrics_dict_val]),
                                    "MSE": np.mean([m["MSE"] for m in metrics_dict_val])}
                metric_tracker.update(mean_metrics_val, epoch, "val")
                metric_tracker.plot_all_metrics_of_all_sets(save_plot=True)
            save_meta_learned_model(args, meta_model, meta_optim, epoch, metric_tracker)

        random.shuffle(dp_train_idcs)
        for i, idc in enumerate(dp_train_idcs):
            print("Subject %d/%d" % (i + 1, len(dp_train_idcs)))
            inner_model, inner_optim, inner_scheduler = init_inner_model(args, meta_model, dps_train[idc],
                                                                         file_dicts_train[idc])
            metrics_dict_train.append(fit_model(args, inner_model, inner_optim, loss_fns, dps_train[idc], inner_scheduler, meta_epoch=1)[0])
            with torch.no_grad():
                for meta_param, inner_param in zip(meta_model.sr_net.parameters(), inner_model.sr_net.parameters()):
                    meta_param.grad = meta_param - inner_param
                if not sr_only:
                    for meta_param, inner_param in zip(meta_model.tf_net.parameters(), inner_model.tf_net.parameters()):
                        meta_param.grad = meta_param - inner_param
            meta_optim.step()
            meta_optim.zero_grad()
        if scheduler_meta is not None:
            scheduler_meta.step()

        if metric_tracker is not None:
            # mean_metrics_train = {"PSNR": 0,
            #                       "SSIM": 0,
            #                       "MSE": 0}
            # metric_tracker.update(mean_metrics_train, epoch, "train")
            mean_metrics_train = {"PSNR": np.mean([m["PSNR"] for m in metrics_dict_train]),
                                  "SSIM": np.mean([m["SSIM"] for m in metrics_dict_train]),
                                  "MSE": np.mean([m["MSE"] for m in metrics_dict_train])}
            metric_tracker.update(mean_metrics_train, epoch, "train")


def init_inner_model(args, meta_model, data_processor, file_dict):
    args.path_img_ref = file_dict["path_img_ref"]
    args.path_stack_img = file_dict["path_stack_img"]
    args.rec_file_name = file_dict["rec_file_name"]

    inner_model = INR_Rec(args, data_processor.get_bbox(), *data_processor.get_psf_stds()).to(args.device)
    inner_model.sr_net.load_state_dict(copy.deepcopy(meta_model.sr_net.state_dict()))
    inner_model.tf_net.load_state_dict(copy.deepcopy(meta_model.tf_net.state_dict()))
    params_inner_model = [{"name": "network_sr",
                           "params": inner_model.sr_net.parameters(),
                           "weight_decay": 0,
                           "lr": args.optim_lr_sr},
                          {"name": "network_tf",
                           "params": inner_model.tf_net.parameters(),
                           "weight_decay": 0,
                           "lr": args.optim_lr_tf}]
    if args.slice_weighting:
        params_inner_model.append({"name": "slice_weights",
                       "params": data_processor.slice_weights,
                       "lr": 1e-3})
    if args.voxel_weighting:
        params_inner_model.append({"name": "voxel_weights",
                       "params": data_processor.get_all_voxel_weights(),
                       "lr": 1e-3})
    if args.slice_scaling:
        params_inner_model.append({"name": "slice_scalings",
                       "params": data_processor.slice_scalings,
                       "lr": 1e-3})
    inner_optim = torch.optim.Adam(params=params_inner_model)
    inner_scheduler = CosineAnnealingLR(inner_optim, T_max=args.max_epochs, eta_min=1e-5, verbose=False)
    inner_scheduler = None
    return inner_model, inner_optim, inner_scheduler


def load_checkpoint(args, model, optimizer=None, filename=""):
    path_checkpoint = os.path.join(args.path_save_model, filename)
    if os.path.isfile(path_checkpoint):
        print("=> loading checkpoint '{}'".format(path_checkpoint))
        checkpoint = torch.load(path_checkpoint)
        start_epoch = checkpoint['epoch']
        model.sr_net.load_state_dict(checkpoint['sr_state_dict'])
        best_mse = checkpoint['best_mse']
        best_ssim = checkpoint['best_ssim']
        best_psnr = checkpoint['best_psnr']
        if optimizer is not None:
            new_optim = torch.optim.Adam(model.sr_net.parameters(), lr=1e-4)
            new_optim.load_state_dict(checkpoint['optimizer'])
            optimizer.param_groups[0] = new_optim.param_groups[0]
        print("=> loaded checkpoint '{}' (epoch {}, best_mse {}, best_ssim {}, best_psnr {})"
              .format(path_checkpoint, checkpoint['epoch'], checkpoint['best_mse'], checkpoint['best_ssim'], checkpoint['best_psnr']))
