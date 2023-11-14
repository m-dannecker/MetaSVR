import argparse
import os
import torch
import numpy as np
import random
from models.inr_rec import INR_Rec, INR_SR
from dataprocessing.data_processor import DataProcessor
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from meta_learning.train_reptile import fit_reptile_model, fit_hypomodel, run_reptile_meta_learning
from utils.utils import loss_functions, load_checkpoint
import copy
import glob
import pandas as pd

torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description='INR4REC_Reptile')
# ------------------------------ Task setup ------------------------------
parser.add_argument('--save_metrics', default=True, type=bool, help='...')
parser.add_argument('--train_prior', default=False, type=bool, help='...')
parser.add_argument('--rec_grad', default=False, type=bool, help='...')
parser.add_argument('--slice_thickness', default=None, type=int, help='...')
parser.add_argument('--load_prior', default=False, type=bool, help='...')
parser.add_argument('--load_meta_learned_model', default="", type=str, help='...')
parser.add_argument('--intensity_preproc', default='normalization', type=str, help='...')
parser.add_argument('--bias_field_correction', default=False, type=bool, help='apply bias field correction')
parser.add_argument('--denoise', default='none', type=str, help='rician, gaussian, none')
parser.add_argument('--stack_pre_align', default='none', type=str, help='"individual", "global", "none"')
parser.add_argument('--affine_stack2atlas', default=None, help='...')
parser.add_argument('--rec_post_processing', default='mask', type=str, help='"none", "mask", "reg", "mask+reg"')
parser.add_argument('--freeze_rec_epochs', default=0, type=int, help='num epochs to freeze reconstruction')
parser.add_argument('--unfreeze_patience', default=0, type=int, help='num epochs to freeze reconstruction')
parser.add_argument('--sample_halo_for_num_epochs', default=2000, type=int, help='num epochs a halo of background '
                                                                                 'is sampled around the brain as guidance for better motion correction')
# ----------------------------- Network setup -----------------------------
parser.add_argument('--hash_grid', default=False, type=bool, help='hash grid for SR instead of SIREN')
parser.add_argument('--normalize_siren', default=False, type=bool, help='apply layer norm to siren')
parser.add_argument('--siren_motion_correction', default=True, type=bool, help='use sine activation for motion correction mlp')
parser.add_argument('--sr_hidden_size', default=330, type=int, help='size of hidden layers of sr model')
parser.add_argument('--sr_num_layers', default=6, type=int, help='number hidden layers of model')
parser.add_argument('--tf_hidden_size', default=64, type=int, help='size of hidden layers of transformation model')
parser.add_argument('--tf_num_layers', default=2, type=int, help='number hidden layers of transformation model')
parser.add_argument('--slice_emb_dim', default=2, type=int, help='dim of unproj. slice embedding([slice_id, stack_id])')
parser.add_argument('--psf_k_size', default=32, type=int, help='...')
parser.add_argument('--psf_k_size_inf', default=0, type=int, help='...')
parser.add_argument('--val_psf_k_size', default=32, type=int, help='...')
parser.add_argument('--psf_k_cap', default=32, type=int, help='...')
parser.add_argument('--psf_learnable', default=False, type=int, help='...')
parser.add_argument('--slice_weighting', default=False, type=bool, help='slice wise weighting for outlier rejection')
parser.add_argument('--voxel_weighting', default=0, type=int, help='voxel wise weighting for outlier rejection')
parser.add_argument('--slice_scaling', default=False, type=bool, help='voxel wise weighting for outlier rejection')
parser.add_argument('--tf_parameter_learning', default=False, type=bool, help='use learnable parameters for transformation model instead of siren')
# ----------------------------- Training setup
parser.add_argument('--device', default='cuda:0', type=str, help='...')
parser.add_argument('--amp', default=True, type=bool, help='...')
parser.add_argument('--max_epochs', default=1000, type=int, help='max number of training epochs')
parser.add_argument('--val_max_epochs', default=1000, type=int, help='max number of training epochs')
parser.add_argument('--val_every', default=1000, type=int, help='val training by sampling entire img every x epochs')
parser.add_argument('--meta_epochs', default=50, type=int, help='max number of training epochs')
parser.add_argument('--val_meta_every', default=1, type=int, help='val training by sampling entire img every x epochs')
parser.add_argument('--n_samples_total', default=12000, type=int, help='total amount of voxels/coordinates to sample')
parser.add_argument('--loss_metric_sr', default="L1", type=str, help='loss metric for reconstruction')
parser.add_argument('--optim_lr_sr', default=3e-5, type=float, help='optimization learning rate') # 5e-4 hash grid
parser.add_argument('--optim_lr_tf', default=5e-4, type=float, help='optimization learning rate')
parser.add_argument('--optim_lr_meta', default=1e-4, type=float, help='optimization learning rate')
parser.add_argument('--tf_reg_weight', default=0.0, type=float, help='regularization weight for rigid transforms')
parser.add_argument('--lrschedule', default='cosine_anneal', type=str, help='type of learning rate scheduler')
# parser.add_argument('--lrschedule', default='StepLR', type=str, help='type of learning rate scheduler')
parser.add_argument('--psf_scheduler', default='quadratic', type=str, help='type of learning rate scheduler')
# parser.add_argument('--lrschedule', default='none', type=str, help='type of learning rate scheduler')
parser.add_argument('--eta_min', default=1e-5, type=float, help='min lr for cosine annealing')
# ----------------------------- Data setup -----------------------------
parser.add_argument('--path_tmp', default='/users/MetaSVR/tmp/', type=str, help='path to tmp directory for pre-processing and stack-alignment via flirt')
parser.add_argument('--path_flirt', default='/path/to/flirt', type=str, help='path to flirt used for stack alignment')
parser.add_argument('--path_atlas_img', default='', type=str, help='path to atlas image file')
parser.add_argument('--path_stack_img', default=[''], type=str, action='append', help='paths to img stack files')
parser.add_argument('--path_stack_mask', default=None, type=str, action='append', help='paths to img stack files')
parser.add_argument('--path_img_ref', default='', type=str, help='path to reference img file')
parser.add_argument('--path_save', default='', type=str, help='path to save results')
parser.add_argument('--path_save_model', default='', type=str, help='path to save results')


def main():
    args = parser.parse_args()
    setup_dict = setup_reptile_training(args, dataset="dHCP_fetal")
    run_reptile_meta_learning(args, setup_dict["meta_model"], setup_dict["meta_optim"], setup_dict["loss_fns"],
                              setup_dict["dps_train"], setup_dict["dps_val"], setup_dict["meta_scheduler"],
                              setup_dict["file_dicts_train"], setup_dict["file_dicts_val"],
                              sr_only=False, start_epoch=0)

def setup_reptile_training(args, dataset):
    setup_dict = {"meta_model": INR_Rec(args, bbox=None, psf_stds=None, res=None).to(args.device)}
    params_meta_model = [{"name": "network_sr",
                          "params": setup_dict["meta_model"].sr_net.parameters(),
                          "weight_decay": 0,
                          "lr": args.optim_lr_meta},
                         {"name": "network_tf",
                          "params": setup_dict["meta_model"].tf_net.parameters(),
                          "weight_decay": 0,
                          "lr": args.optim_lr_meta}]
    setup_dict["meta_optim"] = torch.optim.Adam(params=params_meta_model)
    if args.lrschedule == 'cosine_anneal':
        setup_dict["meta_scheduler"] = CosineAnnealingLR(setup_dict["meta_optim"], T_max=50*args.meta_epochs, eta_min=args.eta_min, verbose=False)
    elif args.lrschedule == 'StepLR':
        setup_dict["meta_scheduler"] = StepLR(setup_dict["meta_optim"], step_size=1, gamma=0.90)
    else:
        setup_dict["meta_scheduler"] = None
    setup_dict["meta_scheduler"] = None

    setup_dict["loss_fns"] = loss_functions(args)

    dps_train, dps_val, file_dicts_train, file_dicts_val = load_dataprocessors(args, dataset, slice_thickness, motion)

    setup_dict["dps_train"] = dps_train
    setup_dict["dps_val"] = dps_val
    setup_dict["file_dicts_train"] = file_dicts_train
    setup_dict["file_dicts_val"] = file_dicts_val
    return setup_dict


def load_dataprocessors(args, dataset):
    dataprocessors_train = []
    dataprocessors_val = []
    file_dicts_train = []
    file_dicts_val = []
    sub_ids = get_sub_ids(dataset)
    sub_ids_train = sub_ids[0]
    sub_ids_val = sub_ids[1]
    for sub_id in sub_ids_train:
        file_dict_train = setup_data(args, sub_id)
        file_dicts_train.append(file_dict_train)
        args.affine_stack2atlas = None
        data_processor = DataProcessor(args)
        dataprocessors_train.append(data_processor)

    for sub_id in sub_ids_val:
        file_dict_val = setup_data(args, sub_id)
        file_dicts_val.append(file_dict_val)
        args.affine_stack2atlas = None
        data_processor = DataProcessor(args)
        dataprocessors_val.append(data_processor)
    return dataprocessors_train, dataprocessors_val, file_dicts_train, file_dicts_val


def get_sub_ids(dataset):
    path = "path_to_data"
    sub_ids_train = glob.glob(os.path.join(path, "train", "data"))
    sub_ids_val = glob.glob(os.path.join(path, "val", "data"))
    sub_ids_train = [sub_id.split("/img_")[-1][:-7] for sub_id in sub_ids_train]
    sub_ids_val = [sub_id.split("/img_")[-1][:-7] for sub_id in sub_ids_val]
    sub_ids_train.sort()
    sub_ids_val.sort()
    return sub_ids_train, sub_ids_val


def setup_data(args, sub_id):
    args.stack_ids = [0, 1, 2]
    args.inf_res = torch.Tensor([.8, .8, .8]).cuda(0)

    input_path = "path_to_data/sub_id={}/".format(sub_id)

    file_dict = {}
    args.path_stack_img = [input_path + "stack_with_stack_id={}.nii.gz".format(s_id) for s_id in
                           args.stack_ids]
    file_dict["path_stack_img"] = args.path_stack_img

    args.rec_file_name = "outputname_reconstruction_sub_id={}.nii.gz".format(sub_id)
    file_dict["rec_file_name"] = args.rec_file_name

    args.path_img_ref = "/path_to_reference/reference_sub_id={}.nii.gz".format(sub_id)
    file_dict["path_img_ref"] = args.path_img_ref

    args.path_save_model = '/path_to_save_model/'
    args.name_meta_learned_model = "MetaModel_nl{}_sh{}_nms{}_psf={}.pt".format(args.sr_num_layers,
                                                                              args.sr_hidden_size,
                                                                              args.meta_epochs, args.psf_k_cap)
    os.makedirs(args.path_save, exist_ok=True)
    os.makedirs(args.path_save_model, exist_ok=True)
    return file_dict

if __name__ == "__main__":
    main()
