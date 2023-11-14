import copy
import json
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import os
import ants
import SimpleITK as sitk
import time
from skimage import exposure
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.measure import label
from skimage import filters
import scipy.ndimage as ndi
from scipy.ndimage import affine_transform
from dipy.align.imaffine import transform_centers_of_mass, MutualInformationMetric, AffineRegistration
from dipy.align.transforms import TranslationTransform3D, RigidTransform3D
# from monai.losses import DiceCELoss
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# from brainextractor import BrainExtractor
import cv2


class MetricTracker:
    def __init__(self, metrics_to_track=None, sets_to_track=None, save_dir=None):
        self.metrics_to_track = ["PSNR", "SSIM", "MSE"] if metrics_to_track is None else metrics_to_track
        self.sets_to_track = ["train", "val"] if sets_to_track is None else sets_to_track
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.metrics = {}

        for set_ in self.sets_to_track:
            self.metrics[set_] = {}
            for metric in self.metrics_to_track:
                self.metrics[set_][metric] = []

    def update(self, metrics_dict, epoch, set_):
        for metric in self.metrics_to_track:
            self.metrics[set_][metric].append((epoch, metrics_dict[metric]))

    def get_best_epoch(self, metric, set_):
        return max(self.metrics[set_][metric], key=lambda x: x[1])[0]

    def get_best_metric(self, metric, set_):
        if metric == "MSE":
            return min(self.metrics[set_][metric], key=lambda x: x[1])[1]
        else:
            return max(self.metrics[set_][metric], key=lambda x: x[1])[1]

    def get_last_epoch(self, metric, set_):
        return self.metrics[set_][metric][-1][0]

    def get_last_metric(self, metric, set_):
        return self.metrics[set_][metric][-1][1]

    def plot_metric_of_all_sets(self, metric, color_dict=None, save_plot=False):
        if color_dict is None:
            color_dict = {"train": "blue", "val": "red"}

        for set_ in self.sets_to_track:
            plt.plot(*zip(*self.metrics[set_][metric]), color=color_dict[set_], label=set_ + " " + metric)
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(metric)
        if save_plot:
            plt.savefig(os.path.join(self.save_dir, metric + ".png"))
        else:
            plt.show()
        plt.close()

    def plot_all_metrics_of_all_sets(self, color_dict=None, save_plot=False):
        if color_dict is None:
            color_dict = {"train": "blue", "val": "red"}

        for metric in self.metrics_to_track:
            self.plot_metric_of_all_sets(metric, color_dict, save_plot)


def save_meta_learned_model(args, model, optimizer, current_epoch, metric_tracker=None):
    current_mse=0
    current_ssim=0
    current_psnr=0
    current_epoch=current_epoch
    if metric_tracker is not None:
        current_epoch = metric_tracker.get_last_epoch("PSNR", "val")
        current_psnr = metric_tracker.get_last_metric("PSNR", "val")
        current_ssim = metric_tracker.get_last_metric("SSIM", "val")
        current_mse = metric_tracker.get_last_metric("MSE", "val")
        if current_psnr >= metric_tracker.get_best_metric("PSNR", "val"):
            model_name = args.name_meta_learned_model.replace(".pt", "_BestPSNR.pt")
            save_checkpoint(args, model, optimizer, current_epoch, model_name, current_psnr, current_ssim, current_mse)
        if current_ssim >= metric_tracker.get_best_metric("SSIM", "val"):
            model_name = args.name_meta_learned_model.replace(".pt", "_BestSSIM.pt")
            save_checkpoint(args, model, optimizer, current_epoch, model_name, current_psnr, current_ssim, current_mse)
        if current_mse <= metric_tracker.get_best_metric("MSE", "val"):
            model_name = args.name_meta_learned_model.replace(".pt", "_BestMSE.pt")
            save_checkpoint(args, model, optimizer, current_epoch, model_name, current_psnr, current_ssim, current_mse)
    model_name = args.name_meta_learned_model.replace(".pt", "_Latest.pt")
    save_checkpoint(args, model, optimizer, current_epoch, model_name, current_mse, current_ssim, current_psnr)


class CESoftmaxLoss(nn.Module):
    def __init__(self, ce_loss):
        super().__init__()
        self.ce = ce_loss
    def forward(self, v_p, v):
        v = v.permute(0, 2, 1).squeeze(1).long()
        v_p = v_p.softmax(dim=-1).permute(0, 2, 1)
        loss = self.ce(v_p, v).unsqueeze(-1)
        return loss


def loss_functions(args):
    reduction = 'none' if args.slice_weighting or args.voxel_weighting else 'mean'
    loss_fns = {"MSE": torch.nn.MSELoss(reduction=reduction),
              "L1": torch.nn.L1Loss(reduction=reduction),
              "CE": CESoftmaxLoss(torch.nn.CrossEntropyLoss(reduction=reduction)),
              # "DiceCE": DiceCELoss(include_background=True,
              #                      to_onehot_y=args.num_seg_classes,
              #                      softmax=True,)
                                     }
    loss_fns = {'sr': loss_fns[args.loss_metric_sr]}
    return loss_fns


def freeze_sr(args, model, data_processor, freeze_sr_epochs, epoch, loss_sr, prev_loss_sr):
    if epoch == 0:
        for param in model.sr_net.parameters():
            param.requires_grad = False
        if args.slice_weighting:
            data_processor.slice_weights.requires_grad = False
        if args.voxel_weighting:
            voxel_weights = data_processor.get_all_voxel_weights()
            for vw in voxel_weights:
                vw.requires_grad = False

    if freeze_sr_epochs == 0 and args.unfreeze_patience > 0:
        if 1.0 * prev_loss_sr <= loss_sr:
            args.unfreeze_patience -= 1
        prev_loss_sr = loss_sr
    elif epoch == freeze_sr_epochs or (args.unfreeze_patience == 0 and freeze_sr_epochs == 0):
        if args.freeze_rec_epochs > 0 or args.unfreeze_patience > 0:
            print("----------------------Unfreezing sr weights!--------------------------------")
        for param in model.sr_net.parameters():
            param.requires_grad = True
        if args.slice_weighting:
            data_processor.slice_weights.requires_grad = True
        if args.voxel_weighting:
            voxel_weights = data_processor.get_all_voxel_weights()
            for vw in voxel_weights:
                vw.requires_grad = True
        args.unfreeze_patience = -1
    return prev_loss_sr


def psf_scheduler(args, model, epoch, verbose=99999999):
    if args.psf_k_cap > 0:
        if args.psf_scheduler == "linear":
            frac = (epoch / args.max_epochs)
            frac_std = np.sqrt(epoch / args.max_epochs)
        elif args.psf_scheduler == "quadratic":
            frac = (epoch / args.max_epochs) ** 2
            frac_std = np.sqrt(epoch / args.max_epochs)
        else:  # "none"
            frac = 1.0
        # model.psf_stds = args.psf_stds * frac_std
        model.n_s_psf = int(max(args.psf_k_cap * frac, 1))
        args.n_samples = max(args.n_samples_total // (model.n_s_psf*len(args.psf_stds)), args.n_samples_total // (10*len(args.psf_stds)))
        if epoch == 0 or not (epoch+1) % verbose:
            print(f"PSF STDs: {model.psf_stds[0].tolist()},  PSF size: {model.n_s_psf},", "n_samples_total", args.n_samples*len(args.psf_stds))


def psf_flag(model, freeze_sr_epochs, epoch, use_psf_sr):
    if epoch < freeze_sr_epochs:
        model.use_psf_sr = False
    if epoch == freeze_sr_epochs:
        model.use_psf_sr = use_psf_sr


def calc_sr_loss(v_p, v, loss_fn, slice_scaling=None):
    v_p = v_p * slice_scaling if slice_scaling is not None else v_p
    return loss_fn(v_p, v)


def calc_seg_loss(v_p, v, loss_fn):
    loss = loss_fn(v_p, v) if v_p is not None else torch.Tensor([0.0]).to(v.device)
    return loss


def calc_tf_loss(embed_tf):
    loss_rot = (embed_tf[..., :3] ** 2).mean()
    loss_trans = (embed_tf[..., 3:] ** 2).mean()
    return loss_rot + 1e-3 * loss_trans


def calc_slice_weighting_loss(loss_sr, slice_weights):
    if slice_weights is not None:
        loss_sr = (loss_sr / slice_weights ** 2) + 0.5 * (slice_weights ** 2).log()
        # loss_sr = (loss_sr * slice_weights ** 2) - 0.5 * (slice_weights ** 2).log()
    return loss_sr


def calc_voxel_weighting_loss(loss_sr, voxel_weights):
    if voxel_weights is not None:
        loss_sr = (loss_sr / voxel_weights ** 2) + 0.5 * (voxel_weights ** 2).log()
    return loss_sr


def print_epoch_stats(epoch, loss, loss_sr, loss_tf, epoch_time, print_every=1):
    if epoch ==0 or not (epoch+1) % print_every:
        print(f"Iteration {epoch+1}: loss={loss:.6f}, loss_sr={loss_sr:.6f}"
              f" loss_tf_reg={loss_tf:.8f}, time={time.time()-epoch_time:.2f}s")
        epoch_time = time.time()
    return epoch_time


def get_center_cc(segmentation):
    center = np.array(segmentation.shape) // 2
    roi_size = (0.1 * np.array(segmentation.shape)).astype(int)
    labels = label(segmentation)
    assert(labels.max() != 0)  # assume at least 1 CC
    brain_label = np.median(labels[center[0] - roi_size[0]:center[0] + roi_size[0],
                            center[1] - roi_size[1]:center[1] + roi_size[1],
                            center[2] - roi_size[2]:center[2] + roi_size[2]]).astype(int)
    brainCC = labels == brain_label
    return brainCC


def post_process_reconstruction(args, values_p, affine, values_sr_p_halo=None, reg=True, values_grad_p=None, values_lp_p=None):
    values_p_nii = nib.Nifti1Image(values_p, affine)
    values_p_masked_nii = None
    mask_p_nii = None
    reg_affine = None
    values_grad_p_masked_nii = None
    values_lp_p_masked_nii = None
    if "mask" in args.rec_post_processing:
        # bet = BrainExtractor(img=values_p_nii)
        # bet.run()
        # mask = bet.compute_mask()
        # mask = (ndi.gaussian_filter(mask, sigma=2) > 0.5).astype(np.uint8)
        values_for_mask = values_p if values_sr_p_halo is None else values_sr_p_halo
        threshold = filters.threshold_otsu(values_for_mask)*0.95
        mask = values_for_mask > threshold
        mask = ndi.binary_fill_holes(mask)
        mask = get_center_cc(mask)
        mask = ndi.binary_closing(mask, iterations=9)
        mask = ndi.binary_fill_holes(mask).astype(np.uint8)

        values_p_masked = mask * values_p
        mask_p_nii = nib.Nifti1Image(mask, affine)
        values_p_masked_nii = nib.Nifti1Image(values_p_masked, affine)
        if values_grad_p is not None:
            values_grad_p_masked = mask * values_grad_p
            values_grad_p_masked_nii = nib.Nifti1Image(values_grad_p_masked, affine)
            values_lp_p_masked = mask * values_lp_p
            values_lp_p_masked_nii = nib.Nifti1Image(values_lp_p_masked, affine)

    if args.path_img_ref and "reg" in args.rec_post_processing and reg:
        img_ref = nib.load(args.path_img_ref)
        reg_affine = reg2ref(img_ref, copy.deepcopy(values_p_masked_nii)) if values_p_masked_nii is not None \
            else reg2ref(img_ref, values_p_nii)

    return values_p_nii, values_p_masked_nii, mask_p_nii, reg_affine, values_grad_p_masked_nii, values_lp_p_masked_nii


def reg2ref(target_nib, moving_nib):
    # Some (hyper)parameters of the registration
    metric = MutualInformationMetric(nbins=32, sampling_proportion=0.1)
    level_iters = [10, 5]
    sigmas = [1.0, 0.0]
    factors = [2, 1]
    # target_nib.affine[:3, :3] = np.eye(3)
    # moving_nib.affine[:3, :3] = np.eye(3)
    target_arr = target_nib.get_fdata()
    target_arr = (target_arr - target_arr.min()) / (target_arr.max() - target_arr.min())
    moving_arr = moving_nib.get_fdata()

    c_of_mass = transform_centers_of_mass(target_arr, target_nib.affine,
                                          moving_arr, moving_nib.affine)

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = TranslationTransform3D()
    params0 = None
    translation = affreg.optimize(target_arr, moving_arr, transform, params0,
                                  target_nib.affine, moving_nib.affine,
                                  starting_affine=c_of_mass.affine)

    # passing the translation as starting_affine strangely does not work, therefore we do it manually
    moving_nib.affine[:3, 3] = -translation.affine[:3, 3]
    transform = RigidTransform3D()
    params0 = None
    rigid = affreg.optimize(target_arr, moving_arr, transform, params0,
                            target_nib.affine, moving_nib.affine)

    affine_MNI_to_2D = np.linalg.inv(moving_nib.affine) @ rigid.affine @ target_nib.affine
    mat, vec = nib.affines.to_matvec(affine_MNI_to_2D)
    resampled_img = affine_transform(moving_arr, mat, vec, output_shape=target_arr.shape)
    resampled_img_nii = nib.Nifti1Image(resampled_img.astype(np.float32), target_nib.affine)

    # add the translation to the rigid transform
    rigid.affine[:3, 3] = rigid.affine[:3, 3] + translation.affine[:3, 3]

    return rigid.affine#, resampled_img_nii


def plot_convergence(metrics_dicts, path_save=None, meta_init=False):
    y_label = "PSNR"
    y_values = [round(metrics_dict["PSNR"], 2) for metrics_dict in metrics_dicts]
    x_values_epochs = [round(metrics_dict["epoch"]) for metrics_dict in metrics_dicts]
    x_values_times = [round(metrics_dict["train_time"], 0) for metrics_dict in metrics_dicts]
    x_label1 = "Iterations"
    x_label2 = "Time (s)"
    fig, ax = plt.subplots()
    ax.plot(x_values_epochs, y_values)
    ax2 = ax.twiny()
    ax2.set_xticks(x_values_times)
    ax.set_xlabel(x_label1)
    ax2.set_xlabel(x_label2)
    ax.set_ylabel(y_label)
    plt.show()
    if path_save:
        # save dictionary to file
        with open(os.path.join(path_save, "convergence_dict_meta={}.json".format(meta_init)), 'w') as fp:
            json.dump(metrics_dicts, fp)


def visualize_slice_weights(dataprocessor, stack_id=0, path_save=None):
    num_to_plot = 5
    stack_id = 0
    stack = dataprocessor.stacks[stack_id]
    slices = []
    slice_ids = []
    weights = []
    num_slices_nz = 0
    for slice_id in range(int(stack["vxl_shape"][-1])):
        slice_weight = dataprocessor.get_slice_weights_of_coords(stack, np.array([[0, 0, slice_id]]))
        if slice_weight == 1.0:
            continue
        num_slices_nz += 1
        slice_ids += [slice_id]
        slices.append(stack["stack_img"][..., slice_id].cpu().numpy())
        weights.append(slice_weight.item())

    # cut off first 20 % and last 20 % slices with little overlap of other stacks
    cut_off_start = int(num_slices_nz * 0.20)
    cut_off_end = int(num_slices_nz * 0.80)
    slices = slices[cut_off_start:cut_off_end]
    slice_ids = slice_ids[cut_off_start:cut_off_end]
    weights = weights[cut_off_start:cut_off_end]

    # order slices and weights and slice_ids by weight even if weights are not unique
    slices, slice_ids, weights = zip(*sorted(zip(slices, slice_ids, weights), key=lambda x: x[2]))
    slices = [slices[i] for i in range(0, len(slices), len(slices) // num_to_plot)]
    slices.reverse()
    slice_ids = [slice_ids[i] for i in range(0, len(slice_ids), len(slice_ids) // num_to_plot)]
    slice_ids.reverse()
    weights = 1 / np.array(weights) ** 2
    weights = np.flip(np.array([weights[i] for i in range(0, len(weights), len(weights) // num_to_plot)])
                      / weights.mean(), axis=0)  # factor of which the slice is more important than the mean
    # remove third element in slices
    slices = [slices[i] for i in range(len(slices)) if i != 2]
    slice_ids = [slice_ids[i] for i in range(len(slice_ids)) if i != 2]
    weights = np.concatenate((weights[:2], weights[3:]))
    num_to_plot = len(slices)
    for i, _ in enumerate(slices):
        # crop image to bbox of non-zero values
        slices[i] = slices[i][~np.all(slices[i] == 0, axis=1)]
        slices[i] = slices[i][:, ~np.all(slices[i] == 0, axis=0)]
    max_shape = np.max([s.shape for s in slices], axis=0)
    for i, _ in enumerate(slices):
        # pad same amount of zeros to all sides
        pad_left = (max_shape[0] - slices[i].shape[0]) // 2
        pad_right = max_shape[0] - slices[i].shape[0] - pad_left
        pad_top = (max_shape[1] - slices[i].shape[1]) // 2
        pad_bottom = max_shape[1] - slices[i].shape[1] - pad_top
        slices[i] = np.pad(slices[i], ((pad_left, pad_right), (pad_top, pad_bottom)),
                           mode='constant', constant_values=0)

    # plot all slices in one figure with weights as title
    plt.rcParams.update({'font.size': 30})
    fig, axs = plt.subplots(1, num_to_plot, figsize=(20, 7))
    for i, ax in enumerate(axs.flat):
        img = slices[i]
        ax.imshow(np.flip(img.T), cmap="gray")
        # ax.set_title("weight-factor = {}".format(np.round(weights[i], 2)))
        ax.axis('off')  # clear x- and y-axes
    plt.tight_layout()
    # TODO: try this
    plt.subplots_adjust(wspace=0.00, hspace=-0.35)
    # show as tight as possible
    ticks = np.linspace(weights[0], max(weights), num_to_plot)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(vmin=weights[0] * 0.70, vmax=max(weights) * 1.075),
                                              cmap=plt.cm.get_cmap('RdYlGn')), ax=axs,
                        location='bottom', pad=0.0, aspect=60, ticks=ticks, anchor=(0.5, 1.70))
    cbar.ax.set_xticklabels(str(round(w, 2)) for w in weights)
    cbar.set_label("Slice Weight Factor", fontsize=36)
    plt.gcf().subplots_adjust(bottom=0.25)
    if path_save is None:
        plt.show()
    else:
        plt.savefig(os.path.join(path_save, "slice_weights.png"), bbox_inches='tight')


def visualize_voxel_weights(dataprocessor, stack_id=0, slice_id=18, path_save=None):
    stack = dataprocessor.stacks[stack_id]
    slice_ids = [39, 34, 42, 36]
    imgs = []
    heatmaps = []
    voxel_weights_normed_all = []
    for slice_id in slice_ids:
        slice_img = stack["stack_img"][..., slice_id]
        coords = torch.nonzero(slice_img).float()
        # add slice_id to coords as third dimension
        coords = torch.cat((coords, torch.ones(coords.shape[0], 1).to(slice_img.device) * slice_id), dim=1)
        voxel_weights = dataprocessor.get_voxel_weights_of_coords(stack, coords, 0).detach().cpu()
        voxel_weights_normed = 1 / voxel_weights ** 2
        voxel_weights_normed = voxel_weights_normed / voxel_weights_normed.mean()  # factor of which the voxel is more important than the mean
        voxel_weights_normed_all.append(voxel_weights_normed)
        # voxel_weights_normed = (voxel_weights_normed - voxel_weights_normed.min()) / (
        #             voxel_weights_normed.max() - voxel_weights_normed.min())  # normalize to [0,1]

        heatmap = torch.zeros_like(slice_img).cpu()
        heatmap[coords[:, 0].long(), coords[:, 1].long()] = voxel_weights_normed[:, 0]
        heatmap = heatmap.detach().numpy()
        slice_img = slice_img.detach().cpu().numpy()
        # crop image to bbox of non-zero values
        slice_img = slice_img[~np.all(slice_img == 0, axis=1)]
        slice_img = slice_img[:, ~np.all(slice_img == 0, axis=0)]
        heatmap = heatmap[~np.all(heatmap == 0, axis=1)]
        heatmap = heatmap[:, ~np.all(heatmap == 0, axis=0)]
        heatmap[slice_img == 0] = np.nan
        imgs.append(slice_img)
        heatmaps.append(heatmap)
    plt.rcParams.update({'font.size': 24})
    fig, axs = plt.subplots(2, len(slice_ids), figsize=(20, 8))
    cmap = plt.cm.get_cmap('RdYlGn')
    vw_min = min([torch.min(voxel_weights_normed_all[i]).item() for i in range(len(slice_ids))])
    vw_max = max([torch.max(voxel_weights_normed_all[i]).item() for i in range(len(slice_ids))])
    for i in range(len(slice_ids)):
        axs[0, i].imshow(np.flip(imgs[i]), cmap="gray")
        axs[1, i].imshow(np.flip(imgs[i]), cmap="gray")
        heatmap_overlay = axs[1, i].imshow(np.flip(heatmaps[i]), cmap=cmap, alpha=0.4, vmin=vw_min, vmax=vw_max)
        axs[0, i].axis('off')  # clear x- and y-axes
        axs[1, i].axis('off')  # clear x- and y-axes
    axs[0, 0].text(-0.20, 0.5, "input slice", va='center', rotation='vertical', transform=axs[0, 0].transAxes,
                   color="black", fontsize=30)
    axs[1, 0].text(-0.20, 0.5, "weight map", va='center', rotation='vertical', transform=axs[1, 0].transAxes,
                   color="black", fontsize=30)
    plt.subplots_adjust(wspace=-0.20, hspace=0.025)
    cbar = plt.colorbar(heatmap_overlay, ax=axs, shrink=0.8, anchor=(-0.3, 0.5))
    cbar.set_ticks(np.linspace(vw_min, vw_max, num=3).round(2))
    cbar.set_label('weight factor', fontsize=30)
    # fig.patch.set_facecolor('black')
    plt.show()

    # show as tight as possible
    # plt.tight_layout()
    # fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(vmin=min(voxel_weights), vmax=max(voxel_weights)),
    #                                    cmap=plt.cm.jet), ax=axs).set_label("Weight factor. Higher means more important. Mean weight is 1.")
    if path_save is None:
        plt.show()
    else:
        plt.savefig(os.path.join(path_save, "voxel_weights.png"), bbox_inches='tight')


def save_head(values_p, affine, bbox_size, path_save, head_id, subid):
    values_p = values_p.view(list(bbox_size)).cpu().numpy()
    values_p = np.clip(values_p, 0, 1)
    img_nii = nib.Nifti1Image(values_p, affine)
    file_name = "head" + str(head_id) + "_sub" + str(subid) + ".nii.gz"
    nib.save(img_nii, os.path.join(path_save, file_name))
    # print(f"Volume saved to {os.path.join(path_save, file_name)}")
    return os.path.join(path_save, file_name)


def save_sub(values_p, affine, bbox_size, path_save, subid, epochs=None):
    values_p = values_p.view(list(bbox_size)).cpu().numpy()
    values_p = np.clip(values_p, 0, 1)
    img_nii = nib.Nifti1Image(values_p, affine)

    if "/" in subid:
        # get id from subid
        basename = os.path.basename(subid)
        if "sub-" in basename:
            subid = basename.split("sub-")[1]
            subid = subid.split("_T2w")[0]
        else:
            subid = basename.split("-")[0][4:]
    file_name = "sub" + str(subid) + ".nii.gz" if epochs is None else "sub" + str(subid) + "_epoch" + str(epochs) + ".nii.gz"
    nib.save(img_nii, os.path.join(path_save, file_name))
    # print(f"Volume saved to {os.path.join(path_save, file_name)}")
    return os.path.join(path_save, file_name)


def eval_and_save_segmentation(path_img_ref, path_save, stack_ids, rec_file_name=None, epoch=None, ttime=None):
    if rec_file_name is None:
        path_img_rec = os.path.join(path_save, "Stacks[" + "_".join(str(x) for x in stack_ids) + "]_rec.nii.gz")
    else:
        path_img_rec = os.path.join(path_save, rec_file_name)

    f_img = ants.image_read(path_img_ref)
    f_img = ants.from_numpy(f_img.numpy().clip(min=0, max=np.percentile(f_img.numpy(), 99.9)), origin=f_img.origin,
                            spacing=f_img.spacing, direction=f_img.direction)
    f_img = (f_img - f_img.min()) / (f_img.max() - f_img.min())
    mask = f_img > 0.00

    m_img = ants.image_read(path_img_rec)

    mytx = ants.registration(fixed=f_img, moving=m_img, mask=mask, type_of_transform='Rigid')
    m_img_wrpd = ants.apply_transforms(fixed=f_img, moving=m_img, transformlist=mytx['fwdtransforms']) * mask

    m_img_wrpd = ants.from_numpy(m_img_wrpd.numpy().clip(min=0, max=np.percentile(m_img_wrpd.numpy(), 99.9)),
                                 origin=m_img_wrpd.origin,
                                 spacing=m_img_wrpd.spacing, direction=m_img_wrpd.direction)
    m_img_wrpd = (m_img_wrpd - m_img_wrpd.min()) / (m_img_wrpd.max() - m_img_wrpd.min())
    m_img_wrpd = m_img_wrpd * mask

    f_img_np = f_img.numpy()
    m_img_wrpd_np = m_img_wrpd.numpy()
    bbox = get_bbox(f_img_np, get_idcs=True)
    f_img_np = f_img_np[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1], bbox[0][2]:bbox[1][2]]
    m_img_wrpd_np = m_img_wrpd_np[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1], bbox[0][2]:bbox[1][2]]
    # Calculate PSNR
    psnr = 10 * np.log10(1 / np.mean((f_img_np - m_img_wrpd_np) ** 2))
    mse_scikit = mean_squared_error(f_img_np, m_img_wrpd_np)
    ssim_none = ssim(f_img_np, m_img_wrpd_np, data_range=1.0)
    dice_scores = np.array([0.0])

    print("PSNR: {}, SSIM: {}, MSE: {}, Dice_Scores: {}, Dice Score Mean: {}".format(psnr, ssim_none, mse_scikit, dice_scores, dice_scores.mean()))
    metrics_dict = {"PSNR": psnr, "SSIM": ssim_none, "MSE": mse_scikit, "epoch": epoch, "ttime": ttime}
    return metrics_dict


def get_bbox(img, get_idcs=False):
    non_zero_indices = np.array(np.where(img != 0))
    min_indices = np.min(non_zero_indices, axis=1)
    max_indices = np.max(non_zero_indices, axis=1)
    bbox = max_indices - min_indices
    if get_idcs:
        return np.array([min_indices, max_indices])
    else:
        return bbox

def save_checkpoint(args, model, optimizer, epoch, model_name, best_mse, best_ssim, best_psnr):
    sr_state_dict = model.sr_net.state_dict()
    tf_state_dict = model.tf_net.state_dict()
    save_dict = {
        'epoch': epoch,
        'sr_state_dict': sr_state_dict,
        'tf_state_dict': tf_state_dict,
        'best_mse': best_mse,
        'best_ssim': best_ssim,
        'best_psnr': best_psnr
    }
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    filename = os.path.join(args.path_save_model, model_name)
    os.makedirs(args.path_save_model, exist_ok=True)
    torch.save(save_dict, filename)
    print('Saving checkpoint to:', filename)


def load_model(args, model, optimizer=None, load_prior=False):
    model_name = args.path_atlas_img
    model_name = model_name[model_name.rfind('/') + 1:]
    model_name = model_name.replace('.nii.gz', "_prior.pt")

    state = torch.load(os.path.join(args.path_save_model, model_name), map_location='cpu')
    model.sr_net.load_state_dict(state['sr_state_dict'])
    model.seg_net.load_state_dict(state['seg_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])

    print("Loading model= {}, Loading_prior={}".format(model_name, load_prior))
    return model, optimizer


def load_checkpoint(args, model, optimizer=None, filename="", sr_prior=True, tf_prior=True):
    path_checkpoint = os.path.join(args.path_save_model, filename)
    if os.path.isfile(path_checkpoint):
        print("=> loading checkpoint '{}'".format(path_checkpoint))
        checkpoint = torch.load(path_checkpoint)
        start_epoch = checkpoint['epoch']
        if sr_prior:
            model.sr_net.load_state_dict(checkpoint['sr_state_dict'])
        if tf_prior:
            model.tf_net.load_state_dict(checkpoint['tf_state_dict'])
        best_mse = checkpoint['best_mse']
        best_ssim = checkpoint['best_ssim']
        best_psnr = checkpoint['best_psnr']
        if optimizer is not None:
            new_optim = torch.optim.Adam(model.sr_net.parameters(), lr=1e-4)
            new_optim.load_state_dict(checkpoint['optimizer'])
            optimizer.param_groups[0] = new_optim.param_groups[0]
        print("=> loaded checkpoint '{}' (epoch {}, best_mse {}, best_ssim {}, best_psnr {})"
              .format(path_checkpoint, checkpoint['epoch'], checkpoint['best_mse'], checkpoint['best_ssim'], checkpoint['best_psnr']))


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad
