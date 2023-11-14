import time
import ants
import torch
import random
from torch.cuda.amp import GradScaler, autocast
from utils.utils import *
from utils.transform_utils import generate_img_grid
from skimage import exposure
from models.inr_rec import *
import copy


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def fit_model(args, model, optimizer, loss_fns, data_processor, scheduler=None, start_epoch=0, meta_epoch=None):
    halo_model = None
    metrics_dict = None
    metrics_dicts = []
    model.train()
    scaler = GradScaler() if args.amp else None
    epoch_time = time.time()
    total_training_time = 0
    prev_loss_sr = 0
    loss_sr = torch.Tensor([0])
    for epoch in range(start_epoch, args.max_epochs):
        training_time = time.time()
        psf_scheduler(args, model, epoch, verbose=500)
        prev_loss_sr = freeze_sr(args, model, data_processor, args.freeze_rec_epochs, epoch, loss_sr.item(), prev_loss_sr)
        coords, affines, slice_idcs, slice_tf_params, slice_weights, voxel_weights, slice_scalings, v_img = data_processor.get_batch(epoch=epoch)

        with autocast(enabled=args.amp):
            v_img_p, tf_embed, coords_ = model(coords, affines, slice_idcs, slice_tf_params, inf=False)
            loss_sr = calc_sr_loss(v_img_p, v_img, loss_fns['sr'], slice_scalings)
            loss_tf = calc_tf_loss(tf_embed)
            loss_sr = calc_slice_weighting_loss(loss_sr, slice_weights)
            loss_sr = calc_voxel_weighting_loss(loss_sr, voxel_weights)
            loss_sr = loss_sr.mean()
            loss = loss_sr + args.tf_reg_weight * loss_tf

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_training_time += time.time() - training_time
        if not (epoch+1) % args.val_every and meta_epoch != -1:
            metrics_dict = evaluate_model(args, data_processor, model, halo_model=halo_model, epoch=epoch, ttime=total_training_time, meta_epoch=meta_epoch)
            metrics_dicts.append(metrics_dict)
            # visualize_slice_weights(dataprocessor=data_processor, stack_id=0, path_save=args.path_save)
            # visualize_voxel_weights(dataprocessor=data_processor, stack_id=0, slice_id=18, path_save=args.path_save)
            if args.train_prior: save_model(args, epoch, model, optimizer)


        # if not (epoch+1) % args.val_every or epoch == 0:
        #     metrics_dict = evaluate_model(args, data_processor, model)
        #     metrics_dict['train_time'] = total_training_time
        #     metrics_dict['epoch'] = (epoch+1)
        #     metrics_dicts.append(metrics_dict)
        #     plot_convergence(metrics_dicts, args.path_save, meta_init=bool(args.load_meta_learned_model))

        epoch_time = print_epoch_stats(epoch, loss, loss_sr, loss_tf, epoch_time, print_every=500)
        if scheduler is not None: scheduler.step()
        if epoch == args.sample_halo_for_num_epochs:
            halo_model = copy.deepcopy(model)
            halo_model.eval()
    print("\n ################ Total training time (s): ", total_training_time, " ################ \n")
    return metrics_dicts


def evaluate_model(args, data_processor, model, halo_model=None, inf_steps=1000, epoch=None, ttime=None, meta_epoch=None):
    model.eval()
    inf_res = args.inf_res * torch.tensor([1.0, 1.0, 1.0], device=args.device).to(torch.float32)
    bbox = data_processor.bbox
    bbox_size = ((bbox[1] - bbox[0]) / inf_res).round().long()
    coords_vxl = generate_img_grid(bbox_size).cuda(0)
    coords_vxl = coords_vxl * inf_res + bbox[0]
    bs = int(coords_vxl.shape[0] / inf_steps)
    values_sr_p = torch.zeros((1, coords_vxl.shape[0], 1), device=args.device)
    values_sr_p_halo = torch.zeros((1, coords_vxl.shape[0], 1), device=args.device) if halo_model is not None else None

    with torch.no_grad():
        with autocast(enabled=args.amp):
            for j in range(0, coords_vxl.shape[0], bs):
                v_p_sr, _, coords_ = model(coords_vxl[None, j:j + bs], inf=True)
                values_sr_p[0, j:j + bs] = v_p_sr.detach()
                if halo_model is not None:
                    v_p_sr_halo, _, _ = halo_model(coords_vxl[None, j:j + bs], inf=True)
                    values_sr_p_halo[0, j:j + bs] = v_p_sr_halo.detach()

    values_sr_p = np.clip(values_sr_p.view(list(bbox_size)).cpu().numpy(), 0.0, 1.0)
    if halo_model is not None:
        values_sr_p_halo = np.clip(values_sr_p_halo.view(list(bbox_size)).cpu().numpy(), 0.0, 1.0)

    affine = np.array([[inf_res[0].cpu(), 0, 0, 0],
                       [0, inf_res[1].cpu(), 0, 0],
                       [0, 0, inf_res[2].cpu(), 0],
                       [0, 0, 0, 1]])

    rec_nii, rec_masked_nii, mask_nii, reg_affine, _, _ = post_process_reconstruction(args, values_sr_p, affine,
                                                                                values_sr_p_halo=values_sr_p_halo)
    suffix = ""#"_ep={}_ttime={}s".format(epoch, ttime) if epoch is not None else ""
    if meta_epoch is None: nib.save(rec_nii, os.path.join(args.path_save,  args.rec_file_name.split(".")[0] + "{}.nii.gz".format(suffix)))
    if rec_masked_nii is not None:
        if meta_epoch is not None:
            rec_masked_nii_name = os.path.join(args.path_save, args.rec_file_name.split(".")[0] + f"_masked_mep_{meta_epoch}.nii.gz")
            nib.save(rec_masked_nii, rec_masked_nii_name)
        else:
            nib.save(rec_masked_nii, os.path.join(args.path_save, args.rec_file_name.split(".")[0] + "_masked.nii.gz"))
    # print(f"Volume saved to {os.path.join(args.path_save, args.rec_file_name)}")

    # align with reference image (e.g., atlas) if available
    if reg_affine is not None:
        coords_reg = (torch.from_numpy(reg_affine[:3, :3]).to(args.device, torch.float) @ coords_vxl.T).T
        # coords_reg = (coords_reg - torch.amin(coords_reg, dim=0)) / (torch.amax(coords_reg, dim=0) - torch.amin(coords_reg, dim=0)) * 2 - 1
        values_sr_p = torch.zeros((1, coords_vxl.shape[0], 1), device=args.device)
        values_grad_p = torch.zeros((1, coords_vxl.shape[0], 1), device=args.device)
        values_lp_p = torch.zeros((1, coords_vxl.shape[0], 1), device=args.device)
        values_sr_p_halo = torch.zeros((1, coords_vxl.shape[0], 1),
                                       device=args.device) if halo_model is not None else None
        with torch.no_grad():
            with autocast(enabled=args.amp):
                for j in range(0, coords_reg.shape[0], bs):
                    v_p_sr, _, coords_ = model(coords_reg[None, j:j + bs], inf=True, grad=args.rec_grad)
                    values_sr_p[0, j:j + bs] = v_p_sr.detach()
                    if args.rec_grad:
                        values_grad_p[0, j:j + bs] = gradient(v_p_sr, coords_).detach().norm(dim=-1, keepdim=True)
                        # values_lp_p[0, j:j + bs] = laplace(v_p_sr, coords_).detach()
                    if halo_model is not None:
                        v_p_sr_halo, _, _ = halo_model(coords_reg[None, j:j + bs], inf=True)
                        values_sr_p_halo[0, j:j + bs] = v_p_sr_halo.detach()
        affine[:3, 3] = -reg_affine[:3, 3]
        values_sr_p = np.clip(values_sr_p.view(list(bbox_size)).cpu().numpy(), 0.0, 1.0)
        values_grad_p = values_grad_p.view(list(bbox_size)).cpu().numpy()
        values_lp_p = values_lp_p.view(list(bbox_size)).cpu().numpy()
        if halo_model is not None:
            values_sr_p_halo = np.clip(values_sr_p_halo.view(list(bbox_size)).cpu().numpy(), 0.0, 1.0)

        rec_reg_nii, rec_reg_masked_nii, mask_rg_nii, _, grad_reg_nii, lp_reg_nii = post_process_reconstruction(args, values_sr_p, affine,
                                                                                      values_sr_p_halo=values_sr_p_halo,
                                                                                      reg=False, values_grad_p=values_grad_p, values_lp_p=values_lp_p)
        suffix = ""#"_ep={}_ttime={}s".format(epoch, ttime) if epoch is not None else ""
        nib.save(rec_reg_masked_nii,
                 os.path.join(args.path_save, args.rec_file_name.split(".")[0] + "_masked_reg{}.nii.gz".format(suffix)))
        print("Volume saved to {}".format(os.path.join(args.path_save,
                                                       args.rec_file_name.split(".")[0] + "_masked_reg{}.nii.gz".format(suffix))))
        if args.rec_grad:
            nib.save(grad_reg_nii,
                     os.path.join(args.path_save,
                                  args.rec_file_name.split(".")[0] + "_grad_masked_reg{}.nii.gz".format(suffix)))
            nib.save(lp_reg_nii,
                     os.path.join(args.path_save,
                                  args.rec_file_name.split(".")[0] + "_lp_masked_reg{}.nii.gz".format(suffix)))

    if reg_affine is not None:
        rec_filename = args.rec_file_name.replace(".nii.gz", "_masked_reg.nii.gz")
    elif rec_masked_nii is not None:
        rec_filename = rec_masked_nii_name
    else:
        rec_filename = args.rec_file_name
    if "HCP_pruned" in args.path_img_ref or "dHCP_fetal" in args.path_img_ref:
        metrics_dict = eval_and_save_segmentation(args.path_img_ref, args.path_save, args.stack_ids, rec_filename, epoch, ttime)
    else:
        metrics_dict = None
    return metrics_dict
