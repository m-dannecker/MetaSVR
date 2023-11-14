import numpy as np
import torch
import nibabel as nib
import nibabel.processing as nip
import scipy.ndimage as ndi
import math
# import ants
from skimage import exposure
import subprocess
import os
import fsl.data.image as fimage
import fsl.transform.flirt as flirt
import ants


def apply_pre_processing(args, stack_id):
    stack = ants.image_read(args.path_stack_img[stack_id])
    if args.path_stack_mask is not None:
        print("Masking stack with mask from: ", args.path_stack_mask[stack_id])
        mask = ants.image_read(args.path_stack_mask[stack_id])
        stack = ants.mask_image(stack, mask)
    if args.bias_field_correction:
        print("Performing bias field correction")
        stack = ants.n4_bias_field_correction(stack)
    if args.denoise != 'none':
        print("Performing denoising")
        stack = ants.denoise_image(stack, noise_model=args.denoise)
    tmp_file_name = os.path.join(args.path_tmp, args.path_stack_img[stack_id].split("/")[-1].replace(".nii.gz", "_preproc.nii.gz"))
    print("Writing pre-processed stack to: ", tmp_file_name)
    ants.image_write(stack, tmp_file_name)
    if args.stack_pre_align != 'none':  # align stacks with atlas
        if args.stack_pre_align == 'individual':
            stack, affine_stack2atlas = align_stack(args, tmp_file_name)
        elif args.stack_pre_align == 'global':
            stack, affine_stack2atlas = align_stack(args, tmp_file_name, affine_stack2atlas=args.affine_stack2atlas)
            args.affine_stack2atlas = affine_stack2atlas
    else:
        stack = ants.to_nibabel(stack)
    return stack


def load_nifti(args, stack_id, zero_center=True, flip_img=False, downsample=False):
    stack = apply_pre_processing(args, stack_id)
    if downsample:
        stack = resample_nib(stack, voxel_spacing=(1.0, 1.0, 1.0), order=3)
    stack = crop_nifti_to_non_zero_bbox(stack, padding_percentage=0.15)
    spacing = stack.header["pixdim"][1:4]
    volume = stack.get_fdata().astype(np.float32)
    w, h, d = volume.shape[:3]
    affine_stack = np.copy(stack.affine)
    # normalize spacing of affine matrix
    affine_stack[:3, :3] = affine_stack[:3, :3] @ np.diag(1 / spacing)

    if flip_img:
        flip = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if np.linalg.det(affine_stack[:3, :3]) < 0:
            affine_stack[:3, :3] = flip @ affine_stack[:3, :3]
            affine_stack[:3, -1:] = flip @ affine_stack[:3, -1:]
    if zero_center:
        # correct center 0 shift in origin (i.e., since origin vxl becomes -1*center voxel, correct world coordinate)
        center = np.array([w, h, d]) / 2
        center = affine_stack[:3, :3] @ (center * spacing)
        affine_stack[:3, 3] += center
    volume = np.clip(volume, a_min=0, a_max=np.percentile(volume, 99.9))
    return volume, affine_stack, stack.affine, spacing


def align_stack(args, path_stack, affine_stack2atlas=None):
    # path_mat = os.path.join(os.getcwd().split('INR4REC')[0], 'INR4REC/tmp/stack2atlas.mat')
    path_mat = os.path.join(args.path_tmp, 'stack2atlas.mat')
    path_out = os.path.join(args.path_tmp, path_stack.split('/')[-1].split('.')[0] + '_reg.nii.gz')
    if affine_stack2atlas is None:
        os.makedirs(os.path.dirname(path_mat), exist_ok=True)
        command = "{} " \
                  "-in {} " \
                  "-ref {} " \
                  "-out {}" \
                  " -omat {}" \
                  " -bins 256" \
                  " -cost corratio" \
                  " -searchrx -180 180" \
                  " -searchry -180 180" \
                  " -searchrz -180 180" \
                  " -dof 6" \
                  " -interp trilinear".format(args.path_flirt, path_stack, args.path_img_ref, path_out, path_mat)
        new_env = os.environ.copy()
        new_env['FSLOUTPUTTYPE'] = "NIFTI_GZ"
        subprocess.run(command, shell=True, env=new_env, capture_output=True)
        mat_stack2atlas = np.loadtxt(path_mat)
        affine_stack2atlas = flirt.fromFlirt(mat_stack2atlas, fimage.Nifti(nib.load(path_stack).header),
                                             fimage.Nifti(nib.load(args.path_img_ref).header), from_='world', to='world')

    stack_nii = nib.load(path_stack)
    new_affine = affine_stack2atlas @ stack_nii.affine
    new_img = nib.Nifti1Image(stack_nii.get_fdata(), new_affine)
    nib.save(new_img, path_out.split('.')[0] + '_reg2atlas.nii.gz')
    return new_img, affine_stack2atlas

def load_npz(file, ref_shape):
    npz_file = np.load(file)
    volume = npz_file['probabilities']
    # order dimensions according to ref_shape
    volume = np.transpose(volume, (3, 2, 1, 0))
    assert (volume.shape[:-1] == ref_shape).all(), "Shape mismatch between reference shape and loaded volume"
    return volume


def register_atlas_to_stack(path_atlas_img, path_stack_img, path_atlas_seg, path_stack_seg, segmentation=False):
    new_path_atlas_img = path_atlas_img.replace('.nii.gz', '_reg.nii.gz')
    new_path_atlas_seg = ""
    fi = nib.load(path_stack_img)
    mi = nib.load(path_atlas_img)
    fi = ants.from_nibabel(resample_nib(fi, voxel_spacing=(1.0, 1.0, 1.0), order=3))
    mi = ants.from_nibabel(resample_nib(mi, voxel_spacing=(1.0, 1.0, 1.0), order=3))

    fi_n = ants.iMath_normalize(fi)
    mi_n = ants.iMath_normalize(mi)
    mytx = ants.registration(fixed=fi_n, moving=mi_n, type_of_transform='Rigid')
    mi_wrpd = ants.apply_transforms(fixed=fi, moving=mi, transformlist=mytx['fwdtransforms'])

    mask = mi_wrpd.numpy() > 0
    mi_wrpd_np = exposure.match_histograms(np.clip(mi_wrpd.numpy(), 0, None), fi.numpy()) * mask
    mi_wrpd = ants.from_numpy(mi_wrpd_np, origin=mi_wrpd.origin, spacing=mi_wrpd.spacing, direction=mi_wrpd.direction)
    ants.image_write(mi_wrpd, new_path_atlas_img)

    if segmentation:
        new_path_atlas_seg = path_atlas_seg.replace('.nii.gz', '_reg.nii.gz')
        fs = nib.load(path_stack_seg)
        ms = nib.load(path_atlas_seg)
        fs = ants.from_nibabel(resample_nib(fs, voxel_spacing=(1.0, 1.0, 1.0), order=0))
        ms = ants.from_nibabel(resample_nib(ms, voxel_spacing=(1.0, 1.0, 1.0), order=0))

        ms_wrpd = ants.apply_transforms(fixed=fs, moving=ms, transformlist=mytx['fwdtransforms'],
                                        interpolator='multiLabel')
        ants.image_write(ms_wrpd, new_path_atlas_seg)

    return new_path_atlas_img, new_path_atlas_seg


def gaussian_blur(stack, stack_id, stds=None, _2D=False, threshold=0.01):
    mask_w_halo = (stack > 0.01).cpu().numpy()
    mask = mask_w_halo.copy()
    if stds is None:
        # get bbox of non-zero voxels in stack to determine magnitude of stds for gaussian blurring
        bbox = np.array(np.where(mask_w_halo))
        bbox = np.array([bbox.min(axis=1), bbox.max(axis=1)])
        bbox_size = bbox[1] - bbox[0]
        stds = (bbox_size * 0.030)
    else:
        stds = stds.cpu().numpy()
    mask_mean_num_non_zero = np.count_nonzero(mask_w_halo) / mask_w_halo.shape[2]
    if _2D:
        stds = stds[:2]
        for i in range(mask_w_halo.shape[2]):
            if np.count_nonzero(mask_w_halo[:, :, i]) < 0.1 * mask_mean_num_non_zero:
                # if a slice is cut-off in the middle or has holes, blurring introduces zero intesnities within the brain
                print("Discarding slice {} of stack {} due to low number of non-zero voxels. "
                      "{} non-zero values found. Mean non-zero values per slice for this stack: {}"
                      .format(i, stack_id, np.count_nonzero(mask_w_halo[:, :, i]), mask_mean_num_non_zero))
                mask_w_halo[:, :, i] = 0
            else:
                mask_w_halo[:, :, i] = ndi.gaussian_filter(mask_w_halo[:, :, i].astype(np.float32), tuple(stds)) > threshold
    else:
        mask_w_halo = ndi.gaussian_filter(mask_w_halo.astype(np.float32), tuple(stds)) > threshold
    return mask_w_halo, mask


def resample_nib(img, voxel_spacing=(1, 1, 1), order=3, verbose=True):
    """CODE FROM SUPROSANNA SHIT"""
    """Resamples the nifti from its original spacing to another specified spacing

    Parameters:
    ----------
    img: nibabel image
    voxel_spacing: a tuple of 3 integers specifying the desired new spacing
    order: the order of interpolation

    Returns:
    ----------
    new_img: The resampled nibabel image 

    """
    # resample to new voxel spacing based on the current x-y-z-orientation
    aff = img.affine
    shp = img.shape
    zms = img.header.get_zooms()
    # Calculate new shape
    new_shp = tuple(np.rint([
        shp[0] * zms[0] / voxel_spacing[0],
        shp[1] * zms[1] / voxel_spacing[1],
        shp[2] * zms[2] / voxel_spacing[2]
    ]).astype(int))
    new_aff = nib.affines.rescale_affine(aff, shp, voxel_spacing, new_shp)
    new_img = nip.resample_from_to(img, (new_shp, new_aff), order=order, cval=0)
    if verbose:
        print("[*] Image resampled to voxel size:", voxel_spacing)
    return new_img


def seg2distance_map(img_np, spacing, n_classes, sdf=False):
    distance_maps = []
    for c in range(0, n_classes):
        img = img_np == c
        dist_inside = ndi.distance_transform_edt(img, sampling=spacing)
        if sdf:
            img_inv = np.logical_not(img)
            dist_outside = ndi.distance_transform_edt(img_inv, sampling=spacing)
            dist_inside -= dist_outside
            dist_inside = np.tanh(dist_inside)
        distance_maps.append(dist_inside)
    return np.stack(distance_maps, axis=-1)


def crop_non_zero_3D_bbox(volume):
    non_zero_indices = np.array(np.where(volume != 0))
    min_indices = np.min(non_zero_indices, axis=1)
    max_indices = np.max(non_zero_indices, axis=1)

    cropped_arr = volume[min_indices[0]:max_indices[0] + 1,
                  min_indices[1]:max_indices[1] + 1,
                  min_indices[2]:max_indices[2] + 1]

    return cropped_arr


def crop_nifti_to_non_zero_bbox(nifti_img, padding_percentage=0.1):
    nifti_data = nifti_img.get_fdata()

    non_zero_indices = np.array(np.where(nifti_data != 0))
    min_indices = np.min(non_zero_indices, axis=1)
    max_indices = np.max(non_zero_indices, axis=1)

    cropped_data = nifti_data[min_indices[0]:max_indices[0] + 1,
                   min_indices[1]:max_indices[1] + 1,
                   min_indices[2]:max_indices[2] + 1]

    padding = np.ceil(np.array(cropped_data.shape) * (padding_percentage/2)).astype(int)
    padded_data = np.pad(cropped_data, ((padding[0], padding[0]),
                                        (padding[1], padding[1]),
                                        (padding[2], padding[2])),
                         mode='constant')
    new_affine = nifti_img.affine.copy()
    new_affine[:3, 3] += new_affine[:3, :3] @ np.array([min_indices[0], min_indices[1], min_indices[2]])
    new_affine[:3, 3] -= new_affine[:3, :3] @ np.array([padding[0], padding[1], padding[2]])
    cropped_nifti_img = nib.Nifti1Image(padded_data, new_affine)
    return cropped_nifti_img