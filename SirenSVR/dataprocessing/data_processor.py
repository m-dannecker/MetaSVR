import math
import torch
from .data_utils import *
from skimage import exposure
import numpy as np


class DataProcessor:
    def __init__(self, args):
        args.n_samples = args.n_samples_total // len(args.path_stack_img)
        self.args = args
        self.device = args.device
        self.n_slices_total = 0
        self.stacks = []

        slice_num_offset = 0
        for stack_id, stack_img_file in enumerate(args.path_stack_img):
            dict_entry = self.load_stack(args, stack_img_file, stack_id, slice_num_offset)
            self.n_slices_total += int(dict_entry['n_slices'])
            slice_num_offset += int(dict_entry['n_slices'])
            self.init_voxel_weights(dict_entry)
            self.stacks.append(dict_entry)

        self.n_stacks = len(self.stacks)
        self.bbox = self.get_bbox()
        self.slice_tf_params = torch.nn.Parameter(torch.nn.init.normal_(torch.Tensor(size=(self.n_slices_total, 6)),
                                                                        mean=0.0, std=0.15).to(self.device),
                                                  requires_grad=True) if args.tf_parameter_learning else None
        self.slice_weights = torch.nn.Parameter(torch.ones((self.n_slices_total, 1),
                                                           device=self.device),
                                                requires_grad=True) if args.slice_weighting else None
        self.slice_scalings = torch.nn.Parameter(torch.ones((self.n_slices_total, 1),
                                                            device=self.device),
                                                 requires_grad=True) if args.slice_scaling else None

    def load_stack(self, args, stack_name, stack_id, slice_num_offset, downsample=False):
        stack_img, affine, org_affine, spacing = load_nifti(args, stack_id, zero_center=True, flip_img=False,
                                                            downsample=downsample)
        stack_img = torch.from_numpy(stack_img).to(self.device, torch.float)
        affine = torch.from_numpy(affine).to(self.device, torch.float)
        org_affine = torch.from_numpy(org_affine).to(self.device, torch.float)
        vxl_shape = torch.Tensor(list(stack_img.shape)).to(self.device, torch.float)
        phys_shape = torch.cat([affine[:3, :3] @ vxl_shape[:3], torch.Tensor([1]).to(self.device, torch.float)])

        mask_w_halo, mask = gaussian_blur(stack_img, stack_id, _2D=True)

        if args.intensity_preproc == "normalization":
            stack_img[mask] = (stack_img[mask] - stack_img[mask].min()) / (
                    stack_img[mask].max() - stack_img[mask].min())
        elif args.intensity_preproc == "standard-normalization":
            stack_img[mask] = (stack_img[mask] - stack_img[mask].mean()) / stack_img[mask].std()
            stack_img[mask] = (stack_img[mask] - stack_img[mask].min()) / (
                    stack_img[mask].max() - stack_img[mask].min())
        else:
            raise NotImplementedError("Unknown intensity preprocessing method")

        coords_nz_w_halo = torch.nonzero(torch.from_numpy(mask_w_halo)).to(self.device)
        coords_nz = torch.nonzero(torch.from_numpy(mask)).to(self.device)
        dict_entry = {"stack_name": stack_name,
                      "stack_id": stack_id,
                      "stack_img": stack_img,
                      "coords_nz_w_halo": coords_nz_w_halo,
                      "coords_nz": coords_nz,
                      "affine": affine,
                      "org_affine": org_affine,
                      "spacing": torch.from_numpy(spacing).to(self.device, torch.float),
                      "vxl_shape": vxl_shape,
                      "phys_shape": phys_shape,
                      "voxel_weights": None,
                      "maxs": None,
                      "mins": None,
                      "n_slices": vxl_shape[2].item(),
                      "slice_num_offset": slice_num_offset,
                      "smpl_cnt_w_halo": 9e8,
                      "smpl_cnt": 9e8}
        return dict_entry

    def get_batch(self, epoch=0, subs=None, get_all=False):
        rel_stacks = self.stacks if subs is None else [self.stacks[i] for i in subs]
        n_stacks = len(rel_stacks)
        n_samples = self.args.n_samples
        n_samples = self.stacks[subs[0]]["coords_nz"].shape[0] if get_all else n_samples
        coords_all = torch.ones((n_stacks, n_samples, 3)).to(self.device, torch.float)
        slice_idcs = torch.empty((n_stacks, n_samples, 2)).to(self.device, torch.float)
        slice_tf_params = torch.empty((n_stacks, n_samples, 6)).to(self.device, torch.float) \
            if self.args.tf_parameter_learning else None
        slice_weights = torch.empty((n_stacks, n_samples, 1)).to(self.device, torch.float) \
            if self.args.slice_weighting else None
        slice_scalings = torch.empty((n_stacks, n_samples, 1)).to(self.device,
                                                                  torch.float) if self.args.slice_scaling else None
        voxel_weights = torch.empty((n_stacks, n_samples, 1)).to(self.device, torch.float) \
            if self.args.voxel_weighting else None
        values_img = torch.zeros((n_stacks, n_samples, 1)).to(self.device, torch.float)
        affines = torch.zeros((n_stacks, 4, 4)).to(self.device, torch.float)

        for i, stack in enumerate(rel_stacks):
            smpl_cnt = "smpl_cnt_w_halo" if epoch < self.args.sample_halo_for_num_epochs else "smpl_cnt"
            coords_nz = "coords_nz_w_halo" if epoch < self.args.sample_halo_for_num_epochs else "coords_nz"
            # if iterated through all coordinates of stack, reset and reshuffle
            if (stack[smpl_cnt] + 1) * n_samples >= len(stack[coords_nz]):
                stack[smpl_cnt] = 0
                stack[coords_nz] = stack[coords_nz][torch.randperm(len(stack[coords_nz]))]

            coords_vxl = stack[coords_nz][stack[smpl_cnt] * n_samples: (stack[smpl_cnt] + 1) * n_samples]
            stack[smpl_cnt] = stack[smpl_cnt] + 1

            # set coord-origin to center of image
            center = (stack["vxl_shape"][:3] - 1) / 2
            coords_all[i] = (coords_vxl - center) * stack["spacing"]

            slice_idcs[i] = self.get_slice_idcs(stack, coords_vxl, n_samples)
            if self.args.tf_parameter_learning:
                slice_tf_params[i] = self.slice_tf_params[coords_vxl[:, -1] + stack["slice_num_offset"]].to(torch.float)
            if self.args.slice_weighting:
                slice_weights[i] = self.slice_weights[coords_vxl[:, -1] + stack["slice_num_offset"]].to(torch.float)
            if self.args.voxel_weighting:
                if epoch < self.args.sample_halo_for_num_epochs:  # don't apply voxel-weighting during halo period
                    voxel_weights[i] = torch.ones_like(coords_vxl[:, 0:1], device=self.device, dtype=torch.float)
                else:
                    voxel_weights[i] = self.get_voxel_weights_of_coords(stack, coords_vxl, n_samples)
            if self.args.slice_scaling:
                slice_scalings[i] = self.slice_scalings[coords_vxl[:, -1] + stack["slice_num_offset"]].to(torch.float)
            values_img[i] = stack["stack_img"][coords_vxl[:, 0], coords_vxl[:, 1], coords_vxl[:, 2], None]
            affines[i] = stack["affine"]
        return coords_all, affines, slice_idcs, slice_tf_params, slice_weights, voxel_weights, slice_scalings, values_img

    def get_slice_idcs(self, stack, coords_vxl, n_samples):
        slice_idcs = torch.stack([stack["stack_id"] * torch.ones(n_samples, device=self.device),
                                  coords_vxl[:, -1]], dim=1).to(torch.float)
        slice_idcs = 2 * (slice_idcs / (torch.Tensor([self.n_stacks, stack["n_slices"]]).to(self.device))) - 1
        return slice_idcs

    def get_voxel_weights_of_coords(self, stack, coords_vxl, n_samples):
        if stack["voxel_weights"] is not None:
            resolution = self.args.voxel_weighting
            x_min, y_min, z_min = stack["mins"]
            x = ((coords_vxl[:, 0] - x_min) / resolution).long()
            y = ((coords_vxl[:, 1] - y_min) / resolution).long()
            z = ((coords_vxl[:, 2] - z_min) / resolution).long()
            voxel_weights = stack["voxel_weights"][x, y, z][..., None]
        else:
            voxel_weights = torch.empty((n_samples, 1)).to(self.device, torch.float)
        return voxel_weights

    def get_slice_weights_of_coords(self, stack, coords_vxl):
        slice_weights = self.slice_weights[coords_vxl[:, -1] + stack["slice_num_offset"]].to(torch.float)
        return slice_weights

    def get_slice_scaling_of_coords(self, stack, coords_vxl):
        slice_scaling = self.slice_scalings[coords_vxl[:, -1] + stack["slice_num_offset"]].to(torch.float)
        return slice_scaling

    def get_bbox(self):
        bbox = torch.Tensor([[1e5, 1e5, 1e5], [-1e5, -1e5, -1e5]]).to(self.device)
        for i, stack in enumerate(self.stacks):
            coords_vxl = stack["coords_nz"].to(torch.float)
            center = (stack["vxl_shape"][:3] - 1) / 2
            coords_vxl = (coords_vxl - center) * stack["spacing"]
            coords_wld = (stack['affine'][:3, :3] @ coords_vxl.T).T + stack['affine'][:3, 3]
            bbox_stack = torch.stack((torch.amin(coords_wld, dim=0), torch.amax(coords_wld, dim=0)))
            # add 5% margin to bbox to prevent cut-offs
            bbox_stack[0] = bbox_stack[0] - 0.1 * (bbox_stack[1] - bbox_stack[0])
            bbox_stack[1] = bbox_stack[1] + 0.1 * (bbox_stack[1] - bbox_stack[0])
            stack['bbox'] = bbox_stack
            bbox = torch.stack((torch.minimum(bbox[0], bbox_stack[0]), torch.maximum(bbox[1], bbox_stack[1])))
        return bbox

    def get_psf_stds(self, fraction=1):
        psf_stds = torch.zeros((self.n_stacks, 3)).to(self.device, torch.float)
        res = torch.zeros((self.n_stacks, 3)).to(self.device, torch.float)
        for i, stack in enumerate(self.stacks):
            pre_facs = torch.ones(3).to(self.device, torch.float) * 1.2
            spacing = stack["spacing"].clone()
            pre_facs[spacing.argmax()] = 1.0 if spacing.argmax() != spacing.argmin() else 1.2
            if self.args.slice_thickness is not None: spacing[spacing.argmax()] = self.args.slice_thickness  # if slice thickness is given, use it
            psf_stds[i] = ((spacing * pre_facs) / 2.355) * fraction
            res[i] = spacing
        self.args.psf_stds = psf_stds
        return psf_stds, res

    def get_all_voxel_weights(self):
        voxel_weights = []
        for stack in self.stacks:
            if stack["voxel_weights"] is not None:
                voxel_weights.append(stack["voxel_weights"])
        return voxel_weights

    def init_voxel_weights(self, stack):
        if self.args.voxel_weighting > 0:
            maxs = stack["coords_nz"].max(dim=0)[0]
            mins = stack["coords_nz"].min(dim=0)[0]
            lat_shape = torch.div((maxs - mins), self.args.voxel_weighting, rounding_mode='trunc') + 1
            init = torch.ones(tuple(lat_shape), device=self.args.device)
            voxel_weights_stack = torch.nn.Parameter(init, requires_grad=True)
        else:
            voxel_weights_stack = None
            maxs = None
            mins = None
        stack["voxel_weights"] = voxel_weights_stack
        stack["mins"] = mins
        stack["maxs"] = maxs
