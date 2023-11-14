import torch.nn as nn
import torch
import numpy as np
import tinycudann as tcnn
from .siren import Siren, MLP  # , SirenMultiHead#, SirenSingleHeadMeta, SirenMeta
# import tinycudann as tcnn
from utils.transform_utils import *
# from torchmeta.modules import (MetaModule, MetaSequential, MetaLinear)
from collections import OrderedDict

CONFIG_TINY = {
            "encoding": {
                "otype": "HashGrid",
                "n_levels": 12, #12
                "n_features_per_level": 2,
                "log2_hashmap_size": 19, #19
                "base_resolution": 6, # 7
                "per_level_scale": 1.3819,
            },
            "encoding_seg": {
                "otype": "HashGrid",
                "n_levels": 6, #12
                "n_features_per_level": 2,
                "log2_hashmap_size": 7, #19
                "base_resolution": 8, # 7
                "per_level_scale": 1.2, #1.3819
            },
            "encodingTW": {
                "otype": "Frequency",
                "n_frequencies": 12,
            },
            "network": {
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 128,
                "n_hidden_layers": 2
            }
        }


class INR_Rec(nn.Module):
    def __init__(self, args, bbox, psf_stds, res):
        super().__init__()
        self.args = args
        self.bbox = bbox
        self.n_s_psf = self.args.psf_k_size
        self.n_s_psf_inf = self.args.psf_k_size_inf
        self.psf_stds = psf_stds
        self.tiny_model = self.args.hash_grid
        hidden_omega = 1.75 if args.normalize_siren else 32.5
        first_omega = 30 if args.normalize_siren else 40.0
        if self.tiny_model:
            self.sr_net = tcnn.NetworkWithInputEncoding(3, 1,
                CONFIG_TINY["encoding"], CONFIG_TINY["network"]
            )
        else:
            self.sr_net = Siren(3, 1, args.sr_hidden_size,
                                args.sr_num_layers, first_omega_0=first_omega, hidden_omega_0=hidden_omega, outermost_linear=True, normalize=args.normalize_siren)
        if args.siren_motion_correction:
            self.tf_net = Siren(args.slice_emb_dim, 6, args.tf_hidden_size,
                                args.tf_num_layers, first_omega_0=first_omega, hidden_omega_0=hidden_omega, outermost_linear=True, normalize=args.normalize_siren)
        else:
            self.tf_net = MLP(args.slice_emb_dim, 6, args.tf_hidden_size, args.tf_num_layers)

    def forward(self, coords, affines=None, s_idcs=None, slice_tf_params=None, inf=False, mni=False, grad=False):
        if grad:
            coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        bs, ns, _ = coords.shape
        if (self.n_s_psf > 0 and not inf) or (inf and self.n_s_psf_inf > 0):
            coords, dists = self.apply_psf(coords, bs, ns, inf)
            s_idcs = s_idcs[:, :, None].expand(-1, -1, self.n_s_psf, -1).reshape(bs, -1, 2)
            if slice_tf_params is not None:
                slice_tf_params = slice_tf_params[:, :, None].expand(-1, -1, self.n_s_psf, -1).reshape(bs, -1, 6)

        coords, tf_embed = self.slice_trafos(coords, s_idcs, slice_tf_params, inf)
        if not inf:
            coords = torch.einsum('bxy, bsy -> bsx', affines[:, :3, :3], coords) + affines[:, None, :3, 3]

        if self.tiny_model:
            coords = (coords - self.bbox[0]) / (self.bbox[1] - self.bbox[0])
            output_sr = self.sr_net(coords.reshape(-1, 3)).view(bs, coords.shape[1], -1)
        else:
            coords = (coords - self.bbox[0]) / (self.bbox[1] - self.bbox[0]) * 2 - 1 if not mni else coords
            output_sr = self.sr_net(coords.reshape(-1, 3)).view(bs, coords.shape[1], -1)

        if (self.n_s_psf > 0 and not inf) or (inf and self.n_s_psf_inf > 0):
            output_sr = output_sr.view(bs, ns, -1).mean(keepdim=True, dim=-1)

        return output_sr, tf_embed, coords

    def slice_trafos(self, coords, s_idcs, slice_tf_params, inf):
        if s_idcs is None or inf:
            embed = torch.zeros(1).to(coords.device)
        else:
            embed = self.tf_net(s_idcs.reshape(-1, 2)).view(coords.shape[0], coords.shape[1], -1) if slice_tf_params is None else slice_tf_params
            # embed[:, :, 3:] = 0
            # embed[:, :, :3] = 0
            # embed=torch.zeros(size=coords.shape[:2]+(6,)).to(coords.device)
            R, t = embed2affine(embed)
            coords = torch.einsum('bcxy,bcy->bcx', R, coords) + t
        return coords, embed

    def apply_psf(self, coords, bs, ns, inf):
        n_s_psf = self.n_s_psf if not inf else self.n_s_psf_inf
        offset = torch.randn(bs, ns, n_s_psf, 3, dtype=coords.dtype, device=coords.device)
        # coords = coords[:, :, None].expand(-1, -1, n_s_psf, -1) + offset * self.psf_stds[0:bs, None, None]
        coords = coords[:, :, None].repeat(1, 1, n_s_psf, 1) + offset * self.psf_stds[0:bs, None, None]
        coords = coords.view(bs, -1, 3)
        return coords, None

    def apply_psf_learnable(self, coords, bs, ns):
        offset = (torch.rand(bs, ns, self.n_s_psf, 3, dtype=coords.dtype, device=coords.device) * 2.5) - 1.25
        coords_psf = coords[:, :, None].expand(-1, -1, self.n_s_psf, -1) + offset
        distances = (coords_psf - coords[:, :, None, :]).view(bs, -1, 3)
        coords_psf = coords_psf.view(bs, -1, 3)
        return coords_psf, distances


class INR_SR(nn.Module):
    def __init__(self, args, bbox):
        super().__init__()
        self.args = args
        self.bbox = bbox
        self.sr_net = Siren(3, 1, args.sr_hidden_size,
                            args.sr_num_layers, first_omega_0=30, hidden_omega_0=1.75, outermost_linear=True)

    def forward(self, coords, affines=None, bbox=None, inf=False):
        bs, ns, _ = coords.shape
        coords = torch.einsum('bxy, bsy -> bsx', affines[:, :3, :3], coords) + affines[:, None, :3,
                                                                               3] if not inf else coords
        if bbox is not None:
            coords = (coords - bbox[0]) / (bbox[1] - bbox[0]) * 2 - 1
        else:
            coords = (coords - self.bbox[0]) / (self.bbox[1] - self.bbox[0]) * 2 - 1
        assert torch.all(coords <= 1.01) and torch.all(coords >= -1.01), 'coords not in [-1, 1]'
        output_sr = self.sr_net(coords)
        return output_sr
