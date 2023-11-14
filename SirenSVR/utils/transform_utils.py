import torch
import numpy as np
import scipy.ndimage as ndi

def embed2affine(embed):
    R = euler2rot(embed[..., :3])
    t = embed[..., 3:]
    return R, t


def euler2rot(theta):
    c1 = torch.cos(theta[:, :, 0])
    s1 = torch.sin(theta[:, :, 0])
    c2 = torch.cos(theta[:, :, 1])
    s2 = torch.sin(theta[:, :, 1])
    c3 = torch.cos(theta[:, :, 2])
    s3 = torch.sin(theta[:, :, 2])
    r11 = c1*c3 - c2*s1*s3
    r12 = -c1*s3 - c2*c3*s1
    r13 = s1*s2
    r21 = c3*s1 + c1*c2*s3
    r22 = c1*c2*c3 - s1*s3
    r23 = -c1*s2
    r31 = s2*s3
    r32 = c3*s2
    r33 = c2
    R = torch.stack([r11, r12, r13, r21, r22, r23, r31, r32, r33], dim=2)
    R = R.view(R.shape[0], R.shape[1], 3, 3)
    return R


def generate_img_grid(img_size):
    grid = torch.meshgrid([torch.arange(s) for s in img_size],
                          indexing="ij")
    grid_coords = torch.stack([g.flatten() for g in grid])
    return grid_coords.T