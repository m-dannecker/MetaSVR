import torch.nn as nn
import torch
import numpy as np
from utils.transform_utils import *
import math
#from torchmeta.modules import (MetaModule, MetaSequential, MetaLinear)
from collections import OrderedDict


# class BatchLinear(nn.Linear, MetaModule):
#     '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
#     hypernetwork.'''
#     __doc__ = nn.Linear.__doc__
#
#     def forward(self, input, params=None):
#         if params is None:
#             params = OrderedDict(self.named_parameters())
#
#         bias = params.get('bias', None)
#         weight = params['weight']
#
#         output = input.matmul(weight.permute(*[i for i in range(len(weight.shape)-2)], -1, -2))
#         output += bias.unsqueeze(-2)
#         return output
# class SineLayerMeta(MetaModule):
#     # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
#
#     # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
#     # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
#     # hyperparameter.
#
#     # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
#     # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
#
#     def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
#         super().__init__()
#         self.omega_0 = float(omega_0)
#
#         self.is_first = is_first
#         self.in_features = in_features
#         self.linear = BatchLinear(in_features, out_features, bias=bias)
#         self.init_weights()
#
#     def init_weights(self):
#         with torch.no_grad():
#             if self.is_first:
#                 self.linear.weight.uniform_(-1 / self.in_features,
#                                             1 / self.in_features)
#             else:
#                 self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
#                                             np.sqrt(6 / self.in_features) / self.omega_0)
#
#     def forward(self, input, params=None):
#         intermed = self.linear(input, params=self.get_subdict(params, 'linear'))
#         return torch.sin(self.omega_0 * intermed)
# class SineLayerMeta_LayerNorm(MetaModule):
#     # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
#
#     # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
#     # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
#     # hyperparameter.
#
#     # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
#     # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
#
#     def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
#         super().__init__()
#         self.omega_0 = omega_0
#         self.is_first = is_first
#         self.in_features = in_features
#         self.ln = nn.LayerNorm(out_features, elementwise_affine=False)
#         self.linear = BatchLinear(in_features, out_features, bias=bias)
#         self.init_weights()
#
#     def init_weights(self):
#         with torch.no_grad():
#             if self.is_first:
#                 self.linear.weight.uniform_(-1 / self.in_features,
#                                             1 / self.in_features)
#             else:
#                 self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / 30,
#                                             np.sqrt(6 / self.in_features) / 30)
#
#     def forward(self, input, params=None):
#         intermed = self.linear(input, params=self.get_subdict(params, 'linear'))
#         intermed = self.ln(intermed)
#         return torch.sin(self.omega_0 * intermed)
# class SirenMeta(MetaModule):
#     def __init__(self, in_size, out_size, hidden_size, num_layers, first_omega_0, hidden_omega_0, outermost_linear, noFirst=False):
#         super().__init__()
#
#         self.net = [SineLayerMeta_LayerNorm(in_size, hidden_size, is_first=not noFirst, omega_0=first_omega_0)]
#         self.hidden_size = hidden_size
#         for i in range(num_layers):
#             self.net.append(SineLayerMeta_LayerNorm(hidden_size, hidden_size, is_first=False, omega_0=hidden_omega_0))
#
#         if outermost_linear:
#             final_linear = BatchLinear(hidden_size, out_size)
#             with torch.no_grad():
#                 final_linear.weight.uniform_(-np.sqrt(6 / hidden_size) / 30,
#                                              np.sqrt(6 / hidden_size) / 30)
#             self.net.append(final_linear)
#         else:
#             self.net.append(SineLayerMeta_LayerNorm(hidden_size, out_size, is_first=False, omega_0=hidden_omega_0))
#
#         self.net = nn.ModuleList(self.net)
#
#     def forward(self, x, params=None):
#         for i, layer in enumerate(self.net):
#             x = layer(x, params=self.get_subdict(params, f'sr_net.net.{i}'))
#         return x
#
# class SirenSingleHeadMeta(nn.Module):
#     def __init__(self, in_size, out_size, hidden_size, hidden_size_heads, num_layers, num_layers_heads,
#                  first_omega_0, hidden_omega_0, first_omega_0_head, hidden_omega0_head):
#         super().__init__()
#         self.in_size = in_size
#         self.hidden_size = hidden_size
#         self.hidden_size_heads = hidden_size_heads
#         self.num_layers_heads = num_layers_heads
#         self.first_omega_0_head = first_omega_0_head
#         self.hidden_omega0_head = hidden_omega0_head
#         self.head = Siren(hidden_size_heads, out_size, hidden_size_heads, num_layers_heads-1,  # -1 because the first layer is already in main_net
#                           first_omega_0=first_omega_0_head, hidden_omega_0=hidden_omega0_head,
#                           outermost_linear=True, noFirst=False)
#         self.main_net = Siren(in_size, hidden_size_heads, hidden_size, num_layers,
#                               first_omega_0=first_omega_0, hidden_omega_0=hidden_omega_0,
#                               outermost_linear=False, noFirst=False)
#
#     def forward(self, coords):
#         intermediate = self.main_net(coords)
#         output = self.head(intermediate)
#         return output
#
#     def init_heads(self):
#         # del self.head
#         # self.head = Siren(self.in_size, self.hidden_size_heads, self.hidden_size_heads, self.num_layers_heads-1, # -1 because the first layer is already in main_net
#         #                   first_omega_0=self.first_omega_0_head, hidden_omega_0=self.hidden_omega0_head,
#         #                   outermost_linear=False, noFirst=False).cuda(0).to(torch.float32)
#         for l in self.head.net:
#             # if l is nn.Linear:
#             if isinstance(l, nn.Linear):
#                 with torch.no_grad():
#                     l.weight.uniform_(-np.sqrt(6 / self.head.hidden_size) / 30, np.sqrt(6 / self.head.hidden_size) / 30)
#                     stdv = 1. / math.sqrt(l.weight.size(1))
#                     l.bias.data.uniform_(-stdv, stdv)
#             else:
#                 l.init_weights()
#                 stdv = 1. / math.sqrt(l.linear.weight.size(1))
#                 l.linear.bias.data.uniform_(-stdv, stdv)

class SineLayer_LayerNorm(nn.Module):
    # adapted and modified from Sitzmann et al. 2020, Implicit Neural Representations
    # with Periodic Activation Functions
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30, normalize=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.ln = nn.LayerNorm(out_features, elementwise_affine=False) if normalize else nn.Identity()
        self.bn = nn.BatchNorm1d(out_features, affine=False) if normalize else nn.Identity()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / 30,
                                            np.sqrt(6 / self.in_features) / 30)

    def forward(self, input):
        intermed = self.linear(input)
        intermed = self.ln(intermed)
        # intermed = self.bn(intermed)
        # intermed = (intermed-intermed[:, :, 0:1].mean(dim=-2, keepdim=True)) * intermed[:, :, 0:1].var(dim=-2, keepdim=True)
        return torch.sin(self.omega_0 * intermed)


class Siren(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, num_layers, first_omega_0, hidden_omega_0, outermost_linear,
                 normalize=False):
        super().__init__()
        self.net = [SineLayer_LayerNorm(in_size, hidden_size, is_first=True, omega_0=first_omega_0, normalize=normalize)]
        self.hidden_size = hidden_size
        for i in range(num_layers):
            self.net.append(SineLayer_LayerNorm(hidden_size, hidden_size, is_first=False, omega_0=hidden_omega_0, normalize=normalize))

        if outermost_linear:
            final_linear = nn.Linear(hidden_size, out_size)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_size) / 30,
                                             np.sqrt(6 / hidden_size) / 30)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer_LayerNorm(hidden_size, out_size, is_first=False, omega_0=hidden_omega_0, normalize=normalize))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output


class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, num_layers):
        super().__init__()

        self.net = [nn.Linear(in_size, hidden_size, bias=True), nn.ReLU()]
        self.hidden_size = hidden_size
        for i in range(num_layers):
            self.net.append(nn.Linear(hidden_size, hidden_size, bias=True))
            self.net.append(nn.ReLU())

        self.net.append(nn.Linear(hidden_size, out_size, bias=True))
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output


class SirenMultiHead(nn.Module):
    def __init__(self, in_size, out_size, num_heads, hidden_size, hidden_size_heads, num_layers, num_layers_heads,
                 first_omega_0, hidden_omega_0, first_omega_0_head, hidden_omega0_head, outermost_linear, heads2tails):
        super().__init__()

        main_net_in_size = hidden_size_heads if heads2tails else in_size
        main_net_out_size = out_size if heads2tails else hidden_size_heads
        self.heads_in_size = in_size if heads2tails else hidden_size_heads
        self.heads_out_size = hidden_size_heads if heads2tails else out_size
        self.heads_first_omega = first_omega_0_head if heads2tails else first_omega_0_head
        self.hidden_omega_0 = hidden_omega_0
        self.num_heads = num_heads
        self.hidden_size_heads = hidden_size_heads
        self.num_layers_heads = num_layers_heads
        main_net_first_omega = hidden_omega_0 if heads2tails else first_omega_0

        self.heads2tails = heads2tails
        self.main_net = Siren(main_net_in_size, main_net_out_size, hidden_size, num_layers,
                              first_omega_0=main_net_first_omega, hidden_omega_0=hidden_omega_0, outermost_linear=heads2tails)
        self.heads = nn.ModuleList([Siren(self.heads_in_size, self.heads_out_size, self.hidden_size_heads, self.num_layers_heads-1, # -1 because the first layer is already in main_net
                                          first_omega_0=self.heads_first_omega, hidden_omega_0=hidden_omega0_head,
                                          outermost_linear=not self.heads2tails, noFirst=True) for i in range(self.num_heads)])

    def forward(self, coords):
        if self.heads2tails:
            intermediates = [head(coords[i]) for i, head in enumerate(self.heads)]
            output = self.main_net(torch.stack(intermediates, dim=0))
        else:
            intermediate = self.main_net(coords)
            output = torch.stack([head(intermediate[i]) for i, head in enumerate(self.heads)], dim=0)
        return output

    # def init_heads(self):
    #     self.heads = nn.ModuleList([Siren(self.heads_in_size, self.heads_out_size, self.hidden_size_heads,
    #                                       self.num_layers_heads - 1,
    #                                       # -1 because the first layer is already in main_net
    #                                       first_omega_0=self.heads_first_omega, hidden_omega_0=self.hidden_omega_0,
    #                                       outermost_linear=not self.heads2tails) for i in range(self.num_heads)]).cuda(0)

    def init_heads(self):
        for head in self.heads:
            for l in head.net:
                # if l is nn.Linear:
                if isinstance(l, nn.Linear):
                    with torch.no_grad():
                        l.weight.uniform_(-np.sqrt(6 / head.hidden_size) / 30, np.sqrt(6 / head.hidden_size) / 30)
                else:
                    l.init_weights()





