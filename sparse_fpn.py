import torch
import torch.nn as nn
import spconv.pytorch as spconv
from collections import OrderedDict
from functools import partial
import os
import math


SPCONV_ALGO = 'auto'    # 'auto', 'implicit_gemm', 'native'
env_spconv_algo = os.environ.get('SPCONV_ALGO')
if env_spconv_algo is not None and env_spconv_algo in ['auto', 'implicit_gemm', 'native']:
    SPCONV_ALGO = env_spconv_algo
import torch

# This implementation is GPU-only. Require CUDA to avoid CPU fallbacks.
if not torch.cuda.is_available():
    raise RuntimeError('Res16FPNBase requires CUDA. Please run on a GPU.')

algo = None
if SPCONV_ALGO == 'native':
    algo = spconv.ConvAlgo.Native
elif SPCONV_ALGO == 'implicit_gemm':
    algo = spconv.ConvAlgo.MaskImplicitGemm
else:
    # auto on CUDA: prefer MaskImplicitGemm for best performance
    algo = spconv.ConvAlgo.MaskImplicitGemm



class BasicBlock(spconv.SparseModule):
    expansion = 1
    def __init__(self, in_channels, embed_channels, stride=1, norm_fn=None, indice_key=None, bias=False):
        super().__init__()

        assert norm_fn is not None

        if in_channels == embed_channels:
            self.proj = spconv.SparseSequential(nn.Identity())
        else:
            self.proj = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, embed_channels, kernel_size=1, bias=False),
                norm_fn(embed_channels, momentum=0.02))

        self.conv1 = spconv.SubMConv3d(
            in_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key, algo=algo)

        self.bn1 = norm_fn(embed_channels)
        self.relu = nn.LeakyReLU()
        self.conv2 = spconv.SubMConv3d(
            embed_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key, algo=algo)

        self.bn2 = norm_fn(embed_channels)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        out = out.replace_feature(out.features + self.proj(residual).features)
        out = out.replace_feature(self.relu(out.features))

        return out



class Res16FPNBase(nn.Module):
    channels = (32, 64, 128, 256, 256, 256, 256, 256)
    layers = (2, 2, 2, 2, 2, 2, 2, 2)
    bn_momentum = 0.02

    def __init__(self, in_channels, base_channels=32):
        super().__init__()
        assert len(self.layers) % 2 == 0
        assert len(self.layers) == len(self.channels)
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = len(self.layers) // 2

        norm_fn = partial(nn.BatchNorm1d, eps=1e-5, momentum=self.bn_momentum)
        block = BasicBlock

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                base_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem"),
            norm_fn(base_channels, momentum=0.02),
            nn.LeakyReLU())

        enc_channels = base_channels
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.enc = nn.ModuleList()

        for s in range(self.num_stages):
            # encode num_stages
            self.down.append(spconv.SparseSequential(spconv.SparseConv3d(
                        enc_channels,
                        self.channels[s],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f"spconv{s + 1}", algo=algo),
                    norm_fn(self.channels[s], momentum=self.bn_momentum),
                    nn.LeakyReLU()))

            self.enc.append(spconv.SparseSequential(OrderedDict([(f"block{i}",
                                block(self.channels[s],
                                    self.channels[s],
                                    norm_fn=norm_fn,
                                    indice_key=f"subm{s + 1}"))
                            for i in range(self.layers[s])])))

            # decode num_stages: create inverse conv (upsample) modules like SpUNet
            self.up.append(spconv.SparseSequential(spconv.SparseInverseConv3d(
                        128,
                        128,
                        kernel_size=2,
                        bias=False,
                        indice_key=f"spconv{s + 1}", algo=algo),
                    norm_fn(128, momentum=self.bn_momentum),
                    nn.LeakyReLU()))

            enc_channels = self.channels[s]

        self.delayer4 = nn.Linear(256, 128, bias=False)
        self.delayer3 = nn.Linear(128, 128, bias=False)
        self.delayer2 = nn.Linear(64, 128, bias=False)
        self.delayer1 = nn.Linear(32, 128, bias=False)
        # self.head = spconv.SubMConv3d(128, 384, kernel_size=1, padding=1, bias=True, algo=algo)
        self.final = nn.Linear(128, 384)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, y):
        x = self.conv_input(y)

        x1 = self.down[0](x)
        x1 = self.enc[0](x1) ##32

        x2 = self.down[1](x1)
        x2 = self.enc[1](x2) ##64

        x3 = self.down[2](x2)
        x3 = self.enc[2](x3) ##128

        x4 = self.down[3](x3)
        x4 = self.enc[3](x4) ##256

        # dec forward
        x4 = x4.replace_feature(self.delayer4(x4.features))
        up4 = self.up[3](x4)

        x3 = x3.replace_feature(self.delayer3(x3.features))
        x3 = x3.replace_feature(x3.features + up4.features)
        up3 = self.up[2](x3)

        x2 = x2.replace_feature(self.delayer2(x2.features))
        x2 = x2.replace_feature(x2.features + up3.features)
        up2 = self.up[1](x2)

        x1 = x1.replace_feature(self.delayer1(x1.features))
        x1 = x1.replace_feature(x1.features + up2.features)
        up1 = self.up[0](x1)

        out = up1
        out = self.final(out.features)
        return out



class Res16FPN14(Res16FPNBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class Res16FPN18(Res16FPNBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class Res16FPN34(Res16FPNBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class Res16FPN14A(Res16FPN14):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16FPN14A2(Res16FPN14A):
    LAYERS = (1, 1, 1, 1, 2, 2, 2, 2)


class Res16FPN14B(Res16FPN14):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16FPN14B2(Res16FPN14B):
    LAYERS = (1, 1, 1, 1, 2, 2, 2, 2)


class Res16FPN14B3(Res16FPN14B):
    LAYERS = (2, 2, 2, 2, 1, 1, 1, 1)


class Res16FPN14C(Res16FPN14):
    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class Res16FPN14D(Res16FPN14):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16FPN18A(Res16FPN18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16FPN18B(Res16FPN18):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16FPN18D(Res16FPN18):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16FPN32B(Res16FPN34):
    PLANES = (32, 64, 128, 256, 256, 64, 64, 64)


class Res16FPN34A(Res16FPN34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class Res16FPN34B(Res16FPN34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class Res16FPN34C(Res16FPN34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)