from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from seam_carving import SeamCarver


class ResBlock(nn.Module):
    def __init__(self, channel_size: int, negative_slope: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel_size, channel_size, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(channel_size),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(channel_size, channel_size, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(channel_size)
        )

    def forward(self, x):
        return x + self.block(x)


class Resizer(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.interpolate_mode = cfg.resizer.interpolate_mode
        self.out_width = cfg.data.out_width
        self.out_height = cfg.data.out_height 

        n = cfg.resizer.num_kernels
        r = cfg.resizer.num_resblocks
        slope = cfg.resizer.negative_slope

        # Initialize the SeamCarver to resize the input image before the network processes it
        self.seam_carver = SeamCarver(cfg.resizer.in_channels, self.out_height, self.out_width)

        self.module1 = nn.Sequential(
            nn.Conv2d(cfg.resizer.in_channels, n, kernel_size=7, padding=3),
            nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(n, n, kernel_size=1),
            nn.LeakyReLU(slope, inplace=True),
            nn.BatchNorm2d(n)
        )

        resblocks = []
        for i in range(r):
            resblocks.append(ResBlock(n, slope))
        self.resblocks = nn.Sequential(*resblocks)

        self.module3 = nn.Sequential(
            nn.Conv2d(n, n, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n)
        )

        self.module4 = nn.Conv2d(n, cfg.resizer.out_channels, kernel_size=7,
                                 padding=3)

        self.interpolate = partial(F.interpolate,
                                   scale_factor=self.scale_factor,
                                   mode=self.interpolate_mode,
                                   align_corners=False,
                                   recompute_scale_factor=False)

    def forward(self, x):
        # Apply SeamCarving to resize the image before passing it through the rest of the network
        x_resized = self.seam_carver(x)

        # Continue with the rest of the Resizer model as before
        residual = self.interpolate(x_resized)

        out = self.module1(x_resized)
        out_residual = self.interpolate(out)

        out = self.resblocks(out_residual)
        out = self.module3(out)
        out = out + out_residual

        out = self.module4(out)

        out = out + residual

        return out