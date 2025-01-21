import os
import sys
import cv2
import json
import torch
import torch.nn as nn
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, ROOT_DIR)
from models.common import *

class RoiNet(nn.Module):
    def __init__(self, ch_in, ch_out, ls_mid_ch=([32] * 6), out_k_size=11, k_size=3,
                 cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock):
        super().__init__()
        self.dict_module = nn.ModuleDict()
        ch1 = ch_in

        for i in range(len(ls_mid_ch)):
            ch2 = ls_mid_ch[i]

            # Add convolutional layers (init or conv block)
            if ch1 != ch2:
                self.dict_module.add_module(f"conv{i}", cls_init_block(ch1, ch2, k_size=k_size, layer_num=i))
            else:
                self.dict_module.add_module(f"conv{i}", cls_conv_block(ch1, ch2, k_size=k_size, layer_num=i))

            # Add pooling layers for downscaling (after Layer 1 and Layer 2)
            if i == 1:  # After Layer 1
                self.dict_module.add_module(f"pool{i}", nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),  # Halve spatial dimensions
                    nn.Conv2d(ch2, ch2 * 2, kernel_size=1)  # Double channels
                ))
                ch1 = ch2 * 2
            elif i == 2:  # After Layer 2
                self.dict_module.add_module(f"pool{i}", nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(ch2, ch2 * 2, kernel_size=1)
                ))
                ch1 = ch2 * 2

            # Add upscaling layers (after Layer 3 and Layer 4)
            elif i == 3:  # After Layer 3
                self.dict_module.add_module(f"up{i}", nn.Sequential(
                    nn.ConvTranspose2d(ch2, ch2 // 2, kernel_size=2, stride=2)  # Double spatial dimensions
                ))
                ch1 = ch2 // 2
            elif i == 4:  # After Layer 4
                self.dict_module.add_module(f"up{i}", nn.Sequential(
                    nn.ConvTranspose2d(ch2, ch2 // 2, kernel_size=2, stride=2)
                ))
                ch1 = ch2 // 2
            else:
                ch1 = ch2

        # Add final layer
        self.dict_module.add_module("final", nn.Sequential(
            nn.Conv2d(ch1, ch_out * 4, out_k_size, padding=out_k_size // 2, bias=False),
            nn.Sigmoid()
        ))

        self.ls_mid_ch = ls_mid_ch

    def forward(self, x):
        for i in range(len(self.ls_mid_ch)):
            x = self.dict_module[f"conv{i}"](x)
            if f"pool{i}" in self.dict_module:
                x = self.dict_module[f"pool{i}"](x)
            elif f"up{i}" in self.dict_module:
                x = self.dict_module[f"up{i}"](x)

        x = self.dict_module["final"](x)
        x = torch.max(x, dim=1, keepdim=True)[0]
        return x