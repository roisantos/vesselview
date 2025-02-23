import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, ROOT_DIR)
from models.common import *

class RoiNet(nn.Module):

    def __init__(self, ch_in, ch_out, ls_mid_ch=[32, 64, 128, 128, 64, 32],
                 k_size=9,
                 cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock):
        super().__init__()
        self.dict_module = nn.ModuleDict()
        ch = ch_in  # current channel count

        # ------------------ Encoder ------------------
        # Block 0: Full resolution features.
        self.dict_module.add_module("conv0", cls_init_block(ch, ls_mid_ch[0], k_size=k_size, layer_num=0))
        ch = ls_mid_ch[0]  # 32

        # Block 1: Downsample once.
        self.dict_module.add_module("conv1", cls_init_block(ch, ls_mid_ch[1], k_size=k_size, layer_num=1))
        ch = ls_mid_ch[1]  # 64
        # Downsample & double channels.
        self.dict_module.add_module("pool1", nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(ch, ch * 2, kernel_size=1)
        ))
        ch = ch * 2  # becomes 128
        # We'll save the output after pool1 as skip connection "skip1"

        # Block 2: Further encoding.
        self.dict_module.add_module("conv2", cls_init_block(ch, ls_mid_ch[2], k_size=k_size, layer_num=2))
        ch = ls_mid_ch[2]  # 128
        # Downsample & double channels.
        self.dict_module.add_module("pool2", nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(ch, ch * 2, kernel_size=1)
        ))
        ch = ch * 2  # becomes 256
        # Save output after pool2 as skip connection "skip2"

        # ------------------ Bottleneck (Deepened) ------------------
        # Add extra blocks to deepen the bottleneck.
        self.dict_module.add_module("bottle1", cls_conv_block(ch, ch, k_size=k_size, layer_num="bottle1"))
        self.dict_module.add_module("bottle2", cls_conv_block(ch, ch, k_size=k_size, layer_num="bottle2"))
        # Merge skip2 (from encoder) with the deepened bottleneck output.
        self.dict_module.add_module("merge2", nn.Sequential(
            nn.Conv2d(ch * 2, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        ))
        # After merging, our tensor will have 256 channels at 1/4 resolution.

        # ------------------ Decoder ------------------
        # Block 3: Upsample from bottleneck.
        self.dict_module.add_module("conv3", cls_init_block(ch, ls_mid_ch[3], k_size=k_size, layer_num=3))
        ch = ls_mid_ch[3]  # 128 originally, 192 in the scaled version
        self.dict_module.add_module("up3", nn.Sequential(
            nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2)
        ))
        ch = ch // 2  # becomes 64 originally, 96 in the scaled version
        # Merge with skip connection from Block 1 (pool1 output has ls_mid_ch[1]*2 channels)
        self.dict_module.add_module("merge3", nn.Sequential(
            nn.Conv2d((ls_mid_ch[3] // 2) + (ls_mid_ch[1] * 2), ls_mid_ch[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ls_mid_ch[1]),
            nn.ReLU(inplace=True)
        ))
        ch = ls_mid_ch[1]  # now set to ls_mid_ch[1]

        # Block 4: Further upsampling.
        self.dict_module.add_module("conv4", cls_init_block(ch, ls_mid_ch[4], k_size=k_size, layer_num=4))
        ch = ls_mid_ch[4]
        self.dict_module.add_module("up4", nn.Sequential(
            nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2)
        ))
        ch = ch // 2  # becomes ls_mid_ch[4]//2
        # Merge with skip connection from Block 0.
        self.dict_module.add_module("merge4", nn.Sequential(
            nn.Conv2d((ls_mid_ch[4] // 2) + ls_mid_ch[0], ls_mid_ch[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ls_mid_ch[0]),
            nn.ReLU(inplace=True)
        ))
        ch = ls_mid_ch[0]


        # ------------------ Final Classification ------------------
        # Instead of a heavy convolution with a large kernel and max over channels,
        # we simply map from 32 channels to ch_out using a 1x1 conv.
        self.dict_module.add_module("final", nn.Sequential(
            nn.Conv2d(ch, ch_out, kernel_size=1, bias=False),
            nn.Sigmoid()
        ))

        self.ls_mid_ch = ls_mid_ch

    def forward(self, x):
        # Encoder
        out0 = self.dict_module["conv0"](x)           # (B, 32, H, W)   -> skip0
        out1 = self.dict_module["conv1"](out0)          # (B, 64, H, W)
        out1 = self.dict_module["pool1"](out1)          # (B, 128, H/2, W/2) -> skip1
        skip1 = out1

        out2 = self.dict_module["conv2"](out1)          # (B, 128, H/2, W/2)
        out2 = self.dict_module["pool2"](out2)          # (B, 256, H/4, W/4) -> skip2
        skip2 = out2

        # Bottleneck (deepened)
        bottle1 = self.dict_module["bottle1"](out2)       # (B, 256, H/4, W/4)
        bottle2 = self.dict_module["bottle2"](bottle1)      # (B, 256, H/4, W/4)
        # Merge the original skip2 with the deepened features.
        bottle_cat = torch.cat([bottle2, skip2], dim=1)     # (B, 512, H/4, W/4)
        bottle_out = self.dict_module["merge2"](bottle_cat) # (B, 256, H/4, W/4)

        # Decoder
        out3 = self.dict_module["conv3"](bottle_out)       # (B, 128, H/4, W/4)
        out3 = self.dict_module["up3"](out3)               # (B, 64, H/2, W/2)
        # Merge with skip1 (from pool1)
        out3 = torch.cat([out3, skip1], dim=1)             # (B, 64+128=192, H/2, W/2)
        out3 = self.dict_module["merge3"](out3)            # (B, 64, H/2, W/2)

        out4 = self.dict_module["conv4"](out3)             # (B, 64, H/2, W/2)
        out4 = self.dict_module["up4"](out4)               # (B, 32, H, W)
        # Merge with skip0 (from conv0)
        out4 = torch.cat([out4, out0], dim=1)              # (B, 32+32=64, H, W)
        out4 = self.dict_module["merge4"](out4)            # (B, 32, H, W)

        out5 = self.dict_module["conv5"](out4)             # (B, 32, H, W)
        final = self.dict_module["final"](out5)            # (B, ch_out, H, W)
        return final