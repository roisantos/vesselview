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
                 out_k_size=25, k_size=9,
                 cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock):
        super().__init__()
        self.dict_module = nn.ModuleDict()
        ch1 = ch_in

        # Build the encoder and decoder blocks.
        for i in range(len(ls_mid_ch)):
            ch2 = ls_mid_ch[i]
            # Add convolutional block: use an "init" block if channel size changes.
            if ch1 != ch2:
                self.dict_module.add_module(f"conv{i}", cls_init_block(ch1, ch2, k_size=k_size, layer_num=i))
            else:
                self.dict_module.add_module(f"conv{i}", cls_conv_block(ch1, ch2, k_size=k_size, layer_num=i))
            
            # For blocks 1 and 2, add pooling to downsample and double the channels.
            if i == 1:
                self.dict_module.add_module(f"pool{i}", nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(ch2, ch2 * 2, kernel_size=1)  # doubles channels: 64 -> 128
                ))
                ch1 = ch2 * 2  # update channels after pooling
            elif i == 2:
                self.dict_module.add_module(f"pool{i}", nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(ch2, ch2 * 2, kernel_size=1)  # doubles channels: 128 -> 256
                ))
                ch1 = ch2 * 2
            # For blocks 3 and 4, add upsampling layers to recover spatial resolution.
            elif i == 3:
                self.dict_module.add_module(f"up{i}", nn.Sequential(
                    nn.ConvTranspose2d(ch2, ch2 // 2, kernel_size=2, stride=2)  # halves channels: 128 -> 64
                ))
                ch1 = ch2 // 2
            elif i == 4:
                self.dict_module.add_module(f"up{i}", nn.Sequential(
                    nn.ConvTranspose2d(ch2, ch2 // 2, kernel_size=2, stride=2)  # halves channels: 64 -> 32
                ))
                ch1 = ch2 // 2
            else:
                ch1 = ch2  # no pooling/updating channels

        # Final classification layer.
        self.dict_module.add_module("final", nn.Sequential(
            nn.Conv2d(ch1, ch_out * 4, out_k_size, padding=out_k_size // 2, bias=False),
            nn.Sigmoid()
        ))
        
        # --- Add merge layers for skip connections ---
        # For the first skip: after upsampling in block 3 we are at half resolution.
        # Skip connection from block 1: after pooling, block 1 yields channels=64*2 = 128.
        # Up3 output (after conv3 + up3) has 64 channels.
        # Merge by concatenating (64+128=192) then reducing to 64 channels.
        self.dict_module.add_module("merge3", nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ))
        # For the second skip: after upsampling in block 4 we are at full resolution.
        # Skip connection from block 0 (conv0) yields 32 channels.
        # Up4 output (after conv4 + up4) has 32 channels.
        # Merge by concatenating (32+32=64) then reducing to 32 channels.
        self.dict_module.add_module("merge4", nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        ))
        
        self.ls_mid_ch = ls_mid_ch

    def forward(self, x):
        # ----- Encoder -----
        # Block 0: full resolution.
        out0 = self.dict_module["conv0"](x)  # Shape: (B, 32, H, W)

        # Block 1: conv then pool. (Skip connection #1 will be taken after pooling.)
        out1 = self.dict_module["conv1"](out0)
        if "pool1" in self.dict_module:
            out1 = self.dict_module["pool1"](out1)  # Now: (B, 128, H/2, W/2)
        skip1 = out1  # store skip connection from encoder level 1.

        # Block 2: bottleneck part of the encoder.
        out2 = self.dict_module["conv2"](out1)
        if "pool2" in self.dict_module:
            out2 = self.dict_module["pool2"](out2)  # Now: (B, 256, H/4, W/4)

        # ----- Decoder -----
        # Block 3: begin decoding.
        out3 = self.dict_module["conv3"](out2)
        if "up3" in self.dict_module:
            out3 = self.dict_module["up3"](out3)  # Upsample to: (B, 128//2=64, H/2, W/2)
        # Merge with skip connection from block 1.
        out3 = torch.cat([out3, skip1], dim=1)  # (B, 64 + 128 = 192, H/2, W/2)
        out3 = self.dict_module["merge3"](out3)   # Fused to 64 channels.

        # Block 4: continue decoding.
        out4 = self.dict_module["conv4"](out3)
        if "up4" in self.dict_module:
            out4 = self.dict_module["up4"](out4)  # Upsample to: (B, 64//2=32, H, W)
        # Merge with skip connection from block 0.
        out4 = torch.cat([out4, out0], dim=1)  # (B, 32 + 32 = 64, H, W)
        out4 = self.dict_module["merge4"](out4)   # Fused to 32 channels.

        # Block 5: final conv block.
        out5 = self.dict_module["conv5"](out4)
        final = self.dict_module["final"](out5)
        final = torch.max(final, dim=1, keepdim=True)[0]
        return final
