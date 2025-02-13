import torch
import sys
import os
"""
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/config')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/datasets')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/evaluation')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/inference')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/models')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/scripts')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/utils')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/training')))

"""
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, ROOT_DIR)
from models.common import *

class FRNet(nn.Module):
    def __init__(self, ch_in, ch_out, ls_mid_ch=([96]*6), out_k_size=33, k_size=11,
                 cls_init_block = ResidualBlock, cls_conv_block = ResidualBlock) -> None:
        super().__init__()
        self.dict_module = nn.ModuleDict()

        ch1 = ch_in
        for i in range(len(ls_mid_ch)):
            ch2 = ls_mid_ch[i]
            # self.dict_module.add_module(f"conv{i}",nn.Sequential(
            #                         ResidualBlock(ch1,ch2, k_size=1),ResidualBlock(ch2,ch2)))
            if ch1 != ch2:
                self.dict_module.add_module(f"conv{i}",cls_init_block(ch1,ch2, k_size=k_size, layer_num=i))
            else:
                if cls_conv_block == RecurrentConvNeXtBlock:
                    module = RecurrentConvNeXtBlock(dim=ch1, layer_scale_init_value=1)
                else:
                    module = cls_conv_block(ch1, ch2, k_size=k_size, layer_num=i)
                self.dict_module.add_module(f"conv{i}", module)

            # self.dict_module.add_module(f"shortcut{i}",ResidualBlock(ch1+ch2,ch2))
            ch1 = ch2

        # out_k_size = 11
        self.dict_module.add_module(f"final", nn.Sequential(
            nn.Conv2d(ch1, ch_out*4, out_k_size, padding=out_k_size//2, bias=False),
            nn.Sigmoid()
        ))
        
        self.ls_mid_ch = ls_mid_ch

    def forward(self,x):

        for i in range(len(self.ls_mid_ch)):
            conv = self.dict_module[f'conv{i}']
            x = conv(x)

        x = self.dict_module['final'](x)
        x = torch.max(x, dim=1, keepdim=True)[0]

        return  x
