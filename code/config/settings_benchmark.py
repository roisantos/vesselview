"""


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/config')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/datasets')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/evaluation')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/inference')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/models')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/scripts')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/utils')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code/training')))
from models.frnet import *


class ObjectCreator:
    def __init__(self, args, cls) -> None:
        self.args = args
        self.cls_net = cls
    def __call__(self):
        return self.cls_net(**self.args)


models = {
    
    "FRNet-base": ObjectCreator(cls=FRNet, args=dict(
        ch_in=1, ch_out=1, cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock
    )),
    
    
    "RoiNet": ObjectCreator(cls=RoiNet, args=dict(
        ch_in=1, ch_out=1, ls_mid_ch=[32, 64, 128, 128, 64, 32], out_k_size=11,
        k_size=3, cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock
    )),
    
    "FRNet": ObjectCreator(cls=FRNet, args=dict(
        ch_in=1, ch_out=1, cls_init_block=RRCNNBlock, cls_conv_block=RecurrentConvNeXtBlock
    )),
    # More models can be added here......
}

"""
