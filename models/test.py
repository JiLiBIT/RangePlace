
import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('../tools/')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
from modules.netvlad import NetVLADLoupe
from tools.read_samples import read_one_need_from_seq
import yaml
import math
def pad_to_multiples_of(tensor, n, value=0):
    """
    Pad tensor to be a multiple of n .

    Args:
        tensor: input tensor with shape [B, H, W , C]
        n: int, the multiple to pad to
        value: int, value to fill the padding

    Returns:
        padded tensor
    """
    _, h, w, _= tensor.size()
    pad_h = (n - h % n) % n
    pad_w = (n - w % n) % n
    if pad_h == 0 and pad_w == 0:
        return tensor
    else:
        return F.pad(tensor, (0, pad_h, pad_w, 0), mode = 'constant', value=value)
    
def add_to_multiples_of(x, n):
    
    x = x + (n - x % n) % n
    return x

def get_input_resolution(x, i_layer):
    for _ in range(i_layer):
        x = add_to_multiples_of(x, 2) // 2
        x = add_to_multiples_of(x, 2)
    return x

if __name__ == '__main__':
   
    x = torch.empty([1,8,113,1])
    print(x.shape)
    y = pad_to_multiples_of(x,2)
    print(y.shape)
    