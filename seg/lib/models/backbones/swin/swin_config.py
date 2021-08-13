# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN


# basic configs for SwinTransformer
Swin = CN()
# global configs
Swin.IN_CHANS = 3
Swin.PATCH_SIZE = 4
Swin.PATCH_NORM = True
Swin.DROP_RATE = 0
Swin.ATTN_DROP_RATE = 0
Swin.DROP_PATH_RATE = 0.3
Swin.QKV_BIAS = True
Swin.QKV_SCALE = None
# layer-wise configs
Swin.NUM_BLOCKS = [2, 2, 6, 2]
Swin.NUM_CHANNELS = [96, 192, 384, 768]
Swin.NUM_HEADS = [3, 6, 12, 24]
Swin.NUM_MLP_RATIOS = [4, 4, 4, 4]
Swin.NUM_WINDOW_SIZES = [[7, 7], [7, 7], [7, 7, 7, 7, 7, 7], [7, 7]]
Swin.NUM_RESOLUTIONS = [[56, 56], [28, 28], [14, 14], [7, 7]]
Swin.BLOCK_TYPES = [
    "TRANSFORMER_BLOCK",
    "TRANSFORMER_BLOCK",
    "TRANSFORMER_BLOCK",
    "TRANSFORMER_BLOCK",
]
Swin.ATTN_TYPES = [
    ["msw", "shift_msw"],
    ["msw", "shift_msw"],
    ["msw", "shift_msw", "msw", "shift_msw", "msw", "shift_msw"],
    ["msw", "shift_msw"],
]
Swin.FFN_TYPES = [
    ["mlp", "mlp"],
    ["mlp", "mlp"],
    ["mlp", "mlp", "mlp", "mlp", "mlp", "mlp"],
    ["mlp", "mlp"],
]


# configs for Swin-Tiny
SwinTiny = Swin.clone()


# configs for Swin-Small
SwinSmall = Swin.clone()
SwinSmall.NUM_BLOCKS = [2, 2, 18, 2]


# configs for Swin-Base
SwinBase = Swin.clone()
SwinBase.NUM_BLOCKS = [2, 2, 18, 2]
SwinBase.NUM_CHANNELS = [192, 384, 768, 1536]
SwinBase.NUM_HEADS = [4, 8, 16, 32]


# configs for Swin-Large
SwinLarge = Swin.clone()
SwinLarge.NUM_BLOCKS = [2, 2, 18, 2]
SwinLarge.NUM_CHANNELS = [128, 256, 512, 1024]
SwinLarge.NUM_HEADS = [6, 12, 24, 48]


# model configs dict
MODEL_CONFIGS = {
    "swin_tiny": SwinTiny,
    "swin_small": SwinSmall,
    "swin_base": SwinBase,
    "swin_large": SwinLarge,
}
