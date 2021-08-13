import os
import pdb
import math
import logging
import torch
import torch.nn as nn
from functools import partial

from .multihead_isa_pool_attention import InterlacedPoolAttention
from .ffn_block import MlpDWBN


BN_MOMENTUM = 0.1


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return "drop_prob={}".format(self.drop_prob)


class GeneralTransformerBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super(GeneralTransformerBlock, self).__init__()
        self.dim = inplanes
        self.out_dim = planes
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.attn = InterlacedPoolAttention(
            self.dim,
            num_heads=num_heads,
            window_size=window_size,
            rpe=True,
            dropout=attn_drop,
        )

        self.norm1 = norm_layer(self.dim)
        self.norm2 = norm_layer(self.out_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(self.dim * mlp_ratio)

        self.mlp = MlpDWBN(
            in_features=self.dim,
            hidden_features=mlp_hidden_dim,
            out_features=self.out_dim,
            act_layer=act_layer,
            dw_act_layer=act_layer,
            drop=drop,
        )
 
    def forward(self, x, mask=None):
        B, C, H, W = x.size()
        # reshape
        x = x.view(B, C, -1).permute(0, 2, 1)
        # Attention
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        # reshape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return "num_heads={}, window_size={}, mlp_ratio={}".format(
            self.num_heads, self.window_size, self.mlp_ratio
        )
