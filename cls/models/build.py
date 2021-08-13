# --------------------------------------------------------
# High Resolution Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Rao Fu, RainbowSecret
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .vision_transformer import VisionTransformer
from .pvt_v2 import PyramidVisionTransformerV2
from .hrnet import HighResolutionNet
from .hrt import HighResolutionTransformer


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == "hrnet":
        model = HighResolutionNet(
            config.MODEL.HRNET, num_classes=config.MODEL.NUM_CLASSES
        )

    elif model_type == "hrt":
        model = HighResolutionTransformer(
            config.MODEL.HRT, num_classes=config.MODEL.NUM_CLASSES
        )

    elif model_type == "swin":
        model = SwinTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )

    elif model_type == "deit":
        model = VisionTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.DEIT.PATCH_SIZE,
            in_chans=config.MODEL.DEIT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.DEIT.EMBED_DIM,
            depth=config.MODEL.DEIT.DEPTHS,
            num_heads=config.MODEL.DEIT.NUM_HEADS,
            mlp_ratio=config.MODEL.DEIT.MLP_RATIO,
            qkv_bias=config.MODEL.DEIT.QKV_BIAS,
            qk_scale=config.MODEL.DEIT.QK_SCALE,
            representation_size=config.MODEL.DEIT.REPRESENTATION_SIZE,
            drop_rate=config.MODEL.DEIT.DROP_RATE,
            attn_drop_rate=config.MODEL.DEIT.ATTN_DROP_RATE,
            drop_path_rate=config.MODEL.DEIT.DROP_PATH_RATE,
            hybrid_backbone=config.MODEL.DEIT.HYBRID_BACKBONE,
            norm_layer=config.MODEL.DEIT.NORM_LAYER,
            patch_norm_layer=config.MODEL.DEIT.PATCH_NORM_LAYER,
        )

    elif model_type == "pvt_v2":
        model = PyramidVisionTransformerV2(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.PVT.PATCH_SIZE,
            in_chans=config.MODEL.PVT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dims=config.MODEL.PVT.EMBED_DIMS,
            num_heads=config.MODEL.PVT.NUM_HEADS,
            mlp_ratios=config.MODEL.PVT.MLP_RATIOS,
            qkv_bias=config.MODEL.PVT.QKV_BIAS,
            qk_scale=config.MODEL.PVT.QK_SCALE,
            drop_rate=config.MODEL.PVT.DROP_RATE,
            attn_drop_rate=config.MODEL.PVT.ATTN_DROP_RATE,
            drop_path_rate=config.MODEL.PVT.DROP_PATH_RATE,
            depths=config.MODEL.PVT.DEPTHS,
            sr_ratios=config.MODEL.PVT.SR_RATIOS,
            num_stages=config.MODEL.PVT.NUM_STAGES,
            linear=config.MODEL.PVT.LINEAR,
        )

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    print(model)
    return model
