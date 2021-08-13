##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Lang Huang
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper


class SwinUper(nn.Module):
    def __init__(self, configer):
        super(SwinUper, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = BackboneSelector(configer).get_backbone()
        num_channels = self.backbone.num_channels

        # extra added layers
        from lib.models.modules.upernet_block import UPerNetModule

        self.uper_head = UPerNetModule(
            fc_dim=num_channels[-1],
            pool_scales=(1, 2, 3, 6),
            fpn_inplanes=num_channels,
            fpn_dim=512,
            bn_type=self.configer.get("network", "bn_type"),
        )
        self.cls_head = nn.Sequential(
            nn.Dropout2d(0.10),
            nn.Conv2d(
                512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
            ),
        )
        self.aux_head = nn.Sequential(
            nn.Conv2d(
                num_channels[-2], 256, kernel_size=3, stride=1, padding=1, bias=False
            ),
            ModuleHelper.BNReLU(256, bn_type=self.configer.get("network", "bn_type")),
            nn.Dropout2d(0.10),
            nn.Conv2d(
                256, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
            ),
        )

    def forward(self, x_):
        # extract features
        x = self.backbone(x_)

        # uper head
        out = self.uper_head(x)
        out = self.cls_head(out)
        out = F.interpolate(
            out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )

        # aux head
        out_aux = self.aux_head(x[-2])
        out_aux = F.interpolate(
            out_aux, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )

        return out_aux, out
