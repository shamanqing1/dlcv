# pylint: disable=arguments-differ, missing-docstring
"""Encoder and Decoder functions.
Encoders are used during training, which assign training targets.
Decoders are used during testing/validation, which convert predictions back to
normal boxes, etc.
"""
from __future__ import absolute_import

import numpy as np
import torch
from torch import nn

class CenterNetDecoder(nn.Module):
    """Decorder for centernet.

    Parameters
    ----------
    topk : int
        Only keep `topk` results.
    scale : float, default is 4.0
        Downsampling scale for the network.

    """
    def __init__(self, topk=100, scale=4.0):
        super(CenterNetDecoder, self).__init__()
        self._topk = topk
        self._scale = scale

    def forward(self, x, wh, reg):
        """Forward of decoder"""
        batch_size, _, out_h, out_w = x.size()
        scores, indices = x.view(batch_size, -1).topk(self._topk)
        topk_classes = indices / (out_h * out_w)
        topk_indices = indices % (out_w * out_w)
        topk_ys = topk_indices / out_w
        topk_xs = topk_indices % out_w
        center = reg.permute(0, 2, 3, 1).view(batch_size, -1, 2)
        wh = wh.permute(0, 2, 3, 1).view(batch_size, -1, 2)
        batch_indices = torch.arange(batch_size).unsqueeze(-1).repeat(1, self._topk)
        reg_xs_indices = torch.zeros_like(batch_indices, dtype=torch.int64)
        reg_ys_indices = torch.ones_like(batch_indices, dtype=torch.int64)
        xs = center[[batch_indices, topk_indices, reg_xs_indices]]
        ys = center[[batch_indices, topk_indices, reg_ys_indices]]
        topk_xs = topk_xs + xs
        topk_ys = topk_ys + ys
        w = wh[[batch_indices, topk_indices, reg_xs_indices]]
        h = wh[[batch_indices, topk_indices, reg_ys_indices]]
        half_w = w / 2
        half_h = h / 2
        results = torch.stack([topk_xs - half_w, topk_ys - half_h, topk_xs + half_w, topk_ys + half_h], dim=-1)
        return topk_classes, scores, results * self._scale
