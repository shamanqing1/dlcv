"""ResNet with Deconvolution layers for CenterNet object detection."""
# pylint: disable=unused-argument
from __future__ import absolute_import

import warnings
import math

import torch
from torch import nn
from torchvision import models
from ..model_zoo import get_model

__all__ = ['DeconvResnet', 'get_deconv_resnet',
           'resnet18_v1b_deconv', 'resnet18_v1b_deconv_dcnv2',
           'resnet50_v1b_deconv', 'resnet50_v1b_deconv_dcnv2',
           'resnet101_v1b_deconv', 'resnet101_v1b_deconv_dcnv2']

class DeconvResnet(nn.Module):
    """Deconvolutional ResNet.
    base_network : str
        Name of the base feature extraction network.
    deconv_filters : list of int
        Number of filters for deconv layers.
    deconv_kernels : list of int
        Kernel sizes for deconv layers.
    pretrained_base : bool
        Whether load pretrained base network.
    norm_layer : torch.nn.Module
        Type of Norm layers, can be BatchNorm2d, SyncBatchNorm, GroupNorm, etc.
    norm_kwargs : dict
        Additional kwargs for `norm_layer`.
    use_dcnv2 : bool
        If true, will use DCNv2 layers in upsampling blocks
    """
    def __init__(self, base_network='resnet18_v1b', deconv_filters=(256, 128, 64),
                 deconv_kernels=(4, 4, 4), pretrained_base=True, norm_layer=nn.BatchNorm2d,
                 norm_kwargs=None, use_dcnv2=False, **kwargs):
        super(DeconvResnet, self).__init__(**kwargs)
        assert 'resnet' in base_network
        net = get_model(base_network, pretrained=pretrained_base)
        # net = models.resnet18(pretrained=pretrained_base)
        self._norm_layer = norm_layer
        self._norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        self._use_dconv2 = use_dcnv2
        if 'v1b' in base_network:
            feat = nn.Sequential(
                net.conv1,
                net.bn1,
                net.relu,
                net.maxpool,
                net.layer1,
                net.layer2,
                net.layer3,
                net.layer4
            )
            self.base_network = feat
        else:
            raise NotImplementedError
        self.deconv = self._make_deconv_layer(deconv_filters, deconv_kernels)

    def _get_deconv_cfg(self, deconv_kernel):
        """Get the deconv configs using presets"""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError('Unsupported deconvolution kernel: {}'.format(deconv_kernel))

        return deconv_kernel, padding, output_padding    
    
    def _make_deconv_layer(self, num_filters, num_kernels):
        # pylint: disable=unused-variable
        """Make deconv layers using the configs"""
        assert len(num_kernels) == len(num_filters), \
            'Deconv filters and kernels number mismatch: {} vs. {}'.format(
                len(num_filters), len(num_kernels))
        
        layers = nn.Sequential()
        in_planes = self.base_network(torch.zeros((1, 3, 256, 256))).size(1)

        for i, (planes, k) in enumerate(zip(num_filters, num_kernels)):
            kernel, padding, output_padding = self._get_deconv_cfg(k)
            if self._use_dconv2:
                raise NotImplementedError
            else:
                conv = nn.Conv2d(in_planes, planes, 1, stride=1, padding=1)
            deconv = nn.ConvTranspose2d(planes, planes, kernel, stride=2, padding=padding, 
                                        output_padding=output_padding, bias=False)
            # TODO BilinearUpSample() for deconv
            upsample = nn.Sequential(
                conv,
                self._norm_layer(planes, momentum=0.9, **self._norm_kwargs),
                nn.ReLU(),
                deconv,
                self._norm_layer(planes, momentum=0.9, **self._norm_kwargs),
                nn.ReLU()
            )

            layers.add_module('upsample_stage{}'.format(i), upsample)
            in_planes = planes

        return layers
    
    def forward(self, x):
        # pylint: disable=arguments-differ
        out = self.base_network(x)

        # print(out.size())
        # for layer in self.deconv:
        #     out = layer(out)
        #     print(out.size())
        out = self.deconv(out)
        return out





def get_deconv_resnet(base_network, pretrained=False, device=torch.device('cpu'), use_dcnv2=False, **kwargs):
    """Get resnet with deconv layers.

    Parameters
    ----------
    base_network : str
        Name of the base feature extraction network.
    pretrained : bool
        Whether load pretrained base network.
    device : torch.Device
        torch.device('cpu') or torch.device('cuda')
    use_dcnv2 : bool
        If true, will use DCNv2 layers in upsampling blocks
    pretrained : type
        Description of parameter `pretrained`.
    Returns
    -------
    get_deconv_resnet(base_network, pretrained=False,
        Description of returned object.

    """
    net = DeconvResnet(base_network=base_network, pretrained_base=pretrained,
                       use_dcnv2=use_dcnv2, **kwargs)
    net.to(device)
    return net


def resnet18_v1b_deconv(**kwargs):
    """Resnet18 v1b model with deconv layers.

    Returns
    -------
    torch.nn.Module
        A Resnet18 v1b model with deconv layers.

    """

    kwargs['use_dcnv2'] = False
    return get_deconv_resnet('resnet18_v1b', **kwargs)