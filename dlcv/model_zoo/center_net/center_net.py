"""CenterNet object detector: Objects as Points, https://arxiv.org/abs/1904.07850"""
from __future__ import absolute_import

import os
import warnings
from collections import OrderedDict
from functools import partial

import torch
from torch import nn
from torchvision import models

from ...nn.coder import CenterNetDecoder

# __all__ = ['CenterNet', 'get_center_net',
#            'center_net_resnet18_v1b_voc', 'center_net_resnet18_v1b_dcnv2_voc',
#            'center_net_resnet18_v1b_coco', 'center_net_resnet18_v1b_dcnv2_coco',
#            'center_net_resnet50_v1b_voc', 'center_net_resnet50_v1b_dcnv2_voc',
#            'center_net_resnet50_v1b_coco', 'center_net_resnet50_v1b_dcnv2_coco',
#            'center_net_resnet101_v1b_voc', 'center_net_resnet101_v1b_dcnv2_voc',
#            'center_net_resnet101_v1b_coco', 'center_net_resnet101_v1b_dcnv2_coco',
#            'center_net_dla34_voc', 'center_net_dla34_dcnv2_voc',
#            'center_net_dla34_coco', 'center_net_dla34_dcnv2_coco',
#            ]

__all__ = ['CenterNet', 'get_center_net', 'center_net_resnet18_v1b_voc']




class CenterNet(nn.Module):
    """Objects as Points. https://arxiv.org/abs/1904.07850v2

    Parameters
    ----------
    base_network : torch.nn.Module
        The base feature extraction network.
    heads : OrderedDict
        OrderedDict with specifications for each head.
        For example: OrderedDict([
            ('heatmap', {'num_output': len(classes), 'bias': -2.19}),
            ('wh', {'num_output': 2}),
            ('reg', {'num_output': 2})
            ])
    classes : list of str
        Category names.
    head_conv_channel : int, default is 0
        If > 0, will use an extra conv layer before each of the real heads.
    scale : float, default is 4.0
        The downsampling ratio of the entire network.
    topk : int, default is 100
        Number of outputs .
    flip_test : bool
        Whether apply flip test in inference (training mode not affected).
    nms_thresh : float, default is 0.
        Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
        By default nms is disabled.
    nms_topk : int, default is 400
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    post_nms : int, default is 100
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.

    """
    def __init__(self, base_network, heads, classes,
                 head_conv_channel=0, scale=4.0, topk=100, flip_test=False,
                 nms_thresh=0, nms_topk=400, post_nms=100, **kwargs):
        if 'norm_layer' in kwargs:
            kwargs.pop('norm_layer')
        if 'norm_kwargs' in kwargs:
            kwargs.pop('norm_kwargs')
        super(CenterNet, self).__init__(**kwargs)
        assert isinstance(heads, OrderedDict), \
            "Expecting heads to be a OrderedDict per head, given {}" \
            .format(type(heads))
        self.classes = classes
        self.topk = topk
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        post_nms = min(post_nms, topk)
        self.post_nms = post_nms
        self.scale = scale
        self.flip_test = flip_test
        self.base_network = base_network
        self.heatmap_nms = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.decoder = CenterNetDecoder(topk=topk, scale=scale)
        self.heads = nn.Sequential()

        in_planes = self.base_network.to(torch.device('cpu'))(torch.zeros(1, 3, 256, 256)).size(1)
        for name, values in heads.items():
            num_output = values['num_output']
            bias = values.get('bias', 0.0)
            if head_conv_channel > 0:
                head = nn.Sequential(
                    nn.Conv2d(in_planes, num_output, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(num_output, num_output, kernel_size=1, stride=1, padding=0)
                )
            else:
                head = nn.Sequential(
                    nn.Conv2d(in_planes, num_output, kernel_size=1, stride=1, padding=0)
                )
            
            self._init_head_weights(head, bias)
            self.heads.add_module(name, head)
    
    def _init_head_weights(self, head, bias=0):
        for layer in head.children():
            if isinstance(layer, nn.Conv2d):
                if bias == 0:
                    nn.init.normal_(layer.weight, 0.001)
                    nn.init.zeros_(layer.bias)
                else:
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.constant_(layer.bias, bias)

    @property
    def num_classes(self):
        """Return number of foreground classes.

        Returns
        -------
        int
            Number of foreground classes

        """
        return len(self.classes)

    def set_nms(self, nms_thresh=0, nms_topk=400, post_nms=100):
        """Set non-maximum suppression parameters.

        Parameters
        ----------
        nms_thresh : float, default is 0.
            Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
            By default NMS is disabled.
        nms_topk : int, default is 400
            Apply NMS to top k detection results, use -1 to disable so that every Detection
             result is used in NMS.
        post_nms : int, default is 100
            Only return top `post_nms` detection results, the rest is discarded. The number is
            based on COCO dataset which has maximum 100 objects per image. You can adjust this
            number if expecting more objects. You can use -1 to return all detections.

        Returns
        -------
        None

        """
        self._clear_cached_op()
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        post_nms = min(post_nms, self.nms_topk)
        self.post_nms = post_nms

    # def reset_class(self, classes, reuse_weights=None):
    #     """Reset class categories and class predictors.

    #     Parameters
    #     ----------
    #     classes : iterable of str
    #         The new categories. ['apple', 'orange'] for example.
    #     reuse_weights : dict
    #         A {new_integer : old_integer} or mapping dict or {new_name : old_name} mapping dict,
    #         or a list of [name0, name1,...] if class names don't change.
    #         This allows the new predictor to reuse the
    #         previously trained weights specified.

    #     Example
    #     -------
    #     >>> net = gluoncv.model_zoo.get_model('center_net_resnet50_v1b_voc', pretrained=True)
    #     >>> # use direct name to name mapping to reuse weights
    #     >>> net.reset_class(classes=['person'], reuse_weights={'person':'person'})
    #     >>> # or use interger mapping, person is the 14th category in VOC
    #     >>> net.reset_class(classes=['person'], reuse_weights={0:14})
    #     >>> # you can even mix them
    #     >>> net.reset_class(classes=['person'], reuse_weights={'person':14})
    #     >>> # or use a list of string if class name don't change
    #     >>> net.reset_class(classes=['person'], reuse_weights=['person'])

    #     """
    #     raise NotImplementedError("Not yet implemented, please wait for future updates.")

    def forward(self, x):
        # pylint: disable=arguments-differ
        """Hybrid forward of center net"""
        y = self.base_network(x)
        out = [head(y) for head in self.heads]
        out[0] = torch.sigmoid(out[0])
        if self.training:
            out[0] = torch.clamp(out[0], 1e-4, 1 - 1e-4)
            return tuple(out)
        
        if self.flip_test:
            y_flip = self.base_network(x.flip([3]))
            out_flip = [head(y_flip) for head in self.heads]
            out_flip[0] = torch.sigmoid(out_flip[0])
            out[0] = (out[0] + out_flip[0].flip([3]))*0.5
            out[1] = (out[1] + out_flip[1].flip([3]))*0.5
        
        heatmap = out[0]
        keep = self.heatmap_nms(heatmap) == heatmap
        results = self.decoder(keep*heatmap, out[1], out[2])
        return results


def get_center_net(name, dataset, pretrained=False, device=torch.device('cpu'),
                   root=os.path.join('~', '.models'), **kwargs):
    """Get a center net instance.

    Parameters
    ----------
    name : str or None
        Model name, if `None` is used, you must specify `features` to be a `HybridBlock`.
    dataset : str
        Name of dataset. This is used to identify model name because models trained on
        different datasets are going to be very different.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    device : torch.device
        Device such as torch.device('cpu'), torch.device('gpu').
    root : str
        Model weights storing path.

    Returns
    -------
    torch.nn.Module
        A CenterNet detection network.

    """
    # pylint: disable=unused-variable
    net = CenterNet(**kwargs)
    if pretrained:
        raise NotImplementedError
        # from ..model_store import get_model_file
        # full_name = '_'.join(('center_net', name, dataset))
        # net.load_parameters(get_model_file(full_name, tag=pretrained, root=root), ctx=ctx)
    return net.to(device)


def center_net_resnet18_v1b_voc(pretrained=False, pretrained_base=True, **kwargs):
    """Center net with resnet18_v1b base network on voc dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    torch.nn.Module
        A CenterNet detection network.

    """
    from .deconv_resnet import resnet18_v1b_deconv
    from ...data import VOCDetection
    classes = VOCDetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet18_v1b_deconv(pretrained=pretrained_base, **kwargs)
    heads = OrderedDict([
        ('heatmap', {'num_output': len(classes), 'bias': -2.19}), # use bias = -log((1 - 0.1) / 0.1)
        ('wh', {'num_output': 2}),
        ('reg', {'num_output': 2})
        ])
    return get_center_net('resnet18_v1b', 'voc', base_network=base_network, heads=heads,
                          head_conv_channel=64, pretrained=pretrained, classes=classes,
                          scale=4.0, topk=40, **kwargs)