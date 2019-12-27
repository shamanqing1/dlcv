import torch
from torch import nn

class Loss(nn.Module):
    """Base class for loss.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """

    def __init__(self, weight, batch_axis, **kwargs):
        super(Loss, self).__init__(**kwargs)
        self._weight = weight
        self._batch_axis = batch_axis

    def __repr__(self):
        s = '{name}(batch_axis={_batch_axis}, w={_weight})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x, *args, **kwargs):
        """Overrides to construct symbolic graph for this `Block`.

        Parameters
        ----------
        x : Symbol or NDArray
            The first input tensor.
        *args : list of Symbol or list of NDArray
            Additional input tensors.

        """
        # pylint: disable= invalid-name
        raise NotImplementedError


class HeatmapFocalLoss(Loss):
    """Focal loss for heatmaps.

    Parameters
    ----------
    from_logits : bool
        Whether predictions are after sigmoid or softmax.
    batch_axis : int
        Batch axis.
    weight : float
        Loss weight.

    """
    def __init__(self, from_logits=False, batch_axis=0, weight=None, **kwargs):
        super(HeatmapFocalLoss, self).__init__(weight, batch_axis, **kwargs)
        self._from_logits = from_logits

    def forward(self, pred, label):
        """Loss forward"""
        if not self._from_logits:
            pred = torch.sigmoid(pred)
        pos_inds = label == 1
        neg_inds = label < 1
        neg_weights = torch.pow(1 - label, 4)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds


        # normalize
        num_pos = torch.clamp(torch.sum(pos_inds).float(), min=1, max=1e30)
        pos_loss = torch.sum(pos_loss)
        neg_loss = torch.sum(neg_loss)
        return -(pos_loss + neg_loss) / num_pos


class MaskedL1Loss(Loss):
    r"""Calculates the mean absolute error between `label` and `pred` with `mask`.

    .. math:: L = \sum_i \vert ({label}_i - {pred}_i) * {mask}_i \vert / \sum_i {mask}_i.

    `label`, `pred` and `mask` can have arbitrary shape as long as they have the same
    number of elements. The final loss is normalized by the number of non-zero elements in mask.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **label**: target tensor with the same size as pred.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(MaskedL1Loss, self).__init__(weight, batch_axis, **kwargs)

    def forward(self, pred, label, mask, sample_weight=None):
        label = label.view_as(pred)
        loss = torch.abs(label * mask - pred * mask)
        if self._weight is not None:
            loss *= self._weight
        norm = torch.sum(mask).clamp(1, 1e30)
        return torch.sum(loss) / norm
