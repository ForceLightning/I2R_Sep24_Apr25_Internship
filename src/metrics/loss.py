"""Implementation of loss functions."""

from typing import override

import torch
from thirdparty.vivim.modeling.utils import InverseTransform2D
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


class StructureLoss(_Loss):
    """Structure loss using Binary Cross-Entropy."""

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # GUARD: Check input dimensions.
        assert (
            input.ndim == 4
        ), f"input of shape {input.shape} does not have 4 dimensions!"

        if input.ndim > target.ndim:
            num_classes = input.shape[-3]
            target_one_hot = F.one_hot(target, num_classes).permute(0, -1, 1, 2)
        else:
            target_one_hot = target

        if not target.dtype.is_floating_point:
            target_float = target_one_hot.to(torch.float32, copy=True)
        else:
            target_float = target_one_hot

        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(target_float, kernel_size=31, stride=1, padding=15)
            - target_float
        )
        wce = F.binary_cross_entropy_with_logits(input, target_float, reduction="none")
        wce = (weit * wce).sum(dim=(-2, -1)) / weit.sum(dim=(-2, -1))

        pred = torch.sigmoid(input)
        inter = ((pred * target_one_hot) * weit).sum(dim=(-2, -1))
        union = ((pred + target_one_hot) * weit).sum(dim=(-2, -1))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        loss = wce + wiou

        match self.reduction:
            case "mean":
                return loss.mean()
            case "sum":
                return loss.sum()
            case _:
                return loss


class JointEdgeSegLoss(_Loss):
    """Joint Edge + Segmentation Structure Loss for Vivim."""

    def __init__(
        self,
        num_classes: int,
        edge_weight: float = 0.3,
        seg_weight: float = 1.0,
        inv_weight: float = 0.3,
        att_weight: float = 0.1,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        """Initialise the Joint Edge + Segmentation Structure Loss.

        Args:
            num_classes: Number of classification targets.
            edge_weight: Weight for edge loss
            seg_weight: Weight for segmentation structure loss.
            inv_weight: Weight for InverseForm loss.
            att_weight: Weight for edge attention loss.
            size_average: Deprecated (see :attr:`reduction`). By default,
                the losses are averaged over each loss element in the batch. Note that for
                some losses, there are multiple elements per sample. If the field :attr:`size_average`
                is set to ``False``, the losses are instead summed for each minibatch. Ignored
                when :attr:`reduce` is ``False``. Default: ``True``
            reduce: Deprecated (see :attr:`reduction`). By default, the
                losses are averaged or summed over observations for each minibatch depending
                on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
                batch element instead and ignores :attr:`size_average`. Default: ``True``
            reduction: Specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of
                elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

        """
        super().__init__(size_average, reduce, reduction)
        self.num_classes = num_classes
        self.seg_loss = StructureLoss()
        self.inverse_distance = InverseTransform2D()
        self.edge_weight = edge_weight
        self.seg_weight = seg_weight
        self.att_weight = att_weight
        self.inv_weight = inv_weight

    def bce2d(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute Binary CrossEntropy loss."""
        assert input.ndim == 4, f"input of shape {input.shape} does not have 4 dims."
        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)

        target_trans = target_t.clone()

        pos_index = target_t == 1
        neg_index = target_t == 0
        ignore_index = target_t > 1

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index = ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.zeros_like(log_p)
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num

        weight[ignore_index] = 0
        loss = F.binary_cross_entropy_with_logits(
            log_p, target_t, weight, reduction=self.reduction
        )
        return loss

    def edge_attention(self, input: Tensor, target: Tensor, edge: Tensor) -> Tensor:
        """Compute edge attention loss."""
        assert input.ndim == 4, f"input of shape {input.shape} does not have 4 dims."
        filler = torch.ones_like(target)
        return self.seg_loss(
            input, torch.where((edge.max(1)[0] > 0.8).unsqueeze(1), target, filler)
        )

    @override
    def forward(
        self, inputs: tuple[Tensor, Tensor], targets: tuple[Tensor, Tensor]
    ) -> Tensor:
        seg_in, edge_in = inputs
        seg_mask, edge_mask = targets

        total_loss = (
            self.seg_weight
            + self.seg_loss(seg_in, seg_mask)
            + self.edge_weight * self.bce2d(edge_in, edge_mask)
            + self.att_weight * self.edge_attention(seg_in, seg_mask, edge_in)
            + self.inv_weight
            + self.inverse_distance(edge_in, edge_mask)
        )
        return total_loss
