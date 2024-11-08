# Standard Library
from math import sqrt
from typing import override

# PyTorch
import torch
from torch import Tensor
from torch.nn import functional as F

# State-of-the-Art (SOTA) code
from thirdparty.AFB_URR.model.AFB_URR import AFB_URR as AfbUrrBase
from thirdparty.AFB_URR.model.FeatureBank import FeatureBank
from thirdparty.AFB_URR.myutils.data import calc_uncertainty, pad_divide_by


class AFB_URR(AfbUrrBase):
    @override
    def memorize(
        self, frame: Tensor, mask: Tensor
    ) -> tuple[list[Tensor], list[Tensor]]:
        _bs, _c, _h, _w = frame.shape
        _, k, _, _ = mask.shape
        (frame, mask), _pad = pad_divide_by(
            [frame, mask], 16, (frame.size()[-2], frame.size()[-1])
        )
        frame = frame.expand(k, -1, -1, -1)

        mask = mask[0].unsqueeze(1).float()
        mask_ones = torch.ones_like(mask)
        mask_inv = (mask_ones - mask).clamp(0, 1)

        embeddings: tuple[Tensor, Tensor] = self.encoder_m(frame, mask, mask_inv)
        r4, _r1 = embeddings

        kv_out: tuple[Tensor, Tensor] = self.keyval_r4(r4)
        k4, v4 = kv_out
        k4_list = [k4[i] for i in range(k)]
        v4_list = [v4[i] for i in range(k)]

        return k4_list, v4_list

    @override
    def segment(
        self, frame: Tensor, fb_global: FeatureBank
    ) -> tuple[Tensor, Tensor | None]:
        obj_n = fb_global.obj_n
        pad: tuple[int, int, int, int] | None = None
        if not self.training:
            [frame], pad = pad_divide_by(
                [frame], 16, (frame.size()[-2], frame.size()[-1])
            )

        bs, f, c, h, w = frame.shape
        frame = frame.reshape(bs * f, c, h, w)

        embeddings: tuple[Tensor, Tensor, Tensor, Tensor] = self.encoder_q(frame)
        r4, r3, r2, r1 = embeddings
        bs, _, global_match_h, global_match_w = r4.shape
        _, _, _local_match_h, _local_match_w = r1.shape

        k4, v4 = self.keyval_r4(r4)  # 1, dim, H/16, W/16
        res_global: Tensor = self.global_matcher(fb_global, k4, v4)
        res_global = res_global.reshape(
            bs * obj_n, v4.shape[1] * 2, global_match_h, global_match_w
        )

        r3_size = r3.shape
        r2_size = r2.shape
        r3 = (
            r3.unsqueeze(1)
            .expand(-1, obj_n, -1, -1, -1)
            .reshape(bs * obj_n, *r3_size[1:])
        )
        r2 = (
            r2.unsqueeze(1)
            .expand(-1, obj_n, -1, -1, -1)
            .reshape(bs * obj_n, *r2_size[1:])
        )

        r1_size = r1.shape
        r1 = (
            r1.unsqueeze(1)
            .expand(-1, obj_n, -1, -1, -1)
            .reshape(bs * obj_n, *r1_size[1:])
        )
        feature_size = (bs, obj_n, r1_size[2], r1_size[3])
        score = self.decoder(res_global, r3, r2, r1, feature_size)

        score = score.view(bs, obj_n, *frame.shape[-2:])

        uncertainty: Tensor
        if self.training:
            uncertainty = calc_uncertainty(F.softmax(score, dim=1))
            uncertainty = uncertainty.view(bs, -1).norm(p=2, dim=1) / sqrt(
                frame.shape[-2] * frame.shape[-1]
            )  # [B,1,H,W]
            uncertainty = (
                uncertainty.mean()  # pyright: ignore[reportOptionalMemberAccess]
            )
        else:
            uncertainty = torch.zeros(1)

        score = torch.clamp(score, 1e-7, 1 - 1e-7)
        score = torch.log((score / (1 - score)))

        if not self.training:
            assert isinstance(pad, tuple)
            if pad[2] + pad[3] > 0:
                score = score[:, :, pad[2] : -pad[3], :]
            if pad[0] + pad[1] > 0:
                score = score[:, :, :, pad[0] : -pad[1]]

        score = score[-1].unsqueeze(0)

        return score, uncertainty
