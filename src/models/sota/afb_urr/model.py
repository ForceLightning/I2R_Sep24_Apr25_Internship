"""AFB-URR Feature bank and modules."""

# Standard Library
from math import sqrt
from typing import Union

# Third-Party
import numpy as np
from numpy import typing as npt

# PyTorch
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_scatter import scatter_mean
from torchvision.models import resnet50

# State-of-the-Art (SOTA) code
from thirdparty.AFB_URR import myutils
from thirdparty.AFB_URR.model.AFB_URR import Refine, ResBlock
from thirdparty.AFB_URR.myutils.data import calc_uncertainty, pad_divide_by


class FeatureBank(nn.Module):
    """Feature Bank module for AFB-URR."""

    def __init__(
        self,
        obj_n: int,
        memory_budget: int,
        device: Union[torch.device, str],
        update_rate: float = 0.1,
        thresh_close: float = 0.95,
    ):
        """Initialise the Feature Bank.

        Args:
            obj_n: Number of objects to segment.
            memory_budget: Total number of features to hold.
            device: Device to perform computations on.
            update_rate: Rate at which to update the bank.
            thresh_close: Threshold for merging features.

        """
        super().__init__()

        self.obj_n = obj_n
        self.update_rate = update_rate
        self.thresh_close = thresh_close
        self.device = device

        #: List of frame indices for each object.
        self.info: list[Tensor] = []
        #: Peak memory usage for each object.
        self.peak_n: npt.NDArray[np.int32] = np.zeros(obj_n, dtype=np.int32)
        #: How many features were replaced.
        self.replace_n: npt.NDArray[np.int32] = np.zeros(obj_n, dtype=np.int32)

        #: Memory budget for each class.
        self.class_budget = memory_budget // obj_n
        if obj_n == 2:
            self.class_budget = 0.8 * self.class_budget

    def init_bank(self, keys: list[Tensor], values: list[Tensor], frame_idx: int = 0):
        """Initialise the Feature Bank with keys and values.

        Args:
            keys: List of keys for each object.
            values: List of values for each object.
            frame_idx: Frame index.

        """
        self.keys = keys
        self.values = values

        for class_idx in range(self.obj_n):
            _, _, bank_n = keys[class_idx].shape
            self.info.append(torch.zeros((bank_n, 2), device=self.device))
            self.info[class_idx][:, 0] = frame_idx
            self.peak_n[class_idx] = max(
                self.peak_n[class_idx], self.info[class_idx].shape[0]
            )

    def update(self, prev_key: Tensor, prev_value: Tensor, frame_idx: int):
        """Update the Feature Bank with new keys and values.

        Note: This method is untested for batch sizes > 1.

        Args:
            prev_key: Previous keys.
            prev_value: Previous values.
            frame_idx: Frame index.

        """
        for class_idx in range(self.obj_n):
            _, d_key, bank_n = self.keys[class_idx].shape
            _, d_val, _ = self.values[class_idx].shape

            normed_keys = F.normalize(self.keys[class_idx], dim=0)
            normed_prev_key = F.normalize(prev_key[class_idx], dim=0)
            mag_keys = self.keys[class_idx].norm(p=2, dim=0)
            corr = torch.mm(
                normed_keys.transpose(0, 1), normed_prev_key
            )  # bank_n, prev_n
            related_bank_idx = corr.argmax(dim=0, keepdim=True)  # 1, HW
            related_bank_corr = torch.gather(corr, 0, related_bank_idx)  # 1, HW

            # greater than threshold, merge them
            selected_idx = (related_bank_corr[0] > self.thres_close).nonzero()
            class_related_bank_idx = related_bank_idx[
                0, selected_idx[:, 0]
            ]  # selected_HW
            unique_related_bank_idx, _cnt = class_related_bank_idx.unique(
                dim=0, return_counts=True
            )  # selected_HW

            # Update key
            key_bank_update = torch.zeros(
                (d_key, bank_n), dtype=torch.float, device=self.device
            )  # d_key, THW
            key_bank_idx = class_related_bank_idx.unsqueeze(0).expand(
                d_key, -1
            )  # d_key, HW
            scatter_mean(
                normed_prev_key[:, selected_idx[:, 0]],
                key_bank_idx,
                dim=1,
                out=key_bank_update,
            )
            # d_key, selected_HW

            self.keys[class_idx][:, unique_related_bank_idx] = mag_keys[
                unique_related_bank_idx
            ] * (
                (1 - self.update_rate) * normed_keys[:, unique_related_bank_idx]
                + self.update_rate * key_bank_update[:, unique_related_bank_idx]
            )

            # Update value
            normed_values = F.normalize(self.values[class_idx], dim=0)
            normed_prev_value = F.normalize(prev_value[class_idx], dim=0)
            mag_values = self.values[class_idx].norm(p=2, dim=0)
            val_bank_update = torch.zeros(
                (d_val, bank_n), dtype=torch.float, device=self.device
            )
            val_bank_idx = class_related_bank_idx.unsqueeze(0).expand(d_val, -1)
            scatter_mean(
                normed_prev_value[:, selected_idx[:, 0]],
                val_bank_idx,
                dim=1,
                out=val_bank_update,
            )

            self.values[class_idx][:, unique_related_bank_idx] = mag_values[
                unique_related_bank_idx
            ] * (
                (1 - self.update_rate) * normed_values[:, unique_related_bank_idx]
                + self.update_rate * val_bank_update[:, unique_related_bank_idx]
            )

            # less than the threshold, concat them
            selected_idx = (related_bank_corr[0] <= self.thres_close).nonzero()

            if self.class_budget < bank_n + selected_idx.shape[0]:
                self.remove(class_idx, selected_idx.shape[0], frame_idx)

            self.keys[class_idx] = torch.cat(
                [self.keys[class_idx], prev_key[class_idx][:, selected_idx[:, 0]]],
                dim=1,
            )
            self.values[class_idx] = torch.cat(
                [self.values[class_idx], prev_value[class_idx][:, selected_idx[:, 0]]],
                dim=1,
            )

            new_info = torch.zeros((selected_idx.shape[0], 2), device=self.device)
            new_info[:, 0] = frame_idx
            self.info[class_idx] = torch.cat([self.info[class_idx], new_info], dim=0)

            self.peak_n[class_idx] = max(
                self.peak_n[class_idx], self.info[class_idx].shape[0]
            )

            self.info[class_idx][:, 1] = torch.clamp(
                self.info[class_idx][:, 1], 0, 1e5
            )  # Prevent inf

    def remove(self, class_idx: int, request_n: int, frame_idx: int) -> int | float:
        """Remove features from the Feature Bank.

        Note: This method is untested for batch sizes > 1.

        Args:
            class_idx: Index of the class.
            request_n: Number of features to remove.
            frame_idx: Frame index.

        Returns:
            int | float: Balance of features.

        """
        old_size = self.keys[class_idx].shape[1]

        LFU = frame_idx - self.info[class_idx][:, 0]  # time length
        LFU = self.info[class_idx][:, 1] / LFU
        thres_dynamic = int(LFU.min()) + 1
        iter_cnt = 0

        while True:
            selected_idx = LFU > thres_dynamic
            self.keys[class_idx] = self.keys[class_idx][:, selected_idx]
            self.values[class_idx] = self.values[class_idx][:, selected_idx]
            self.info[class_idx] = self.info[class_idx][selected_idx]
            LFU = LFU[selected_idx]
            iter_cnt += 1

            balance = (self.class_budget - self.keys[class_idx].shape[1]) - request_n
            if balance < 0:
                thres_dynamic = int(LFU.min()) + 1
            else:
                break

        new_size = self.keys[class_idx].shape[1]
        self.replace_n[class_idx] += old_size - new_size

        return balance

    def print_peak_mem(self) -> None:
        """Print the peak memory usage of the Feature Bank."""
        ur = self.peak_n / self.class_budget
        rr = self.replace_n / self.class_budget
        print(
            f"Obj num: {self.obj_n}.",
            f"Budget / obj: {self.class_budget}.",
            f"UR: {ur}.",
            f"Replace: {rr}.",
        )


class EncoderM(nn.Module):
    """Mask Encoder for AFB-URR."""

    def __init__(self, load_imagenet_params: bool):
        """Initialise the EncoderM module.

        Args:
            load_imagenet_params: Load ImageNet parameters.

        """
        super(EncoderM, self).__init__()
        #: Convolutional layer for the mask.
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #: Convolutional layer for the inverted mask.
        self.conv1_o = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet = resnet50(pretrained=load_imagenet_params)
        #: ResNet-50 conv1
        self.conv1 = resnet.conv1
        #: ResNet-50 bn1
        self.bn1 = resnet.bn1
        #: ResNet-50 relu
        self.relu = resnet.relu  # 1/2, 64
        #: ResNet-50 maxpool
        self.maxpool = resnet.maxpool

        #: ResNet-50 layer1
        self.res2 = resnet.layer1  # 1/4, 256
        #: ResNet-50 layer2
        self.res3 = resnet.layer2  # 1/8, 512
        #: ResNet-50 layer3
        self.res4 = resnet.layer3  # 1/16, 1024

        self.register_buffer(
            "mean", torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, in_f: Tensor, in_m: Tensor, in_o: Tensor):
        """Forward pass for the encoder.

        Args:
            in_f: Input features (B, K, C, H, W)
            in_m: Mask (B, 1, H, W)
            in_o: Inverted mask (B, 1, H, W)

        Returns:
            Tensor: Tensor of shape (B, K, C, H, W)

        """
        b, k, _c, _h, _w = in_f.shape
        in_f = in_f.flatten(0, 1)

        f = (in_f - self.mean) / self.std

        x = self.conv1(f)
        c_m = self.conv1_m(in_m)
        c_o = self.conv1_o(in_o)

        x = x.view(b, k, *x.shape[1:])
        c_m = c_m.view(b, 1, *c_m.shape[1:])
        c_o = c_o.view(b, 1, *c_o.shape[1:])

        x = x + c_m + c_o

        x = x.view(b * k, *x.shape[2:])

        x = self.bn1(x)
        r1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(r1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/16, 1024

        r4 = r4.view(b, k, *r4.shape[1:])
        r1 = r1.view(b, k, *r1.shape[1:])

        return r4, r1


class EncoderQ(nn.Module):
    """Encoder (Query) module for AFB-URR."""

    def __init__(self, load_imagenet_params: bool):
        """Initialise the EncoderQ module.

        Args:
            load_imagenet_params: Load ImageNet parameters.

        """
        super(EncoderQ, self).__init__()
        resnet = resnet50(pretrained=load_imagenet_params)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/8, 1024

        self.register_buffer(
            "mean", torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, in_f: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass for the encoder.

        Args:
            in_f: Input features (B, F, C, H, W)

        """
        b, f, _c, _h, _w = in_f.shape
        in_f = in_f.flatten(0, 1)

        feat = (in_f - self.mean) / self.std

        x = self.conv1(feat)
        x = self.bn1(x)
        r1: Tensor = self.relu(x)  # 1/2, 64
        x = self.maxpool(r1)  # 1/4, 64
        r2: Tensor = self.res2(x)  # 1/4, 256
        r3: Tensor = self.res3(r2)  # 1/8, 512
        r4: Tensor = self.res4(r3)  # 1/8, 1024

        ret = [r4, r3, r2, r1]
        for i, r in enumerate(ret):
            ret[i] = r.reshape(b, f, *r.shape[1:])

        r4, r3, r2, r1 = ret

        return r4, r3, r2, r1


class KeyValue(nn.Module):
    """Key-Value module for AFB-URR."""

    def __init__(self, indim: int, keydim: int, valdim: int):
        """Initialise the Key-Value module.

        Args:
            indim: Input dimension.
            keydim: Key dimension.
            valdim: Value dimension.

        """
        super(KeyValue, self).__init__()
        self.keydim = keydim
        self.valdim = valdim
        self.Key = nn.Conv2d(
            indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1
        )
        self.Value = nn.Conv2d(
            indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass for the Key-Value module.

        Args:
            x: Input tensor (B, K, C, H, W)

        """
        b, k, _c, _h, _w = x.shape
        x = x.flatten(0, 1)
        key = self.Key(x)
        key = key.view(b, k, key.shape[1], -1)  # bs, obj_n, key_dim, pixel_n

        val = self.Value(x)
        val = val.view(b, k, val.shape[1], -1)  # bs, obj_n, key_dim, pixel_n

        return key, val


class Matcher(nn.Module):
    """Matcher module for AFB-URR.

    Matches the keys and queries to the Feature Bank.
    """

    def __init__(self, thres_valid: float = 1e-3, update_bank: bool = False):
        """Initialise the Matcher module.

        Args:
            thres_valid: Threshold for valid matches.
            update_bank: Whether to update the bank.

        """
        super(Matcher, self).__init__()
        self.thres_valid = thres_valid
        self.update_bank = update_bank

    def forward(self, feature_bank: FeatureBank, q_in: Tensor, q_out: Tensor) -> Tensor:
        """Forward pass for the Matcher module.

        Args:
            feature_bank: Feature Bank.
            q_in: Input query tensor (B, F, K * H * W)
            q_out: Output query tensor (B, F, K * H * W)

        Returns:
            Tensor: Output tensor (B, F, K * H * W)

        """
        # q_in: (B, F, K * H * W)
        # q_out: (B, F, K * H * W)
        bs = q_in.shape[0]
        out = []
        for j in range(bs):
            mem_out_list = []
            for i in range(0, feature_bank.obj_n):
                _bs, d_key, _bank_n = feature_bank.keys[i].size()
                try:
                    p = torch.matmul(
                        feature_bank.keys[i][j].transpose(0, 1), q_in[j]
                    ) / sqrt(
                        d_key
                    )  # THW, HW
                    p = F.softmax(p, dim=1)  # bs, bank_n, HW
                    mem = torch.matmul(feature_bank.values[i][j], p)  # bs, D_o, HW
                except RuntimeError:
                    device = feature_bank.keys[i].device
                    key_cpu = feature_bank.keys[i][j].cpu()
                    value_cpu = feature_bank.values[i][j].cpu()
                    q_in_cpu = q_in[j].cpu()

                    p = torch.matmul(key_cpu.transpose(0, 1), q_in_cpu) / sqrt(
                        d_key
                    )  # THW, HW
                    p = F.softmax(p, dim=1)  # bs, bank_n, HW
                    mem = torch.matmul(value_cpu, p).to(device)  # bs, D_o, HW
                    p = p.to(device)
                    print(
                        "\tLine 158. GPU out of memory, use CPU", f"p size: {p.shape}"
                    )

                mem_out_list.append(torch.cat([mem, q_out[j]], dim=1))

                if self.update_bank:
                    try:
                        ones = torch.ones_like(p)
                        zeros = torch.zeros_like(p)
                        bank_cnt = torch.where(p > self.thres_valid, ones, zeros).sum(
                            dim=2
                        )[0]
                    except RuntimeError:
                        device = p.device
                        p = p.cpu()
                        ones = torch.ones_like(p)
                        zeros = torch.zeros_like(p)
                        bank_cnt = (
                            torch.where(p > self.thres_valid, ones, zeros)
                            .sum(dim=2)[0]
                            .to(device)
                        )
                        print(
                            "\tLine 170. GPU out of memory, use CPU",
                            f"p size: {p.shape}",
                        )

                    feature_bank.info[i][:, 1] += torch.log(bank_cnt + 1)

            mem_out_tensor = torch.stack(mem_out_list, dim=0)
            mem_out_tensor = mem_out_tensor.transpose(0, 1)  # f, obj_n, dim, pixel_n
            out.append(mem_out_tensor)

        return torch.stack(out, dim=0)


class Decoder(nn.Module):
    """Decoder module for AFB-URR."""

    def __init__(self, device: torch.device | str):  # mdim_global = 256
        """Initialise the Decoder module.

        Args:
            device: Device to perform computations on.

        """
        super(Decoder, self).__init__()

        self.device = device
        mdim_global = 256
        mdim_local = 32
        local_size = 7

        # Patch-wise
        self.convFM = nn.Conv2d(1024, mdim_global, kernel_size=3, padding=1, stride=1)
        self.ResMM = ResBlock(mdim_global, mdim_global)
        self.RF3 = Refine(512, mdim_global)  # 1/8 -> 1/8
        self.RF2 = Refine(256, mdim_global)  # 1/8 -> 1/4
        self.pred2 = nn.Conv2d(mdim_global, 2, kernel_size=3, padding=1, stride=1)

        # Local
        self.local_avg = nn.AvgPool2d(local_size, stride=1, padding=local_size // 2)
        self.local_max = nn.MaxPool2d(local_size, stride=1, padding=local_size // 2)
        self.local_convFM = nn.Conv2d(
            128, mdim_local, kernel_size=3, padding=1, stride=1
        )
        self.local_ResMM = ResBlock(mdim_local, mdim_local)
        self.local_pred2 = nn.Conv2d(mdim_local, 2, kernel_size=3, padding=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(
        self,
        patch_match: Tensor,
        r3: Tensor,
        r2: Tensor,
        r1: Tensor,
        feature_shape: tuple[int, int, int, int, int],
    ) -> Tensor:
        """Forward pass for the decoder.

        Args:
            patch_match: Global match tensor from the Matcher.
            r3: Residual tensor from the ResNet-50 layer 3.
            r2: Residual tensor from the ResNet-50 layer 2.
            r1: Residual tensor from the ResNet-50 layer 1.
            feature_shape: Shape of the features.

        Returns:
            Tensor: Segmentation mask. Shape: (B, F, K, H, W)

        """
        b, f, k, *_ = patch_match.shape
        p = self.ResMM(self.convFM(patch_match.flatten(0, 2)))
        p = self.RF3(r3.flatten(0, 2), p)  # out: 1/8, 256
        p = self.RF2(r2.flatten(0, 2), p)  # out: 1/4, 256
        p = self.pred2(F.relu(p))

        p = F.interpolate(p, scale_factor=2, mode="bilinear", align_corners=False)
        #: (B, F, K, 2, H, W)
        p = p.reshape(b, f, k, *p.shape[1:])

        bs, f, obj_n, h, w = feature_shape
        rough_seg = F.softmax(p, dim=3)
        rough_seg = rough_seg[:, :, :, 1]
        rough_seg = rough_seg.view(bs * f, obj_n, h, w)
        rough_seg = F.softmax(rough_seg, dim=1)  # object-level normalization

        # Local refinement
        uncertainty = myutils.calc_uncertainty(rough_seg)
        uncertainty = uncertainty.expand(-1, obj_n, -1, -1).reshape(
            bs * f * obj_n, 1, h, w
        )

        rough_seg = rough_seg.view(bs * f * obj_n, 1, h, w)  # bs*f*obj_n, 1, h, w
        r1_weighted = r1.flatten(0, 2) * rough_seg
        r1_local = self.local_avg(r1_weighted)  # bs*f*obj_n, 64, h, w
        r1_local = r1_local / (
            self.local_avg(rough_seg) + 1e-8
        )  # neighborhood reference
        r1_conf = self.local_max(rough_seg)  # bs*f*obj_n, 1, h, w

        local_match = torch.cat([r1.flatten(0, 2), r1_local], dim=1)
        q = self.local_ResMM(self.local_convFM(local_match))
        q = r1_conf * self.local_pred2(F.relu(q))

        p = p.flatten(0, 2) + uncertainty * q
        p = F.interpolate(p, scale_factor=2, mode="bilinear", align_corners=False)
        p = F.softmax(p, dim=1)[:, 1]  # no, h, w

        p = p.reshape(bs, f, obj_n, *p.shape[-2:])

        return p


class AFB_URR(nn.Module):
    """AFB-URR model."""

    def __init__(
        self,
        device: torch.device | str,
        update_bank: bool,
        load_imagenet_params: bool = False,
    ):
        """Initialise the AFB-URR model.

        Args:
            device: Device to perform computations on.
            update_bank: Whether to update the bank.
            load_imagenet_params: Load ImageNet parameters.

        """
        super(AFB_URR, self).__init__()

        self.device = device
        self.encoder_m = EncoderM(load_imagenet_params)
        self.encoder_q = EncoderQ(load_imagenet_params)

        self.keyval_r4 = KeyValue(1024, keydim=128, valdim=512)

        self.global_matcher = Matcher(update_bank=update_bank)
        self.decoder = Decoder(device)

    def memorize(
        self, frame: Tensor, mask: Tensor
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Memorise the first frame and mask.

        Args:
            frame: Frame to memorise. (B, 1, C, H, W)
            mask: Mask to memorise. (B, K, H, W)

        Returns:
            tuple[list[Tensor], list[Tensor]]: List of keys and values.

        """
        _, k, _, _ = mask.shape
        (frame, mask), _pad = pad_divide_by(
            [frame, mask], 16, (frame.size()[-2], frame.size()[-1])
        )
        frame = frame.expand(-1, k, -1, -1, -1)

        mask = mask[:, 0:1].float()
        mask_ones = torch.ones_like(mask)
        mask_inv = (mask_ones - mask).clamp(0, 1)

        embeddings: tuple[Tensor, Tensor] = self.encoder_m(frame, mask, mask_inv)
        r4, _r1 = embeddings

        kv_out: tuple[Tensor, Tensor] = self.keyval_r4(r4)
        k4, v4 = kv_out

        k4_list = [k4[:, i] for i in range(k)]
        v4_list = [v4[:, i] for i in range(k)]

        return k4_list, v4_list

    def segment(
        self, frame: Tensor, fb_global: FeatureBank
    ) -> tuple[Tensor, Tensor | None]:
        """Segment the remaining frames.

        Args:
            frame: Remaining frames.
            fb_global: Global Feature Bank.

        Returns:
            tuple[Tensor, Tensor | None]: Segmentation mask and uncertainty.

        """
        obj_n = fb_global.obj_n
        pad: tuple[int, int, int, int] | None = None
        if not self.training:
            [frame], pad = pad_divide_by(
                [frame], 16, (frame.size()[-2], frame.size()[-1])
            )

        bs, f, _c, _h, _w = frame.shape

        embeddings: tuple[Tensor, Tensor, Tensor, Tensor] = self.encoder_q(frame)
        r4, r3, r2, r1 = embeddings
        _bs, _f, _c1, global_match_h, global_match_w = r4.shape
        _bs, _f, _c2, _local_match_h, _local_match_w = r1.shape

        k4, v4 = self.keyval_r4(r4)  # B, dim, H/16, W/16
        res_global: Tensor = self.global_matcher(fb_global, k4, v4)
        res_global = res_global.reshape(
            bs, f, obj_n, v4.shape[2] * 2, global_match_h, global_match_w
        )

        r3_size = r3.shape
        r2_size = r2.shape
        r3 = (
            r3.unsqueeze(2)
            .expand(-1, -1, obj_n, -1, -1, -1)
            .reshape(bs, f, obj_n, *r3_size[2:])
        )
        r2 = (
            r2.unsqueeze(2)
            .expand(-1, -1, obj_n, -1, -1, -1)
            .reshape(bs, f, obj_n, *r2_size[2:])
        )

        r1_size = r1.shape
        r1 = (
            r1.unsqueeze(2)
            .expand(-1, -1, obj_n, -1, -1, -1)
            .reshape(bs, f, obj_n, *r1_size[2:])
        )
        feature_size = (bs, f, obj_n, r1_size[3], r1_size[4])
        score = self.decoder(res_global, r3, r2, r1, feature_size)

        uncertainty: Tensor
        if self.training:
            uncertainty = calc_uncertainty(F.softmax(score, dim=2))
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

        score = score[:, -1]

        return score, uncertainty


# NOTE: This is for debugging purposes only.
if __name__ == "__main__":
    BATCH_SIZE = 2
    input_shape = (BATCH_SIZE, 10, 3, 224, 224)
    mask_shape = (BATCH_SIZE, 4, 224, 224)
    input = torch.randn(input_shape).cuda()
    masks = torch.randint(0, 2, mask_shape, device="cuda:0")

    model = AFB_URR("cuda:0", update_bank=False, load_imagenet_params=True).cuda()
    fb_global = FeatureBank(4, 300000, "cuda:0").cuda()
    k4_list, v4_list = model.memorize(input[:, 0:1], masks)
    fb_global.init_bank(k4_list, v4_list)

    masks_proba, uncertainty = model.segment(input[:, 1:], fb_global)
