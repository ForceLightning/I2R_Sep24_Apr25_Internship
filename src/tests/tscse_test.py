# Third-Party

# Standard Library
import ssl

# Third-Party
from pretrainedmodels.models.senet import SENet, se_resnet50

# PyTorch
import torch
from torch import nn

# First party imports
from models.tscse.tscse import TSCSENET_ENCODERS, TSCSENet
from models.tscse.utils import LUT_2D_3D

ssl._create_default_https_context = ssl._create_unverified_context


class TestLoadWeights:
    def test_loading(self):
        resnet: SENet = se_resnet50()
        tscse_params = TSCSENET_ENCODERS["tscse_resnet50"]["params"]
        tscse_params |= {"num_frames": 10, "depth": 5, "name": "tscse_resnet50"}
        del tscse_params["out_channels"]
        tscse_resnet50 = TSCSENet(**tscse_params)

        se_resnet_mods = dict(resnet.named_modules())
        tscse_mods = dict(tscse_resnet50.named_modules())

        intersection = set(se_resnet_mods.keys()).intersection(set(tscse_mods.keys()))

        print(se_resnet_mods.keys(), tscse_mods.keys(), intersection, sep="\n")

        for r_mod, t_mod, k in zip(
            [se_resnet_mods[k] for k in intersection],
            [tscse_mods[k] for k in intersection],
            intersection,
            strict=True,
        ):
            if isinstance(r_mod, nn.Sequential) and isinstance(t_mod, nn.Sequential):
                pass
            elif isinstance(r_mod, (tuple(LUT_2D_3D.keys()))):
                func = LUT_2D_3D[type(r_mod)]["func"]
                new = func(r_mod)

                tscse_resnet50.set_submodule(k, new)

                if isinstance(new, nn.Conv3d):
                    assert (
                        dict(tscse_resnet50.named_modules())[k].weight == new.weight
                    ).all()

        tscse_resnet50 = tscse_resnet50.cuda()

        input_img = torch.randn((2, 3, 10, 224, 224), dtype=torch.float32).cuda()

        _ = tscse_resnet50(input_img)
