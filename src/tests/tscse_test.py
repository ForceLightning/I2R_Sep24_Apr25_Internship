# Standard Library
from typing import Any, Generator, Mapping

# Third-Party
from timm.models.resnet import Bottleneck, ResNet, resnet50

# PyTorch
from torch import nn

# First party imports
from models.tscse import TSCSENET_ENCODERS, TSCSENet, TSCSEResNetBottleneck

LUT_2D_3D: dict[type, type] = {
    nn.Conv2d: nn.Conv3d,
    nn.AvgPool2d: nn.AvgPool3d,
    nn.AdaptiveAvgPool2d: nn.AdaptiveAvgPool3d,
    nn.AdaptiveMaxPool2d: nn.AdaptiveMaxPool3d,
    nn.MaxPool2d: nn.MaxPool3d,
    nn.MaxUnpool2d: nn.MaxUnpool3d,
    nn.BatchNorm2d: nn.BatchNorm3d,
}


def flatten_model(
    modules: Generator[tuple[str, nn.Module], Any, Any], prefix=""
) -> Mapping[str, nn.Module]:
    ret: dict[str, nn.Module] = {}
    for name, mod in modules:
        if isinstance(mod, (nn.ModuleList, nn.ModuleDict, nn.Sequential)):
            print(name, dict(mod.named_modules()))
            flatten_model(
                mod.named_modules(), prefix=name if prefix == "" else f"{prefix}.{name}"
            )
        else:
            print(name, end=" ")
            ret[name if prefix == "" else f"{prefix}.{name}"] = mod

    return ret


def convert_2d_weights_to_3d_weights(module_2d: nn.Module, module_3d):
    assert isinstance(
        module_2d,
        tuple(LUT_2D_3D.keys()),
    )

    assert isinstance(module_3d, LUT_2D_3D[type(module_2d)])


class TestLoadWeights:
    def test_loading(self):
        resnet: ResNet = resnet50(pretrained=True)
        tscse_params = TSCSENET_ENCODERS["tscse_resnet50"]["params"]
        tscse_params |= {"num_frames": 30, "depth": 5, "name": "tscse_resnet50"}
        del tscse_params["out_channels"]
        tscse_resnet50 = TSCSENet(**tscse_params)

        resnet_mods = dict(resnet.named_modules())
        tscse_mods = dict(tscse_resnet50.named_modules())

        intersection = set(resnet_mods.keys()).intersection(set(tscse_mods.keys()))
        # difference = set(tscse_mods.keys()).difference(set(resnet_mods.keys()))

        # intersection = {k: type(tscse_mods[k]) for k in intersection}
        # difference = {k: type(tscse_mods[k]) for k in difference}

        for r_mod, t_mod, k in zip(
            [resnet_mods[k] for k in intersection],
            [tscse_mods[k] for k in intersection],
            intersection,
            strict=True,
        ):
            if isinstance(r_mod, nn.Sequential) and isinstance(t_mod, nn.Sequential):
                pass
                # print(dict(r_mod.named_modules()), dict(t_mod.named_modules()))
                # assert 1 == 0
            elif isinstance(r_mod, Bottleneck) and isinstance(
                t_mod, TSCSEResNetBottleneck
            ):
                print(k)
                print("conv1", r_mod.conv1.weight.shape, t_mod.conv1.weight.shape)
                print("bn1", r_mod.bn1.weight, t_mod.bn1.weight, end="\n\n")

        assert 1 == 0
