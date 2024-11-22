"""Helper classes and typedefs for the residual frames-based attention models."""

# Standard Library
from typing import Literal, Optional

# Third-Party
from segmentation_models_pytorch.encoders import TimmUniversalEncoder, encoders

# PyTorch
from torch.utils import model_zoo

# Local folders
from .uncertainty.resnet import resnet_encoders

REDUCE_TYPES = Literal["sum", "prod", "cat", "weighted", "weighted_learnable"]


encoders.update(resnet_encoders)


def get_encoder(
    name,
    in_channels: int = 3,
    depth: int = 5,
    weights: Optional[str] = None,
    output_stride: int = 32,
    **kwargs
):
    """Get encoder by name."""
    if name.startswith("tu-"):
        name = name[3:]
        encoder = TimmUniversalEncoder(
            name=name,
            in_channels=in_channels,
            depth=depth,
            output_stride=output_stride,
            pretrained=weights is not None,
            **kwargs,
        )
        return encoder

    try:
        Encoder = encoders[name]["encoder"]
    except KeyError as e:
        raise KeyError(
            "Wrong encoder name `{}`, supported encoders: {}".format(
                name, list(encoders.keys())
            )
        ) from e

    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    if weights is not None:
        try:
            settings = encoders[name]["pretrained_settings"][weights]
        except KeyError as e:
            raise KeyError(
                "Wrong pretrained weights `{}` for encoder `{}`. Available options are: {}".format(
                    weights, name, list(encoders[name]["pretrained_settings"].keys())
                )
            ) from e
        encoder.load_state_dict(
            model_zoo.load_url(  # pyright: ignore[reportPrivateImportUsage]
                settings["url"]
            )
        )

    encoder.set_in_channels(in_channels, pretrained=weights is not None)
    if output_stride != 32:
        encoder.make_dilated(output_stride)

    return encoder
