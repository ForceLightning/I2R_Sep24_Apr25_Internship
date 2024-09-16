# I2R Sep24 - Apr25 Internship
An in-development repository for Christopher Kok's internship at A*STAR from Sep 2024 to Apr 2025.

# Introduction
This codebase is heavily modified from the [work](https://github.com/ryanmliu0/I2R_Summer_Internship_Code) that Ryan Liu did during his internship. Note that the dataset files are not included in this repository.

See the [changelog](./CHANGELOG.md) for the differences in our approaches.

# Requirements
- A CUDA-supported GPU.
- [CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive)
- Python >= 3.12
- Pipenv

# Installation
Install the dependencies from `Pipfile.lock`.
```sh
pipenv sync
```

# Usage
Some default configurations are included in the `./configs/` directory, which will be used as modular pieces to construct the complete training/validation configuration.
## Description of config modules
- `cine.yaml`, `lge.yaml`, `two_plus_one.yaml`: Incomplete defaults for initialisation.
- `training.yaml`, `testing.yaml`: Overrides for training or validation/testing modes. This must be the last config file loaded in.
- `*_greyscale.yaml`, `*_rgb.yaml`: Defaults for handling either RGB images or greyscale images as inputs.
- `cine_tpo_resnet50.yaml`, `cine_tpo_senet154.yaml`: Defaults for the ResNet50 and SENet154 backbones for the CINE and TwoPlusOne tasks.

## Useful CLI arguments
- `--version`: Sets the name of the experiment for logging and model checkpointing use cases.
- `--data.num_workers`: Set this to a value above 0 for dataloader workers to spawn. Defaults to 8.
- `--data.batch_size`: Set this to a value that will allow the model + data to fit in GPU memory.
- `--model.num_frames`: If the computational or memory complexity of the process overwhelms available compute, set this to a multiple of 5 for the [TwoPlusOne](#TwoPlusOne) task. Defaults to 5.
- `--model.weights_from_ckpt_path`: Set this to the path of a prior checkpoint to load the model's weights. Note that the config must be the same.
- `--model.encoder_name` and `--model.encoder_weights`: This sets the backbone architecture of the UNet model and the desired weights for those available in the [SMP module](https://github.com/qubvel-org/segmentation_models.pytorch).

See the help descriptions for more info.
```sh
python <script_name>.py --help
```

## LGE
The example below runs a UNet with a ResNet50 backbone and RGB image loading.
```sh
python lge.py fit --config ./configs/lge.yaml -- config ./configs/lge_rgb.yaml --configs ./configs/training.yaml --version default
```

## CINE
The example below runs a UNet with a ResNet50 backbone and RGB image loading.
```sh
python cine.py fit --config ./configs/cine.yaml -- config ./configs/cine_rgb.yaml --configs ./configs/cine_tpo_resnet50.yaml --configs ./configs/training.yaml --version default
```

## TwoPlusOne
The example below runs a UNet with TwoPlusOne residual connections and a ResNet50 backbone and RGB image loading.
```sh
python two_plus_one.py fit --config ./configs/two_plus_one.yaml --config ./configs/two_plus_one_rgb.yaml --config ./configs/cine_tpo_resnet50.yaml --config ./configs/training.yaml --version default --model.num_frames 5
```
