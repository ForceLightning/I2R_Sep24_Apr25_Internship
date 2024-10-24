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

## Directory Structure
```sh
.
├── CHANGELOG.md
├── Pipfile
├── Pipfile.lock
├── README.md
├── __init__.py
├── attention_unet.py
├── checkpoints
├── cine.py
├── configs
│   └── *.yaml
├── data
│   ├── Indices         # Used for splitting the train/val dataset.
│   ├── test            # Test dataset.
│   │   ├── Cine        #   CINE image data
│   │   ├── LGE         #   LGE image data
│   │   └── masks       #   Annotated masks
│   └── train_val       # Train/Val dataset.
│   │   ├── Cine        #   CINE image data
│   │   ├── LGE         #   LGE image data
│   │   └── masks       #   Annotated masks
├── dataset
│   ├── __init__.py
│   └── dataset.py
├── docs
├── lge.py
├── metrics
│   ├── __init__.py
│   ├── dice.py
│   └── logging.py
├── models
│   ├── __init__.py
│   ├── attention.py
│   ├── two_plus_one.py
│   └── two_stream.py
├── tests
│   ├── __init__.py
│   ├── conv1d_test.py
│   └── quick_test.py
├── two_plus_one.py
├── two_stream.py
└── utils
```

# Installation
Install the dependencies from `Pipfile.lock`.
```sh
pipenv sync
```

## CUDA-accelerated Optical Flow support
For CUDA-accelerated optical flow calculations, OpenCV must be built from source with CUDA support. See [OpenCV-Python](https://github.com/opencv/opencv-python?tab=readme-ov-file#manual-builds) manual build docs for detailed instructions.

For reference, the build command used in this project was:
```sh
LD_LIBRARY_PATH=/usr/lib/wsl/lib ENABLE_HEADLESS=1 ENABLE_CONTRIB=1 CMAKE_ARGS="-DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_CUBLAS=ON -DWITH_MKL=ON -DMKL_USE_MULTITHREAD=ON -DPYTHON3_NUMPY_INCLUDE_DIRS=<PATH TO VIRTUAL ENV>/lib/python3.12/site-packages/numpy/_core/include" MAKEFLAGS="-j <DISCRETE CPU CORES>" pip wheel . --verbose
```
Set the path to the virutal env to find numpy header files and the number of discrete cpu cores for faster build times.

> [!NOTE]
> On Windows Subsystem for Linux (WSL) environments, ensure that the path to WSL libraries (after installing CUDA toolkit drivers and CUDNN libraries are in the `$PATH` and `$LD_LIBRARY_PATH` environment variables. This may be set in `~/.bashrc` or `~/.zshrc` configurations.
> ```sh
> export PATH="/usr/lib/wsl/lib:$PATH"
> export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
> ```

# Usage
Some default configurations are included in the `./configs/` directory, which will be used as modular pieces to construct the complete training/validation configuration.

> [!NOTE]
> Ensure that the environment variable `PYTHONPATH` is set to `./src/`.
> This can be done with:
> ```sh
> export PYTHONPATH=./src/
> ```
> Alternatively, set it in a `.env` file if using pipenv.

## Description of config modules
- `cine.yaml`, `lge.yaml`, `two_stream.yaml`, `two_plus_one.yaml`, `residual_attention.yaml`: Incomplete defaults for initialisation.
- `training.yaml`, `testing.yaml`, `no_checkpointing.yaml`: Overrides for training or validation/testing/quick run modes. This must be the last config file loaded in.
- `*_greyscale.yaml`, `*_rgb.yaml`: Defaults for handling either RGB images or greyscale images as inputs.
- `cine_tpo_resnet50.yaml`, `cine_tpo_senet154.yaml`: Defaults for the ResNet50 and SENet154 backbones for the CINE, TwoStream, TwoPlusOne, and Attention tasks.

## Useful CLI arguments
- `--version`: Sets the name of the experiment for logging and model checkpointing use cases.
- `--data.num_workers`: Set this to a value above 0 for dataloader workers to spawn. Defaults to 8.
- `--data.batch_size`: Set this to a value that will allow the model + data to fit in GPU memory.
- `--model.num_frames`: If the computational or memory complexity of the process overwhelms available compute, set this to a multiple of 5 for the [TwoPlusOne](#TwoPlusOne) task. Defaults to 5.
- `--model.weights_from_ckpt_path`: Set this to the path of a prior checkpoint to load the model's weights. Note that the config must be the same.
- `--model.encoder_name` and `--model.encoder_weights`: This sets the backbone architecture of the U-Net model and the desired weights for those available in the [SMP module](https://github.com/qubvel-org/segmentation_models.pytorch).
- `--data.augment`: This sets the dataset to augment images when loading them to the model during training only.

See the help descriptions for more info.
```sh
python -m <script_name> --help
```

## LGE
The example below runs a U-Net with a ResNet50 backbone and RGB image loading.
```sh
python -m lge fit --config ./configs/lge.yaml --config ./configs/lge_rgb.yaml --configs ./configs/training.yaml --version default
```

## CINE
The example below runs a U-Net with a ResNet50 backbone and RGB image loading.
```sh
python -m cine --config ./configs/cine.yaml --config ./configs/cine_rgb.yaml --configs ./configs/cine_tpo_resnet50.yaml --configs ./configs/training.yaml --version default
```

## Two Stream
The example below runs a U-Net with a Resnet50 backbone and RGB image loading.
```sh
python -m two_stream fit --config ./configs/two_stream.yaml --config ./configs/two_stream_rgb.yaml --config ./configs/cine_tpo_resnet50.yaml --configs ./configs/training.yaml --version default
```

## 2 + 1
The example below runs a U-Net with 2 + 1 temporal convolution residual connections and a ResNet50 backbone and RGB image loading.
```sh
python -m two_plus_one fit --config ./configs/two_plus_one.yaml --config ./configs/two_plus_one_rgb.yaml --config ./configs/cine_tpo_resnet50.yaml --config ./configs/training.yaml --model.num_frames 5 --version default
```
## Attention U-Net
The example below runs a U-Net with 2 + 1 temporal convolution residual connections and attention mechanism on residual frames using a ResNet50 backbone and Greyscale image loading.
```sh
python -m attention_unet fit --config ./configs/residual_attention.yaml --config ./configs/cine_tpo_resnet50.yaml --config ./configs/residual_attention_greyscale.yaml --config ./configs/training.yaml --model.num_frames 15 --data.batch_size 1 --version default
```
