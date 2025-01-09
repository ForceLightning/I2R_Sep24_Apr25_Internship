# A*STAR/I2R Sep 2024 - Apr 2025 Internship
[![docs](https://github.com/ForceLightning/I2R_Sep24_Apr25_Internship/actions/workflows/docs.yml/badge.svg)](https://github.com/ForceLightning/I2R_Sep24_Apr25_Internship/actions/workflows/docs.yml)

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
├── Conv1D.ipynb
├── Pipfile
├── Pipfile.lock
├── README.md
├── checkpoints                                     # A symlink may be used for this.
│   ├── cine
│   ├── lge
│   ├── residual-attention
│   ├── tscsenet
│   ├── sota
│   ├── two-plus-one
│   ├── two-stream
│   └── urr-residual-attention
├── configs
│   └── *.yaml
├── data                                            # A symlink may be used for this.
│   ├── Indices                                     #   Used for splitting the train/val dataset.
│   ├── test                                        #   Test dataset.
│   │   ├── Cine                                    #       CINE image data
│   │   ├── LGE                                     #       LGE image data
│   │   └── masks                                   #       Annotated masks
│   └── train_val                                   #   Train/Val dataset.
│       ├── Cine                                    #       CINE image data
│       ├── LGE                                     #       LGE image data
│       └── masks                                   #       Annotated masks
├── dataset
├── docs
│   ├── Makefile
│   ├── build
│   ├── make.bat
│   └── source
│       ├── conf.py
│       └── index.rst
├── pretrained_models
│   ├── distance_measures_regressor.pth             # Used for Vivim SOTA model.
│   └── PNSPlus.pth                                 # May or may not be used for PNS+ SOTA model.
├── pyrightconfig.json
├── pytest.ini
├── requirements*.txt
├── src
└── thirdparty
    ├── TransUNet
    ├── VPS
    ├── fla_net
    └── vivim
```

# Installation
Clone the repository with:
```sh
git clone --recursive https://github.com/ForceLightning/I2R_Sep24_Apr25_Internship.git
```
> [!NOTE]
> Ensure that the `--recursive` flag is set if third-party modules are needed.

Install the dependencies from `Pipfile.lock` or `requirements.txt`. Note that the CUDA version used is 12.1, modify it as necessary.
```sh
pipenv sync
# or
pip install -r requirements.txt
```
Optionally, install the dependencies for developement and the third-party libraries.
```sh
pipenv install -d
pipenv install --categories transunet && pipenv install --categories vivim && pipenv install --categories transunet && pipenv install --categories vps
# or
pip install -r requirements-dev.txt
pip install -r requirements-thirdparty.txt
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

## `torch-scatter`
Install PyTorch Scatter with:
```sh
PIP_FIND_LINKS="https://data.pyg.org/whl/torch-${torch_version}+${CUDA}.html" pipenv install torch-scatter
# or
pip install torch-scatter -f https://data.pyg.org/whl/torch-${torch_version}+${CUDA}.html
```
where `${torch_version}` is the pytorch version installed (check the requirements.txt or Pipfile) and `${CUDA}` is either `cpu`, `cu118`, `cu121`, or `cu124` depending on the PyTorch installation. See [PyTorch Scatter readme](https://github.com/rusty1s/pytorch_scatter/?tab=readme-ov-file#installation) for more info.

## Docker
1. Ensure that the prerequisite NVIDIA drivers are installed.
2. Ensure that the NVIDIA Container Toolkit is installed, with docker configured as necessary according to the [documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
3. Build the docker image with the following:
```sh
sudo docker build -t i2r-internship .
```
4. Usage is similar to the section below, but commands can be prepended by `sudo docker run -it --gpus "<GPUS_TO_USE>" -v $(pwd):/code/ -v <DATA_DIR>:/code/data -v <CHECKPOINT_DIR>:/code/checkpoints i2r-internship:latest` where `<GPUS_TO_USE>` can be `all`, `device=0`, etc., while the `DATA_DIR` and `CHECKPOINT_DIR` paths should also be set to the appropriate system file paths.

# Usage
Some default configurations are included in the `./configs/` directory, which will be used as modular pieces to construct the complete training/validation configuration.

> [!NOTE]
> Ensure that the environment variable `PYTHONPATH` is set to `./src/`.
> This can be done with:
> ```sh
> export PYTHONPATH="src/:thirdparty/VPS"
> ```
> Alternatively, set it in a `.env` file if using pipenv, or inside the `activate` script in the virtualenv `bin` directory.

## Description of config modules
- `cine.yaml`, `lge.yaml`, `two_stream.yaml`, `two_plus_one.yaml`, `residual_attention.yaml`, `urr_residual_attention.yaml`: Incomplete defaults for initialisation.
- `training.yaml`, `testing.yaml`, `training_no_checkpointing.yaml`: Overrides for training or validation/testing/quick run modes. This must be the last config file loaded in.
- `*_greyscale.yaml`, `*_rgb.yaml`: Defaults for handling either RGB images or greyscale images as inputs.
- `cine_tpo_resnet50.yaml`, `cine_tpo_senet154.yaml`: Defaults for the ResNet50 and SENet154 backbones for the CINE, TwoStream, TwoPlusOne R(2D+1D), and Attention tasks.

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
python -m attention_unet fit --config ./configs/residual_attention.yaml --config ./configs/cine_tpo_resnet50.yaml --config ./configs/residual_attention_greyscale.yaml --config ./configs/training.yaml --model.num_frames 15 --data.batch_size 2 --version default
```
## Third-party modules
SOTA methods can be run using the following python modules:
- `sota.afb_urr`
- `sota.flanet`
- `sota.pns`
- `sota.vivim`
