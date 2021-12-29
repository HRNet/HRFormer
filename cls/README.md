# HRFormer for Classification


## Introduction
This is the official implementation of [High-Resolution Transformer (HRT)]() for image classification. We present a High-Resolution Transformer (HRT) that learns high-resolution repre-sentations for dense prediction tasks, in contrast to the original Vision Transformerthat produces low-resolution representations and has high memory and computa-tional cost. We take advantage of the multi-resolution parallel design introduced inhigh-resolution convolutional networks (HRNet), along with local-window self-attention that performs self-attention over small non-overlapping image windows,for improving the memory and computation efficiency. In addition, we introduce aconvolution into the FFN to exchange information across the disconnected imagewindows. We demonstrate the effectiveness of the High-Resolution Transformeron human pose estimation and semantic segmentation tasks.

![teaser](figures/HRT_arch5.png)
## ImageNet-1K trained models

| Backbone | acc@1 | acc@5 | #params | FLOPs | ckpt | log | script |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| HRT-T | 78.6% | 94.2% | 8.0M | 1.83G |[ckpt](https://1drv.ms/u/s!Ai-PFrdirDvwj1UXGB63dBVVOuLO?e=ZLOY7r) | [log](https://1drv.ms/t/s!Ai-PFrdirDvwj1S0MH9FzWCwzzxE?e=6p1Q3X) | [script](./configs/hrt/hrt_tiny.yaml)
| HRT-S | 81.2% | 95.6% | 13.5M | 3.56G |[ckpt](https://1drv.ms/u/s!Ai-PFrdirDvwj1cc3tSp4kIKI_JH?e=bHW7xj) | [log](https://1drv.ms/t/s!Ai-PFrdirDvwj1l2RxNkcb6lmGF3?e=hZ9A1K) | [script](./configs/hrt/hrt_small.yaml)
| HRT-B | 82.8% | 96.3% | 50.3M | 13.71G |[ckpt](https://1drv.ms/u/s!Ai-PFrdirDvwj1iNZngTF7PEyik9?e=fv8CG6) | [log](https://1drv.ms/t/s!Ai-PFrdirDvwj1aBKjc1mKQCkwen?e=spYZOe) | [script](./configs/hrt/hrt_base.yaml) |


## Getting Started

### Install
- Clone this repo:

```bash
git clone https://github.com/PkuRainBow/HRT-Cls.git
cd HRT-Cls
```

- Create a conda virtual environment and activate it:

```bash
conda create -n hrt python=3.7 -y
conda activate hrt
```

- Install `CUDA==10.1` with `cudnn7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch==1.7.1` and `torchvision==0.8.2` with `CUDA==10.1`:

```bash
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
```

- Install `timm==0.3.2`:

```bash
pip install timm==0.3.2
```

- Install `Apex`:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

- Install other requirements:

```bash
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8
pip install git+https://github.com/hsfzxjy/mocona@v0.1.0
```

### Data preparation

We use standard ImageNet dataset. You can download it from http://image-net.org/. We provide the following two ways to load data:

- For standard folder dataset, move validation images to labeled sub-folders. The file structure should look like:
  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```
- To boost the slow speed when reading images from massive small files, we also support zipped ImageNet, which includes
  four files:
    - `train.zip`, `val.zip`: which store the zipped folder for train and validate splits.
    - `train_map.txt`, `val_map.txt`: which store the relative path in the corresponding zip file and ground truth label. Make sure the data folder looks like this:

  ```bash
  $ tree data
  data
  └── ImageNet-Zip
      ├── train_map.txt
      ├── train.zip
      ├── val_map.txt
      └── val.zip
  
  $ head -n 5 data/ImageNet-Zip/val_map.txt
  ILSVRC2012_val_00000001.JPEG	65
  ILSVRC2012_val_00000002.JPEG	970
  ILSVRC2012_val_00000003.JPEG	230
  ILSVRC2012_val_00000004.JPEG	809
  ILSVRC2012_val_00000005.JPEG	516
  
  $ head -n 5 data/ImageNet-Zip/train_map.txt
  n01440764/n01440764_10026.JPEG	0
  n01440764/n01440764_10027.JPEG	0
  n01440764/n01440764_10029.JPEG	0
  n01440764/n01440764_10040.JPEG	0
  n01440764/n01440764_10042.JPEG	0
  ```
### Evaluation

To evaluate a pre-trained `HRT` on ImageNet val, run:

```bash
bash run_eval.sh <config-file-name> <checkpoint> <imagenet-path> 
```

For example, to evaluate the `HRT-Base` with a single GPU:

```bash
bash run_eval.sh hrt/hrt_base hrt_base.pth <imagenet-path>
```

### Training from scratch

To train a `HRT` on ImageNet from scratch, run:

```bash
bash run_dish.sh <config-file> <imagenet-path>
```

For example, to train `HRT` with 8 GPU on a single node for 300 epochs, run:

`HRT-Tiny`:

```bash
bash run_dist.sh hrt/hrt_tiny <imagenet-path>
```

`HRT-Small`:

```bash
bash run_dist.sh hrt/hrt_small <imagenet-path>
```

`HRT-Base`:

```bash
bash run_dist.sh hrt/hrt_base <imagenet-path>
```
### Number of Parameters and FLOPs

To measure the number of parameters and FLOPs, run:

```bash
bash run_flops.sh <config-file> <dataset-path> <dataset-name>
```

## Citation

If you find this project useful in your research, please consider cite:

```
@article{YuanFHZCW21,
  title={HRT: High-Resolution Transformer for Dense Prediction},
  author={Yuhui Yuan and Rao Fu and Lang Huang and Chao Zhang and Xilin Chen and Jingdong Wang},
  booktitle={arXiv},
  year={2021}
}
```
