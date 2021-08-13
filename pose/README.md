# HRFormer for Pose Estimation

## Introduction
This is the official implementation of [High-Resolution Transformer (HRT)]() for pose estimation. We present a High-Resolution Transformer (HRT) that learns high-resolution repre-sentations for dense prediction tasks, in contrast to the original Vision Transformerthat produces low-resolution representations and has high memory and computa-tional cost. We take advantage of the multi-resolution parallel design introduced inhigh-resolution convolutional networks (HRNet), along with local-window self-attention that performs self-attention over small non-overlapping image windows,for improving the memory and computation efficiency. In addition, we introduce aconvolution into the FFN to exchange information across the disconnected imagewindows. We demonstrate the effectiveness of the High-Resolution Transformeron human pose estimation and semantic segmentation tasks.


![teaser](./demo/resources/HRT_arch5.png)

## Results and models

### 2d Human Pose Estimation

#### Results on COCO `val2017` with detector having human AP of 56.4 on COCO `val2017` dataset

| Backbone  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR<sup>M</sup> | AR<sup>L</sup> | AR | ckpt | log | script |
| :----------------- | :-----------: | :------: | :------: | :------: | :------:| :------: | :------: |:------: |:------: | :------: |
| HRT-S  | 256x192 | 74.0% | 90.2% | 81.2% | 70.4% | 80.7% | 79.4% | [ckpt](https://1drv.ms/u/s!Ai-PFrdirDvwj2PC53KZd-7v3X0H?e=hUZ0fE) | [log](https://1drv.ms/u/s!Ai-PFrdirDvwj2Bytw64p9XJuYMt?e=Fj8brM) | [script](./configs/top_down/hrt/coco/hrt_small_coco_256x192.py) |
| HRT-S  | 384x288 | 75.6% | 90.3% | 82.2% | 71.6% | 82.5% | 80.7% | [ckpt](https://1drv.ms/u/s!Ai-PFrdirDvwj2TxlkzWYuh9CkvU?e=H50XSl) | [log](https://1drv.ms/u/s!Ai-PFrdirDvwj2FjbD4E7EQi-2n5?e=8xJqCD) | [script](./configs/top_down/hrt/coco/hrt_small_coco_384x288.py) |
| HRT-B  | 256x192 | 75.6% | 90.8% | 82.8% | 71.7% | 82.6% | 80.8% | [ckpt](https://1drv.ms/u/s!Ai-PFrdirDvwj2V-4bLd_7RkjTFW?e=L20Wit) | [log](https://1drv.ms/u/s!Ai-PFrdirDvwj2KhySyLQ-QHUQ4l?e=FEKmfr) | [script](./configs/top_down/hrt/coco/hrt_base_coco_256x192.py) |
| HRT-B  | 384x288 | 77.2% | 91.0% | 83.6% | 73.2% | 84.2% | 82.0% | [ckpt](https://1drv.ms/u/s!Ai-PFrdirDvwj2ZKrF6rWWzoRJUM?e=RCRb0p) | [log](https://1drv.ms/u/s!Ai-PFrdirDvwj100SWSwSYeZvXvL?e=Tu6Gtm) | [script](./configs/top_down/hrt/coco/hrt_base_coco_384x288.py) |


#### Results on COCO `test-dev` with detector having human AP of 56.4 on COCO `val2017` dataset

| Backbone  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR<sup>M</sup> | AR<sup>L</sup> | AR | ckpt | log | script |
| :----------------- | :-----------: | :------: | :------: | :------: | :------:| :------: | :------: |:------: |:------: | :------: |
| HRT-S  | 384x288 | 74.5% | 92.3% | 82.1% | 70.7% | 80.6% | 79.8% | [ckpt](https://1drv.ms/u/s!Ai-PFrdirDvwj2TxlkzWYuh9CkvU?e=H50XSl) | [log](https://1drv.ms/u/s!Ai-PFrdirDvwj2FjbD4E7EQi-2n5?e=8xJqCD) |  [script](./configs/top_down/hrt/coco/hrt_small_coco_384x288.py) |
| HRT-B  | 384x288 | 76.2% | 92.7% | 83.8% | 72.5% | 82.3% | 81.2% | [ckpt](https://1drv.ms/u/s!Ai-PFrdirDvwj2ZKrF6rWWzoRJUM?e=RCRb0p) | [log](https://1drv.ms/u/s!Ai-PFrdirDvwj100SWSwSYeZvXvL?e=Tu6Gtm) |  [script](./configs/top_down/hrt/coco/hrt_base_coco_384x288.py)  |

The models are first pre-trained on ImageNet-1K dataset, and then fine-tuned on COCO `val2017` dataset.


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