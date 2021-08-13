# HRFormer for Semantic Segmentation


## Introduction
This is the official implementation of [High-Resolution Transformer (HRT)]() for semantic segmentation. We present a High-Resolution Transformer (HRT) that learns high-resolution repre-sentations for dense prediction tasks, in contrast to the original Vision Transformerthat produces low-resolution representations and has high memory and computa-tional cost. We take advantage of the multi-resolution parallel design introduced inhigh-resolution convolutional networks (HRNet), along with local-window self-attention that performs self-attention over small non-overlapping image windows,for improving the memory and computation efficiency. In addition, we introduce aconvolution into the FFN to exchange information across the disconnected imagewindows. We demonstrate the effectiveness of the High-Resolution Transformeron human pose estimation and semantic segmentation tasks.

![teaser](imgs/HRT_arch5.png)


## Cityscapes
Performance on the Cityscapes dataset. The models are trained and tested with input size of 512x1024 and 1024x2048 respectively. 

Methods | Backbone | Window Size | Train Set | Test Set | Iterations | Batch Size | OHEM | mIoU | mIoU (Multi-Scale) | Log | ckpt | script |
| :---- | :------- | :---: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |:--: |
OCRNet | HRT-S | 7x7 | Train | Val | 80000 | 8 | Yes | 80.0 | 81.0 | [log](https://1drv.ms/u/s!Ai-PFrdirDvwj3K-rPMQ6sHNV-Fe?e=D3IbNn) | [ckpt](https://1drv.ms/u/s!Ai-PFrdirDvwj3Wsg-_ApKUAEUft?e=BnhLal) | [script](./scripts/cityscapes/hrt/run_hrt_small_ocr_v2_ohem.sh) |
OCRNet | HRT-B | 7x7 | Train | Val | 80000 | 8 | Yes | 81.4 | 82.0 | [log](https://1drv.ms/u/s!Ai-PFrdirDvwj3NtH1LBB0w6yCO3?e=p4v29Z) | [ckpt](https://1drv.ms/u/s!Ai-PFrdirDvwj3zEMdYLM8nZ5gXN?e=v7ehnB) |[script](./scripts/cityscapes/hrt/run_hrt_base_ocr_v2_ohem.sh) |
OCRNet | HRT-B | 15x15 | Train | Val | 80000 | 8 | Yes | 81.9 | 82.6 | [log](https://1drv.ms/u/s!Ai-PFrdirDvwkAlyBb4tGcxSjF_A?e=diIDCV) | [ckpt](https://1drv.ms/u/s!Ai-PFrdirDvwkAp3LjwI-7Csmh0K?e=K1zXrn)|[script](./scripts/cityscapes/hrt/run_hrt_base_ocr_v2_ohem_w15.sh) | 

## PASCAL-Context

The models are trained with the input size of 520x520, and tested with original size.

Methods | Backbone | Window Size | Train Set | Test Set | Iterations | Batch Size | OHEM | mIoU | mIoU (Multi-Scale) | Log | ckpt | script |
| :---- | :------- | :---: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |:--: |
OCRNet | HRT-S | 7x7 | Train | Val | 60000 | 16 | Yes | 53.8 | 54.6 | [log](https://1drv.ms/u/s!Ai-PFrdirDvwj306lzvnI4s5U43l?e=J9mCfg) | [ckpt](https://1drv.ms/u/s!Ai-PFrdirDvwkADkevlhIuUrPC1T?e=hcDx5S) | [script](./scripts/pascal_context/hrt/run_hrt_small_ocr_v2_ohem.sh) |
OCRNet | HRT-B | 7x7 | Train | Val | 60000 | 16 | Yes | 56.3 | 57.1 | [log](https://1drv.ms/u/s!Ai-PFrdirDvwj3_0tiJZqL7HWPv1?e=6ilX0Z) | [ckpt](https://1drv.ms/u/s!Ai-PFrdirDvwkAMUzRnCGmAxEehJ?e=HrCQ9c) |[script](./scripts/pascal_context/hrt/run_hrt_base_ocr_v2_ohem.sh) |
OCRNet | HRT-B | 15x15 | Train | Val | 60000 | 16 | Yes | 57.6 | 58.5 | [log](https://1drv.ms/u/s!Ai-PFrdirDvwj3kphBj2FusLylDg?e=qZSrpp) | [ckpt](https://1drv.ms/u/s!Ai-PFrdirDvwkAIBAkrOlPp_T1YT?e=DeHMdo)|[script](./scripts/pascal_context/hrt/run_hrt_base_ocr_v2_ohem_w15.sh) | 

## COCO-Stuff

The models are trained with input size of 520x520, and tested with original size.

Methods | Backbone | Window Size | Train Set | Test Set | Iterations | Batch Size | OHEM | mIoU | mIoU (Multi-Scale) | Log | ckpt | script |
| :---- | :------- | :---: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |:--: |
OCRNet | HRT-S | 7x7 | Train | Val | 60000 | 16 | Yes | 37.9 | 38.9 | [log](https://1drv.ms/u/s!Ai-PFrdirDvwj3ayL8oHrwsjRP1U?e=uOa0NC) | [ckpt](https://1drv.ms/u/s!Ai-PFrdirDvwj3tbt5BhdCrsu6lK?e=up2HUI) | [script](./scripts/coco_stuff/hrt/run_hrt_small_ocr_v2_ohem.sh) |
OCRNet | HRT-B | 7x7 | Train | Val | 60000 | 16 | Yes | 41.6 | 42.5 | [log](https://1drv.ms/u/s!Ai-PFrdirDvwj3iKM2xyDk-6jnJd?e=HL5s7d) | [ckpt](https://1drv.ms/u/s!Ai-PFrdirDvwkAFAYKZm2wL9C6KL?e=AZiXLK) |[script](./scripts/coco_stuff/hrt/run_hrt_base_ocr_v2_ohem.sh) |
OCRNet | HRT-B | 15x15 | Train | Val | 60000 | 16 | Yes | 42.4 | 43.3 | [log](https://1drv.ms/u/s!Ai-PFrdirDvwj3RtsUasPSb4nhL_?e=WUBe74) | [ckpt](https://1drv.ms/u/s!Ai-PFrdirDvwj37Np48Gpb-Pjowu?e=iwL5UA)|[script](./scripts/coco_stuff/hrt/run_hrt_base_ocr_v2_ohem_w15.sh) |  

## ADE20K

The models are trained with input size of 520x520, and tested with original size. The results with window size 15x15 will be updated latter.

Methods | Backbone | Window Size | Train Set | Test Set | Iterations | Batch Size | OHEM | mIoU | mIoU (Multi-Scale) | Log | ckpt | script |
| :---- | :------- | :---: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |:--: |
OCRNet | HRT-S | 7x7 | Train | Val | 150000 | 8 | Yes | 44.0 | 45.1 | [log](https://1drv.ms/u/s!Ai-PFrdirDvwj3EehoEZZUDMX0NU?e=F8HAQi) | [ckpt](https://1drv.ms/u/s!Ai-PFrdirDvwj28i74aN6_Zk4clX?e=CWGOcd) | [script](./scripts/ade20k/hrt/run_hrt_small_ocr_v2_ohem.sh) |
OCRNet | HRT-B | 7x7 | Train | Val | 150000 | 8 | Yes | 46.3 | 47.6 | [log](https://1drv.ms/u/s!Ai-PFrdirDvwj265qyyZ74PKjfqm?e=Cj7TGl) | [ckpt](https://1drv.ms/u/s!Ai-PFrdirDvwj3epNJ-QFF33tZtr?e=df3fQk) |[script](./scripts/ade20k/hrt/run_hrt_base_ocr_v2_ohem.sh) |
OCRNet | HRT-B | 13x13 | Train | Val | 150000 | 8 | Yes | 48.7 | 50.0 | [log](https://1drv.ms/u/s!Ai-PFrdirDvwkAjmpl5jj0sXz2v-?e=sfhyI4) | [ckpt](https://1drv.ms/u/s!Ai-PFrdirDvwj3oTs_gVPzFDjdyU?e=yjGRKz)|[script](./scripts/ade20k/hrt/run_hrt_base_ocr_v2_ohem_w13.sh) |
OCRNet | HRT-B | 15x15 | Train | Val | 150000 | 8 | Yes | - | - | - | - | - | 



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
