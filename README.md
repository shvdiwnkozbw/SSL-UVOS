# Betrayed by Attention: A Simple yet Effective Approach for Self-supervised Video Object Segmentation
Official pytorch implementation of the paper [Betrayed by Attention: A Simple yet Effective Approach for Self-supervised Video Object Segmentation](https://arxiv.org/abs/2311.17893). Currently, only the inference code is available. Stay tuned for the training code.

## Overview
In this paper, we propose a simple yet effective approach for self-supervised video object segmentation (VOS). Our key insight is that the inherent structural dependencies present in DINO-pretrained Transformers can be leveraged to establish robust spatio-temporal correspondences in videos. Furthermore, simple clustering on this correspondence cue is sufficient to yield competitive segmentation results. We develop a simplified architecture that capitalizes on the emerging objectness from DINO-pretrained Transformers. Specifically, we first introduce a single spatio-temporal Transformer block to process the frame-wise DINO features and establish spatio-temporal dependencies in the form of self-attention. Subsequently, utilizing these attention maps, we implement hierarchical clustering to generate object segmentation masks. Our method demonstrates state-of-the-art performance across multiple unsupervised VOS benchmarks and particularly excels in complex real-world multi-object video segmentation tasks such as DAVIS-17-Unsupervised and YouTube-VIS-19.

![teaser](Figure/teaser.png)

[Project Page](coming soon) [[arXiv]](https://arxiv.org/abs/2311.17893) [[PDF]](https://arxiv.org/pdf/2311.17893.pdf)

## Usage

### Requirements
- pytorch 1.12
- torchvision
- opencv-python
- cvbase
- einops
- kornia
- tensorboardX


### Data preparation
- Download the DAVIS-2017-Unsupervised dataset from the [official website](https://davischallenge.org/davis2017/code.html#unsupervised).

### Pretrained Model
- Download our model checkpoint based on DINO-ViT-S/8 pretrained on YT-VOS 2016 [[google drive]](https://drive.google.com/file/d/1UhSPueJGpV4di9SVlZDmz0KWkuigQApA/view?usp=sharing)

### Downstream Evaluation
After downloading the pretrained model, you can run the inference code by executing:
```python
bash start_eval.sh
```
Ensure that the basepath and davis_path is set to your DAVIS data path. This will provide you with the final performance on DAVIS-2017-Unsupervised. We set the default resolution ```320 480``` for tradeoff between segmentation accuracy and inference speed. Feel free to set a smaller resolution, e.g., ```192 384```, for faster inference and evaluation or larger resolution for higher accuracy.

## Visualization
We have visualized results on video sequences with occlusions. Our model successfully handles partial or complete object occlusion, where an object disappears in some frames and reappears in later ones.
![vis](Figure/vis.png)

## Acknowledgement
Our code is partly based on the implementation of [Motino Grouping](https://github.com/charigyang/motiongrouping). We sincerely thank the authors for their significant contribution. If you have any questions regarding the paper or code, please don't hesitate to send us an email or raise an issue.


## Citation
If our code assists your work, please consider citing:
```
@article{ding2023betrayed,
  title={Betrayed by Attention: A Simple yet Effective Approach for Self-supervised Video Object Segmentation},
  author={Ding, Shuangrui and Qian, Rui and Xu, Haohang and Lin, Dahua and Xiong, Hongkai},
  journal={arXiv preprint arXiv:2311.17893},
  year={2023}
}
```
