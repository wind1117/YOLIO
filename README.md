# YOLIO
The source code and dataset will be publicly available soon.

## You Only Look Intensity Once: Event-Driven Long-Term High-Speed Object Detection

[Wen Dong](https://wind1117.github.io/), [Haiyang Mei](https://mhaiyang.github.io/), Yinglian Ji, Yutong Jiang, Ziqi Wei, [Shengfeng He](https://shengfenghe.github.io/), [Xin Yang](https://xinyangdut.github.io/)

[[Paper](https://link.springer.com/article/10.1007/s11263-026-02749-8)]
[[Project Page](https://wind1117.github.io/publication/2026-IJCV-EventDet)]

### Abstract
In this work, we propose the DPNet, a novel learning-based detector that requires only a single RGB frame at the beginning of a sequence to detect high-speed objects over a 5-second duration, 25 times longer than prior methods.

### Requirements
- Python 3.8.20
- Pytorch 1.12.1
- Torchvision 0.13.1
- CUDA 10.2.0
- numpy 1.24.4

### Dataset
The dataset can be obtained [here]().

### Model Weight
The pretrained model weight can be obtained [here]().

### Training
Run:
```
python train.py
```

### Evaluation
Run:
```
python val.py
```

### Inference
Run:
```
python infer.py
```

### Citation
Please cite our paper if you find it is useful:
```
@article{dong2026you,
  title={You Only Look Intensity Once: Event-Driven Long-Term High-Speed Object Detection},
  author={Dong, Wen and Mei, Haiyang and Ji, Yinglian and Jiang, Yutong and Wei, Ziqi and He, Shengfeng and Yang, Xin},
  journal={International Journal of Computer Vision},
  volume={134},
  number={4},
  pages={149},
  year={2026},
  publisher={Springer}
}
```