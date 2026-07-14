## You Only Look Intensity Once: Event-Driven Long-Term High-Speed Object Detection

[Wen Dong](https://wind1117.github.io/), [Haiyang Mei](https://mhaiyang.github.io/), Yinglian Ji, Yutong Jiang, Ziqi Wei, [Shengfeng He](https://shengfenghe.github.io/), [Xin Yang](https://xinyangdut.github.io/)

[[Paper](https://link.springer.com/article/10.1007/s11263-026-02749-8)]
[[Project Page](https://wind1117.github.io/publication/2026-IJCV-EventDet)]

### Abstract
In this work, we propose the DPNet, a novel learning-based detector that requires only a single RGB frame at the beginning of a sequence to detect high-speed objects over a 5-second duration, 25 times longer than prior methods.

### Requirements
- Python 3.8
- Pytorch 2.1.0
- Torchvision 0.16.0
- CUDA 11.8

### Dataset
The dataset can be obtained [here]().

### Evaluation
Download trained model `dpnet.pt` ([here](https://pan.baidu.com/s/1ApLQNyJ5AmLQ5UnplwlUiQ?pwd=c4aa)), and then run `val.py`.

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