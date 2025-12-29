# YOLIO
The complete implementation and dataset will be publicly available soon.

## You Only Look Intensity Once: Event-Driven Long-Term High-Speed Object Detection

[Author List]()

[[Paper]()]
[[Project Page]()]

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
@article{publication,
  title={yolio},
  author={authors},
  journal={},
  volume={},
  number={},
  year={}
}
```