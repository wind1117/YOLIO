from ultralytics.utils import colorstr

from data.coco_pku import COCOPKU_Test


def build_pku_test_dataset(cfg, img_path, batch, data, mode="test", rect=False, stride=32, multi_modal=False):
    """Build PKU-DAVIS-SOD Dataset for testing."""
    return COCOPKU_Test(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=False,
        hyp=cfg,
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )