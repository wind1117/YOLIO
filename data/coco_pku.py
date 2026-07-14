import glob
import os
from pathlib import Path
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM
from ultralytics.utils.ops import resample_segments

from ultralytics.data.base import BaseDataset
from ultralytics.data.utils import (
    HELP_URL,
    LOGGER,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image_label,
)

from data.augment import (
    Compose,
    Format,
    Instances,
    LetterBox,
)

# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = "1.0.3"

class COCOPKU_Test(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes the YOLODataset with optional configurations for segments and keypoints."""
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, **kwargs)

        # self.cut_len = 8
        self.vids_list = list(Path(self.img_path).iterdir())  

    def __getitem__(self, index):
        png_vid_path = Path(self.vids_list[index])
        png_file_list = sorted(glob.glob(str(png_vid_path / '**' / '*.*'), recursive=True))
        vid_len = len(png_file_list)
        vid_dicts = []  # a list contain whole video in time order
        for i in range(vid_len):  # fid, fid+1, fid+2, ... fid+cut_len-1
            png_file_i = png_file_list[i]
            lbl_file_i = f'{os.sep}labels{os.sep}'.join(png_file_i.rsplit(f'{os.sep}images{os.sep}', 1)).rsplit('.', 1)[0] + '.txt'

            lbl_i = self.path_to_txt_label(lbl_file_i)
            lbl_i.pop('shape', None)
            lbl_i['img'] = None
            lbl_i['evt'] = None
            lbl_i['ori_shape'] = cv2.imread(png_file_i, cv2.IMREAD_UNCHANGED).shape[:2]
            lbl_i['imgsz'] = self.imgsz
            # update labels info
            lbl_i = self.update_labels_info(lbl_i)
            vid_dicts.append(lbl_i)
        return vid_dicts

    def __len__(self):
        return len(self.vids_list)

    def get_sum_len(self):
        count_len = 0
        for vid_path in self.vids_list:
            count_len += len(glob.glob(str(vid_path / '**' / '*.*'), recursive=True))
        return count_len

    def update_labels_info(self, label):
        # add instance into label dict
        bboxes = label.pop('bboxes')
        segments = label.pop('segments', [])
        keypoints = label.pop('keypoints', None)
        bbox_format = label.pop('bbox_format')
        normalized = label.pop('normalized')

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # list[np.array(1000, 2)] * num_samples
            # (N, 1000, 2)
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def path_to_txt_label(txt_path):
        png_path = f'{os.sep}images{os.sep}'.join(txt_path.rsplit(f'{os.sep}labels{os.sep}', 1)).rsplit('.', 1)[0] + '.png'
        lbl_dict = {
            'im_file': png_path,
            'ev_file': f'{os.sep}eframes{os.sep}'.join(txt_path.rsplit(f'{os.sep}labels{os.sep}', 1)).rsplit('.', 1)[0] + '.png',
            # 'ev_file': f'{os.sep}events{os.sep}'.join(txt_path.rsplit(f'{os.sep}labels{os.sep}', 1)).rsplit('.', 1)[0] + '.npy',
            'shape': cv2.imread(png_path, cv2.IMREAD_UNCHANGED).shape[:2],
            'cls': [],
            'bboxes': [],
            'segments': [],
            'keypoints': None,
            'normalized': True,
            'bbox_format': 'xywh'
        }
        with open(txt_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            line = [float(i) for i in line.strip().split(' ')]
            lbl_dict['cls'].append(line[0:1])
            lbl_dict['bboxes'].append(line[1:])
        lbl_dict['cls'] = np.array(lbl_dict['cls'], dtype=np.float32).reshape(-1, 1)
        lbl_dict['bboxes'] = np.array(lbl_dict['bboxes'], dtype=np.float32).reshape(-1, 4)
        return lbl_dict

    @staticmethod
    def make_color_histo(events, frame=None, width=346, height=260):
        """
            simple display function that shows negative events as blue dots and positive as red one
            on a white background
            args :
                - events structured numpy array: timestamp, x, y, polarity.
                - img (numpy array, height x width x 3) optional array to paint event on.
                - width int.
                - height int.
            return:
                - img numpy array, height x width x 3.
        """
        if frame is None:
            frame = 255 * np.ones((height, width, 3), dtype=np.uint8)
        else:
            # if an array was already allocated just paint it grey
            frame[...] = 255

        if events.size:
            assert events['x'].max() < width, \
                "out of bound events: x = {}, w = {}".format(events['x'].max(), width)
            assert events['y'].max() < height, \
                "out of bound events: y = {}, h = {}".format(events['y'].max(), height)

            ON_index = np.where(events['polarity'] == 1)
            frame[events['y'][ON_index], events['x'][ON_index], :] = \
                [30, 30, 220] * events['polarity'][ON_index][:, None]  # red

            OFF_index = np.where(events['polarity'] == 0)
            frame[events['y'][OFF_index], events['x'][OFF_index], :] = \
                [200, 30, 30] * (events['polarity'][OFF_index] + 1)[:, None]  # blue

        return frame

    def cache_labels(self, path=Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file. Default is Path('./labels.cache').

        Returns:
            (dict): labels.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}")
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # list[np.array(1000, 2)] * num_samples
            # (N, 1000, 2)
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        return batch[0]
