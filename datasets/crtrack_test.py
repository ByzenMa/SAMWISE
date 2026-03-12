"""CRTrack test data loader (three-view video streams)."""

from pathlib import Path
import csv
import random
import pickle

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask
from torch.utils.data import Dataset

from datasets.transform_utils import FrameSampler, make_coco_transforms


class CRTrackTestDataset(Dataset):
    """
    CRTrack three-view dataset.

    The loader follows the MeViS-style target format so it can be used by the
    existing training pipeline. Three synchronized views are packed into one
    clip by concatenating frames along temporal dimension:
    - input images: [3 * T, 3, H, W]
    - target fields (boxes/masks/valid/labels/frames_idx): length 3 * T
    - target['view_ids']: 0/1/2 marks view1/view2/view3 for each frame
    """

    VIEW_NAMES = ("view1", "view2", "view3")

    def __init__(self, root: Path, transforms, num_frames: int):
        self.root = Path(root)
        self._transforms = transforms
        self.num_frames = num_frames

        self.images_root = self.root / "images" / "train"
        self.cross_view_root = self.root / "ids_with_text_cross_view"

        self.metas = []
        self.view_data_cache = {}
        self._prepare_metas()

        print("\n clip num: ", len(self.metas))
        print("\n")

    def _prepare_metas(self):
        csv_files = sorted(self.cross_view_root.glob("*/*/*_id_match_texts*.csv"))

        selected_csv = {}
        for csv_file in csv_files:
            clip_key = str(csv_file.parent)
            if clip_key not in selected_csv or csv_file.name.endswith("_with_txt.csv"):
                selected_csv[clip_key] = csv_file

        for csv_path in selected_csv.values():
            scene = csv_path.parents[1].name
            clip = csv_path.parent.name
            pkl_dir = self.images_root / scene / clip

            view_pkls = {
                "view1": pkl_dir / f"{scene}_View1_reprompt_rle.pkl",
                "view2": pkl_dir / f"{scene}_View2_reprompt_rle.pkl",
                "view3": pkl_dir / f"{scene}_View3_reprompt_rle.pkl",
            }
            if not all(path.exists() for path in view_pkls.values()):
                continue

            frame_ids = self._get_frame_ids(view_pkls)
            if len(frame_ids) == 0:
                continue

            with open(csv_path, "r", encoding="utf-8-sig", newline="") as fp:
                reader = csv.DictReader(fp)
                for row in reader:
                    if not self._has_valid_view_ids(row):
                        continue

                    caption = self._build_caption(row)
                    self.metas.append(
                        {
                            "scene": scene,
                            "clip": clip,
                            "caption": caption,
                            "view_obj_ids": {
                                "view1": int(row["view1"]),
                                "view2": int(row["view2"]),
                                "view3": int(row["view3"]),
                            },
                            "frame_ids": frame_ids,
                            "view_pkls": {k: str(v) for k, v in view_pkls.items()},
                        }
                    )

    @staticmethod
    def _has_valid_view_ids(row):
        try:
            int(row["view1"])
            int(row["view2"])
            int(row["view3"])
            return True
        except (KeyError, TypeError, ValueError):
            return False

    @staticmethod
    def _build_caption(row):
        if row.get("text", "").strip():
            caption = row["text"]
        else:
            txts = [row.get("view1_txt", ""), row.get("view2_txt", ""), row.get("view3_txt", "")]
            txts = [x.strip() for x in txts if x and x.strip()]
            caption = " ; ".join(txts) if txts else "unclear"
        return " ".join(caption.lower().split())

    def _get_view_data(self, pkl_path):
        pkl_path = str(pkl_path)
        if pkl_path not in self.view_data_cache:
            with open(pkl_path, "rb") as fp:
                self.view_data_cache[pkl_path] = pickle.load(fp)
        return self.view_data_cache[pkl_path]

    def _get_frame_ids(self, view_pkls):
        frame_sets = []
        for view_name in self.VIEW_NAMES:
            view_data = self._get_view_data(view_pkls[view_name])
            frame_sets.append(set(view_data.keys()))
        common = sorted(frame_sets[0].intersection(frame_sets[1]).intersection(frame_sets[2]))
        return common

    @staticmethod
    def _mask_to_box(mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

    @staticmethod
    def _decode_obj_mask(view_data, frame_id, obj_id):
        frame_dict = view_data.get(frame_id, {})
        if obj_id not in frame_dict:
            return None

        rle = frame_dict[obj_id].get("rle", None)
        if rle is None:
            return None

        if isinstance(rle.get("counts", None), str):
            rle = dict(rle)
            rle["counts"] = rle["counts"].encode("utf-8")

        return coco_mask.decode(rle)

    @staticmethod
    def _infer_hw(view_data):
        for frame_dict in view_data.values():
            for obj_data in frame_dict.values():
                size = obj_data.get("rle", {}).get("size", None)
                if size is not None and len(size) == 2:
                    return int(size[0]), int(size[1])
        return 1080, 1920

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        instance_check = False
        while not instance_check:
            meta = self.metas[idx]
            frame_ids = meta["frame_ids"]

            center_pos = random.randint(0, len(frame_ids) - 1)
            sample_pos = FrameSampler.sample_global_frames(center_pos, len(frame_ids), self.num_frames)
            sample_frame_ids = [frame_ids[p] for p in sample_pos]

            imgs, labels, boxes, masks, valid = [], [], [], [], []
            frames_idx, view_ids = [], []
            per_view_num_frames = []

            for view_id, view_name in enumerate(self.VIEW_NAMES):
                view_data = self._get_view_data(meta["view_pkls"][view_name])
                h, w = self._infer_hw(view_data)
                obj_id = meta["view_obj_ids"][view_name]

                for frame_id in sample_frame_ids:
                    decoded = self._decode_obj_mask(view_data, frame_id, obj_id)
                    if decoded is None:
                        mask_np = np.zeros((h, w), dtype=np.float32)
                    else:
                        mask_np = (decoded > 0).astype(np.float32)

                    # CRTrack_test release in this repo contains RLE masks only.
                    # Build a pseudo RGB frame from mask to keep transform API aligned.
                    img = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L").convert("RGB")

                    label = torch.tensor(0)
                    if (mask_np > 0).any():
                        y1, y2, x1, x2 = self._mask_to_box(mask_np)
                        box = torch.tensor([x1, y1, x2, y2], dtype=torch.float)
                        valid.append(1)
                    else:
                        box = torch.tensor([0, 0, 0, 0], dtype=torch.float)
                        valid.append(0)

                    imgs.append(img)
                    labels.append(label)
                    boxes.append(box)
                    masks.append(torch.from_numpy(mask_np))
                    frames_idx.append(frame_id)
                    view_ids.append(view_id)

                per_view_num_frames.append(len(sample_frame_ids))

            labels = torch.stack(labels, dim=0)
            boxes = torch.stack(boxes, dim=0)
            masks = torch.stack(masks, dim=0)

            width, height = imgs[-1].size
            boxes[:, 0::2].clamp_(min=0, max=width)
            boxes[:, 1::2].clamp_(min=0, max=height)

            target = {
                "frames_idx": torch.tensor(frames_idx, dtype=torch.long),
                "labels": labels,
                "boxes": boxes,
                "masks": masks,
                "valid": torch.tensor(valid, dtype=torch.long),
                "view_ids": torch.tensor(view_ids, dtype=torch.long),
                "per_view_num_frames": torch.tensor(per_view_num_frames, dtype=torch.long),
                "caption": meta["caption"],
                "orig_size": torch.as_tensor([int(height), int(width)]),
                "size": torch.as_tensor([int(height), int(width)]),
                "video_id": f"{meta['scene']}/{meta['clip']}",
                "exp_id": idx,
            }

            imgs, target = self._transforms(imgs, target)
            imgs = torch.stack(imgs, dim=0)

            if torch.any(target["valid"] == 1):
                instance_check = True
            else:
                idx = random.randint(0, self.__len__() - 1)

        return imgs, target


def build(image_set, args):
    root = Path(args.crtrack_path)
    assert root.exists(), f"provided CRTrack path {root} does not exist"

    if image_set not in ["train", "test", "valid", "valid_u"]:
        raise ValueError(f"Unsupported image_set for CRTrack_test: {image_set}")

    transforms_set = "train" if image_set == "train" else "valid_u"
    dataset = CRTrackTestDataset(
        root=root / "CRTrack_In-domain",
        transforms=make_coco_transforms(transforms_set, max_size=args.max_size, resize=args.augm_resize),
        num_frames=args.num_frames,
    )
    return dataset
