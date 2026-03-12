"""CRTrack dataset loader (three-view video streams with real RGB frames)."""

from pathlib import Path
import csv
import logging
import random
import pickle

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask
from torch.utils.data import Dataset

from datasets.transform_utils import FrameSampler, make_coco_transforms


LOGGER = logging.getLogger(__name__)


class CRTrackTestDataset(Dataset):
    """CRTrack three-view dataset with separated stream outputs.

    Returns:
        view_imgs: list[Tensor], len=3, each tensor [T, C, H, W]
        view_targets: list[dict], len=3, aligned with view_imgs
    """

    VIEW_NAMES = ("view1", "view2", "view3")
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(self, root: Path, transforms, num_frames: int, strict_rgb_check: bool = True):
        self.root = Path(root)
        self._transforms = transforms
        self.num_frames = num_frames
        self.strict_rgb_check = strict_rgb_check

        self.images_root = self.root / "images" / "train"
        self.cross_view_root = self.root / "ids_with_text_cross_view"

        self.metas = []
        self.view_data_cache = {}
        self.rgb_index_cache = {}
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

        dropped_for_missing_rgb = 0
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

            rgb_index = self._build_rgb_index(scene, clip)
            rgb_common = set(frame_ids)
            for view_name in self.VIEW_NAMES:
                rgb_common &= set(rgb_index[view_name].keys())
            frame_ids = sorted(rgb_common)
            if self.strict_rgb_check and len(frame_ids) == 0:
                dropped_for_missing_rgb += 1
                LOGGER.warning(
                    "Drop clip %s/%s: no common real RGB frames found for all three views.",
                    scene,
                    clip,
                )
                continue

            with open(csv_path, "r", encoding="utf-8-sig", newline="") as fp:
                reader = csv.DictReader(fp)
                for row in reader:
                    if not self._has_valid_view_ids(row):
                        continue

                    caption = self._build_caption(row)
                    view_texts = {
                        "view1": " ".join(str(row.get("view1_txt", "")).lower().split()),
                        "view2": " ".join(str(row.get("view2_txt", "")).lower().split()),
                        "view3": " ".join(str(row.get("view3_txt", "")).lower().split()),
                    }
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
                            "view_texts": view_texts,
                            "view_pkls": {k: str(v) for k, v in view_pkls.items()},
                        }
                    )

        if dropped_for_missing_rgb > 0:
            LOGGER.warning("Dropped %d clip(s) due to missing real RGB frames.", dropped_for_missing_rgb)

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
        return sorted(frame_sets[0].intersection(frame_sets[1]).intersection(frame_sets[2]))

    def _build_rgb_index(self, scene, clip):
        key = (scene, clip)
        if key in self.rgb_index_cache:
            return self.rgb_index_cache[key]

        base_dir = self.images_root / scene / clip
        frame_map = {"view1": {}, "view2": {}, "view3": {}}
        if base_dir.exists():
            for file_path in base_dir.rglob("*"):
                if not file_path.is_file() or file_path.suffix.lower() not in self.IMG_EXTS:
                    continue
                view_name = self._infer_view_name(file_path.name)
                if view_name is None:
                    continue
                frame_id = self._extract_last_int_token(file_path.stem)
                if frame_id is None:
                    continue
                frame_map[view_name][frame_id] = file_path

        self.rgb_index_cache[key] = frame_map
        return frame_map

    @staticmethod
    def _infer_view_name(filename):
        lower_name = filename.lower()
        if "view1" in lower_name:
            return "view1"
        if "view2" in lower_name:
            return "view2"
        if "view3" in lower_name:
            return "view3"
        return None

    @staticmethod
    def _extract_last_int_token(stem):
        nums = []
        cur = ""
        for ch in stem:
            if ch.isdigit():
                cur += ch
            elif cur:
                nums.append(cur)
                cur = ""
        if cur:
            nums.append(cur)
        if not nums:
            return None
        return int(nums[-1])

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

    def _load_real_rgb_frame(self, meta, view_name, frame_id, h, w):
        rgb_index = self._build_rgb_index(meta["scene"], meta["clip"])
        rgb_path = rgb_index[view_name].get(int(frame_id), None)
        if rgb_path is None:
            return None

        img = Image.open(rgb_path).convert("RGB")
        if img.size != (w, h):
            img = img.resize((w, h), Image.BILINEAR)
        return img

    def _build_single_view_sample(self, meta, sample_frame_ids, view_name, exp_id):
        view_data = self._get_view_data(meta["view_pkls"][view_name])
        h, w = self._infer_hw(view_data)
        obj_id = meta["view_obj_ids"][view_name]

        imgs, labels, boxes, masks, valid = [], [], [], [], []
        for frame_id in sample_frame_ids:
            decoded = self._decode_obj_mask(view_data, frame_id, obj_id)
            if decoded is None:
                mask_np = np.zeros((h, w), dtype=np.float32)
            else:
                mask_np = (decoded > 0).astype(np.float32)

            img = self._load_real_rgb_frame(meta, view_name, frame_id, h, w)
            if img is None:
                return None, None

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

        labels = torch.stack(labels, dim=0)
        boxes = torch.stack(boxes, dim=0)
        masks = torch.stack(masks, dim=0)

        width, height = imgs[-1].size
        boxes[:, 0::2].clamp_(min=0, max=width)
        boxes[:, 1::2].clamp_(min=0, max=height)

        target = {
            "frames_idx": torch.tensor(sample_frame_ids, dtype=torch.long),
            "labels": labels,
            "boxes": boxes,
            "masks": masks,
            "valid": torch.tensor(valid, dtype=torch.long),
            "caption": meta["caption"],
            "orig_size": torch.as_tensor([int(height), int(width)]),
            "size": torch.as_tensor([int(height), int(width)]),
            "video_id": f"{meta['scene']}/{meta['clip']}/{view_name}",
            "exp_id": exp_id,
            "view_name": view_name,
            "view_text": meta.get("view_texts", {}).get(view_name, ""),
        }

        imgs, target = self._transforms(imgs, target)
        imgs = torch.stack(imgs, dim=0)
        return imgs, target

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        instance_check = False
        while not instance_check:
            meta = self.metas[idx]
            frame_ids = meta["frame_ids"]
            if len(frame_ids) == 0:
                idx = random.randint(0, self.__len__() - 1)
                continue

            center_pos = random.randint(0, len(frame_ids) - 1)
            sample_pos = FrameSampler.sample_global_frames(center_pos, len(frame_ids), self.num_frames)
            sample_frame_ids = [frame_ids[p] for p in sample_pos]

            view_imgs, view_targets = [], []
            missing_rgb = False
            for view_name in self.VIEW_NAMES:
                imgs, target = self._build_single_view_sample(meta, sample_frame_ids, view_name, idx)
                if imgs is None:
                    missing_rgb = True
                    LOGGER.warning(
                        "Skip sample %s/%s (%s): missing RGB frame in selected window %s.",
                        meta["scene"],
                        meta["clip"],
                        view_name,
                        sample_frame_ids,
                    )
                    break
                view_imgs.append(imgs)
                view_targets.append(target)

            if missing_rgb:
                idx = random.randint(0, self.__len__() - 1)
                continue

            if any(torch.any(t["valid"] == 1) for t in view_targets):
                instance_check = True
            else:
                idx = random.randint(0, self.__len__() - 1)

        return view_imgs, view_targets


def build(image_set, args):
    root = Path(args.crtrack_path)
    assert root.exists(), f"provided CRTrack path {root} does not exist"

    if image_set not in ["train", "test", "valid", "valid_u"]:
        raise ValueError(f"Unsupported image_set for CRTrack_test: {image_set}")

    dataset_root = root / "CRTrack_In-domain" if (root / "CRTrack_In-domain").exists() else root

    transforms_set = "train" if image_set == "train" else "valid_u"
    dataset = CRTrackTestDataset(
        root=dataset_root,
        transforms=make_coco_transforms(transforms_set, max_size=args.max_size, resize=args.augm_resize),
        num_frames=args.num_frames,
        strict_rgb_check=True,
    )
    return dataset
