"""
CRTrack test data loader (three-view video streams)
"""
from pathlib import Path
import csv
import random
import pickle
import copy
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

from datasets.transform_utils import FrameSampler, make_coco_transforms


class CRTrackTestDataset(Dataset):
    """Dataset for CRTrack test split with synchronized three-view clips."""

    def __init__(self, root: Path, transforms, num_frames: int):
        self.root = Path(root)
        self._transforms = transforms
        self.num_frames = num_frames

        self.images_root = self.root / "images" / "train"
        self.cross_view_root = self.root / "ids_with_text_cross_view"

        self.metas = []
        self._prepare_metas()
        print('\n clip num: ', len(self.metas))
        print('\n')

    def _prepare_metas(self):
        csv_files = sorted(self.cross_view_root.glob("*/*/*_id_match_texts*.csv"))
        # Prefer *_with_txt.csv when both variants exist.
        selected = {}
        for csv_file in csv_files:
            clip_key = str(csv_file.parent)
            if clip_key not in selected or csv_file.name.endswith("_with_txt.csv"):
                selected[clip_key] = csv_file

        for csv_path in selected.values():
            scene = csv_path.parents[1].name
            clip = csv_path.parent.name
            pkl_dir = self.images_root / scene / clip
            view_pkls = {
                "view1": pkl_dir / f"{scene}_View1_reprompt_rle.pkl",
                "view2": pkl_dir / f"{scene}_View2_reprompt_rle.pkl",
                "view3": pkl_dir / f"{scene}_View3_reprompt_rle.pkl",
            }
            if not all(p.exists() for p in view_pkls.values()):
                continue

            with open(view_pkls["view1"], "rb") as fp:
                view1_data = pickle.load(fp)
            frame_ids = sorted(view1_data.keys())
            if len(frame_ids) == 0:
                continue

            with open(csv_path, "r", encoding="utf-8-sig", newline="") as fp:
                reader = csv.DictReader(fp)
                for row in reader:
                    txt = row.get("text", "")
                    if not txt:
                        txt_parts = [row.get("view1_txt", ""), row.get("view2_txt", ""), row.get("view3_txt", "")]
                        txt = " | ".join([t for t in txt_parts if t])

                    self.metas.append({
                        "scene": scene,
                        "clip": clip,
                        "caption": " ".join(txt.lower().split()),
                        "view_obj_ids": {
                            "view1": int(row["view1"]),
                            "view2": int(row["view2"]),
                            "view3": int(row["view3"]),
                        },
                        "frame_ids": frame_ids,
                        "view_pkls": {k: str(v) for k, v in view_pkls.items()},
                    })

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
        # pycocotools expects counts as bytes
        if isinstance(rle.get("counts", None), str):
            rle = dict(rle)
            rle["counts"] = rle["counts"].encode("utf-8")
        return coco_mask.decode(rle)

    def __len__(self):
        return len(self.metas)

    def _build_view_clip(self, meta, sample_frame_ids, view_name, idx):
        with open(meta["view_pkls"][view_name], "rb") as fp:
            view_data = pickle.load(fp)

        imgs, labels, boxes, masks, valid = [], [], [], [], []
        for frame_id in sample_frame_ids:
            decoded = self._decode_obj_mask(view_data, frame_id, meta["view_obj_ids"][view_name])
            if decoded is None:
                # fallback size if frame/object does not exist
                sample_frame = next(iter(view_data.values()))
                if len(sample_frame) > 0:
                    any_obj = next(iter(sample_frame.values()))
                    h, w = any_obj["rle"]["size"]
                else:
                    h, w = 1080, 1920
                decoded = np.zeros((h, w), dtype=np.uint8)

            mask = (decoded > 0).astype(np.float32)
            img = Image.fromarray((mask * 255).astype(np.uint8), mode="L").convert("RGB")

            label = torch.tensor(0)
            if (mask > 0).any():
                y1, y2, x1, x2 = self._mask_to_box(mask)
                box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                valid.append(1)
            else:
                box = torch.tensor([0, 0, 0, 0]).to(torch.float)
                valid.append(0)

            imgs.append(img)
            labels.append(label)
            masks.append(torch.from_numpy(mask))
            boxes.append(box)

        w, h = imgs[-1].size
        target = {
            "frames_idx": torch.tensor(sample_frame_ids, dtype=torch.long),
            "labels": torch.stack(labels, dim=0),
            "boxes": torch.stack(boxes, dim=0),
            "masks": torch.stack(masks, dim=0),
            "valid": torch.tensor(valid),
            "caption": meta["caption"],
            "orig_size": torch.as_tensor([int(h), int(w)]),
            "size": torch.as_tensor([int(h), int(w)]),
            "video_id": f"{meta['scene']}/{meta['clip']}/{view_name}",
            "exp_id": idx,
        }

        target["boxes"][:, 0::2].clamp_(min=0, max=w)
        target["boxes"][:, 1::2].clamp_(min=0, max=h)

        imgs, target = self._transforms(imgs, target)
        imgs = torch.stack(imgs, dim=0)
        return imgs, target

    def __getitem__(self, idx):
        instance_check = False
        while not instance_check:
            meta = self.metas[idx]
            frame_ids = meta["frame_ids"]
            center_pos = random.randint(0, len(frame_ids) - 1)
            sample_pos = FrameSampler.sample_global_frames(center_pos, len(frame_ids), self.num_frames)
            sample_frame_ids = [frame_ids[p] for p in sample_pos]

            view_streams = {}
            view_targets = {}
            for view_name in ["view1", "view2", "view3"]:
                imgs, tgt = self._build_view_clip(meta, sample_frame_ids, view_name, idx)
                view_streams[view_name] = imgs
                view_targets[view_name] = tgt

            # Keep synchronized stream order: [3, T, C, H, W]
            multi_view_imgs = torch.stack([view_streams["view1"], view_streams["view2"], view_streams["view3"]], dim=0)
            target = {
                "caption": meta["caption"],
                "video_id": f"{meta['scene']}/{meta['clip']}",
                "exp_id": idx,
                "frames_idx": torch.tensor(sample_frame_ids, dtype=torch.long),
                "view_targets": view_targets,
            }

            if any(torch.any(view_targets[v]["valid"] == 1) for v in ["view1", "view2", "view3"]):
                instance_check = True
            else:
                idx = random.randint(0, self.__len__() - 1)

        return multi_view_imgs, target


def build(image_set, args):
    root = Path(args.crtrack_path)
    assert root.exists(), f'provided CRTrack path {root} does not exist'

    # CRTrack_test currently only contains test-like split under CRTrack_In-domain.
    if image_set not in ["train", "test", "valid", "valid_u"]:
        raise ValueError(f"Unsupported image_set for CRTrack_test: {image_set}")

    dataset = CRTrackTestDataset(
        root=root / "CRTrack_In-domain",
        transforms=make_coco_transforms("train" if image_set == "train" else "valid_u", max_size=args.max_size, resize=args.augm_resize),
        num_frames=args.num_frames,
    )
    return dataset
