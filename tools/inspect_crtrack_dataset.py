"""Utility to visualize one random sample from CRTrack dataset.

It saves one PNG with side-by-side comparisons for 3 views:
- original real image (RGB frame)
- image + mask overlay
- image + bbox
and appends the corresponding view text under each view row.
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from datasets.crtrack_test import CRTrackTestDataset
from datasets.transform_utils import make_coco_transforms


def _build_real_view_frame_and_ann(dataset, meta, view_name, frame_id):
    view_data = dataset._get_view_data(meta["view_pkls"][view_name])
    h, w = dataset._infer_hw(view_data)
    obj_id = meta["view_obj_ids"][view_name]

    decoded = dataset._decode_obj_mask(view_data, frame_id, obj_id)
    if decoded is None:
        mask = np.zeros((h, w), dtype=np.uint8)
    else:
        mask = (decoded > 0).astype(np.uint8)

    original = dataset._load_real_rgb_frame(meta, view_name, frame_id, h, w)
    if original is None:
        raise RuntimeError(
            f"Missing RGB frame: {meta['scene']}/{meta['clip']}/{view_name}, frame_id={frame_id}. "
            "This sample should have been filtered by dataset strict RGB check."
        )

    box = None
    if mask.any():
        y1, y2, x1, x2 = dataset._mask_to_box(mask)
        box = (int(x1), int(y1), int(x2), int(y2))

    return original, mask, box


def _overlay_mask(img, mask, color=(255, 0, 0), alpha=0.45):
    arr = np.asarray(img).astype(np.float32)
    m = mask.astype(bool)
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    arr[m] = arr[m] * (1 - alpha) + color_arr * alpha
    return Image.fromarray(arr.astype(np.uint8))


def _draw_bbox(img, box, color=(0, 255, 0), width=4):
    out = img.copy()
    if box is None:
        return out
    draw = ImageDraw.Draw(out)
    draw.rectangle(box, outline=color, width=width)
    return out


def _fit_text(text, max_chars=75):
    text = text.strip() if text else ""
    if not text:
        return "(empty view text)"
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def generate_crtrack_preview(
    crtrack_path="data/CRTrack_my1",
    num_frames=8,
    output_path="output/crtrack_preview.png",
    seed=None,
):
    if seed is not None:
        random.seed(seed)

    root = Path(crtrack_path)
    dataset_root = root / "CRTrack_In-domain" if (root / "CRTrack_In-domain").exists() else root
    dataset = CRTrackTestDataset(
        root=dataset_root,
        transforms=make_coco_transforms("valid_u", max_size=1024, resize=False),
        num_frames=num_frames,
        strict_rgb_check=True,
    )

    if len(dataset) == 0:
        raise RuntimeError("CRTrack dataset is empty after strict RGB filtering.")

    idx = random.randrange(len(dataset))
    meta = dataset.metas[idx]
    frame_id = random.choice(meta["frame_ids"])

    col_titles = ["Original (Real RGB)", "+Mask", "+BBox"]
    view_names = dataset.VIEW_NAMES

    rows = []
    for view_name in view_names:
        original, mask, box = _build_real_view_frame_and_ann(dataset, meta, view_name, frame_id)
        with_mask = _overlay_mask(original, mask)
        with_box = _draw_bbox(original, box)
        rows.append((view_name, original, with_mask, with_box))

    w, h = rows[0][1].size
    gap = 20
    left_pad = 18
    top_pad = 18
    row_title_h = 28
    text_h = 28

    canvas_w = left_pad * 2 + (w * 3) + gap * 2
    canvas_h = top_pad * 2 + row_title_h + len(rows) * (h + text_h + gap)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    y = top_pad
    for i, t in enumerate(col_titles):
        x = left_pad + i * (w + gap)
        draw.text((x + 8, y + 6), t, fill=(10, 10, 10), font=font)

    y += row_title_h
    for view_name, im0, im1, im2 in rows:
        for i, im in enumerate([im0, im1, im2]):
            x = left_pad + i * (w + gap)
            canvas.paste(im, (x, y))

        view_text = meta.get("view_texts", {}).get(view_name, "")
        txt = f"{view_name}: {_fit_text(view_text)}"
        draw.text((left_pad + 8, y + h + 6), txt, fill=(20, 20, 20), font=font)
        y += h + text_h + gap

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, format="PNG")
    return out


def main():
    parser = argparse.ArgumentParser("Inspect CRTrack dataset sample")
    parser.add_argument("--crtrack_path", default="data/CRTrack_my1", type=str)
    parser.add_argument("--num_frames", default=8, type=int)
    parser.add_argument("--output_path", default="output/crtrack_preview.png", type=str)
    parser.add_argument("--seed", default=None, type=int)
    args = parser.parse_args()

    out = generate_crtrack_preview(
        crtrack_path=args.crtrack_path,
        num_frames=args.num_frames,
        output_path=args.output_path,
        seed=args.seed,
    )
    print(f"saved preview to: {out}")


if __name__ == "__main__":
    main()
