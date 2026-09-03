#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def read_tsv(path: str | Path) -> list[dict]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def find_slide_path(wsi_root: Path, row: dict) -> Path:
    candidates = []
    if row.get("wsi_path"):
        wsi_path = Path(row["wsi_path"])
        if wsi_path.is_absolute():
            candidates.append(wsi_path)
        else:
            candidates.extend([wsi_path, wsi_root / wsi_path])
    if row.get("file_name"):
        candidates.append(wsi_root / row["file_name"])
    if row.get("file_id"):
        candidates.append(wsi_root / f"{row['file_id']}.svs")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = []
    if row.get("file_id"):
        matches = list(wsi_root.glob(f"*{row['file_id']}*"))
    if not matches and row.get("file_name"):
        matches = list(wsi_root.glob(row["file_name"]))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        "Could not locate WSI for "
        f"file_id={row.get('file_id', '')} "
        f"file_name={row.get('file_name', '')} "
        f"wsi_path={row.get('wsi_path', '')}"
    )


def tissue_fraction(rgb: np.ndarray) -> float:
    pixels = rgb.reshape(-1, 3).astype(np.float32) / 255.0
    max_channel = pixels.max(axis=1)
    min_channel = pixels.min(axis=1)
    saturation = np.divide(max_channel - min_channel, max_channel + 1e-6)
    not_white = pixels.mean(axis=1) < 0.88
    tissue = (saturation > 0.05) & not_white
    return float(tissue.mean())


def tissue_mask_from_thumbnail(slide, max_dim: int = 2048) -> tuple[np.ndarray, float, float]:
    width, height = slide.dimensions
    scale = min(max_dim / max(width, height), 1.0)
    thumb_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    thumbnail = slide.get_thumbnail(thumb_size).convert("RGB")
    rgb = np.asarray(thumbnail).astype(np.float32) / 255.0
    max_channel = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    saturation = np.divide(max_channel - min_channel, max_channel + 1e-6)
    not_white = rgb.mean(axis=2) < 0.88
    mask = (saturation > 0.05) & not_white
    x_scale = mask.shape[1] / width
    y_scale = mask.shape[0] / height
    return mask, x_scale, y_scale


def mask_tissue_fraction(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    height, width = mask.shape
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    return float(mask[y0:y1, x0:x1].mean())


def select_evenly(coords: list[tuple[int, int]], max_patches: int) -> list[tuple[int, int]]:
    if len(coords) <= max_patches:
        return coords
    step = len(coords) / max_patches
    return [coords[int(math.floor(index * step))] for index in range(max_patches)]


def candidate_coordinates_thumbnail(
    slide,
    level: int,
    patch_size: int,
    stride: int,
    max_patches: int,
    tissue_threshold: float,
) -> list[tuple[int, int]]:
    downsample = float(slide.level_downsamples[level])
    level_width, level_height = slide.level_dimensions[level]
    base_patch_size = int(round(patch_size * downsample))
    base_stride = int(round(stride * downsample))
    mask, x_scale, y_scale = tissue_mask_from_thumbnail(slide)
    coords: list[tuple[int, int]] = []
    for y_base in range(0, max(1, int(round(level_height * downsample)) - base_patch_size + 1), base_stride):
        for x_base in range(0, max(1, int(round(level_width * downsample)) - base_patch_size + 1), base_stride):
            x0 = int(round(x_base * x_scale))
            y0 = int(round(y_base * y_scale))
            x1 = int(round((x_base + base_patch_size) * x_scale))
            y1 = int(round((y_base + base_patch_size) * y_scale))
            if mask_tissue_fraction(mask, x0, y0, x1, y1) >= tissue_threshold:
                coords.append((x_base, y_base))
    return select_evenly(coords, max_patches)


def candidate_coordinates_patch_scan(
    slide,
    level: int,
    patch_size: int,
    stride: int,
    max_patches: int,
    tissue_threshold: float,
) -> list[tuple[int, int]]:
    downsample = float(slide.level_downsamples[level])
    level_width, level_height = slide.level_dimensions[level]
    coords: list[tuple[int, int]] = []
    for y_level in range(0, max(1, level_height - patch_size + 1), stride):
        for x_level in range(0, max(1, level_width - patch_size + 1), stride):
            x_base = int(round(x_level * downsample))
            y_base = int(round(y_level * downsample))
            patch = slide.read_region((x_base, y_base), level, (patch_size, patch_size)).convert("RGB")
            if tissue_fraction(np.asarray(patch)) >= tissue_threshold:
                coords.append((x_base, y_base))
    return select_evenly(coords, max_patches)


def candidate_coordinates(
    slide,
    level: int,
    patch_size: int,
    stride: int,
    max_patches: int,
    coord_mode: str,
    tissue_threshold: float,
) -> list[tuple[int, int]]:
    if coord_mode == "thumbnail":
        coords = candidate_coordinates_thumbnail(slide, level, patch_size, stride, max_patches, tissue_threshold)
        if coords:
            return coords
    if coord_mode in {"thumbnail", "patch-scan"}:
        return candidate_coordinates_patch_scan(slide, level, patch_size, stride, max_patches, tissue_threshold)
    raise ValueError(f"Unknown coord_mode: {coord_mode}")


def batch_iter(items: list, batch_size: int) -> Iterable[list]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def pil_to_normalized_tensor(images: list[Image.Image]):
    import torch

    arrays = []
    for image in images:
        resized = image.resize((224, 224), Image.BICUBIC)
        array = np.asarray(resized).astype(np.float32) / 255.0
        array = (array - IMAGENET_MEAN) / IMAGENET_STD
        arrays.append(array.transpose(2, 0, 1))
    return torch.from_numpy(np.stack(arrays, axis=0))


def build_encoder(name: str, device: str):
    import torch

    if name == "colorstats":
        return None, 6
    if name == "uni":
        import timm

        model = timm.create_model(
            "hf-hub:MahmoodLab/UNI",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        model.eval().to(device)
        return model, 1024
    if name == "conch":
        from conch.open_clip_custom import create_model_from_pretrained

        model, preprocess = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch")
        model.preprocess = preprocess
        model.eval().to(device)
        return model, 512
    raise ValueError(f"Unsupported encoder: {name}")


def encode_colorstats(images: list[Image.Image]):
    import torch

    features = []
    for image in images:
        pixels = np.asarray(image.resize((224, 224), Image.BICUBIC)).astype(np.float32) / 255.0
        means = pixels.reshape(-1, 3).mean(axis=0)
        stds = pixels.reshape(-1, 3).std(axis=0)
        features.append(np.concatenate([means, stds], axis=0))
    return torch.from_numpy(np.stack(features, axis=0)).float()


def extract_slide_features(
    slide_path: Path,
    output_path: Path,
    encoder_name: str,
    model,
    device: str,
    level: int,
    patch_size: int,
    stride: int,
    max_patches: int,
    batch_size: int,
    coord_mode: str,
    tissue_threshold: float,
) -> dict:
    import openslide
    import torch

    slide = openslide.OpenSlide(str(slide_path))
    if level >= slide.level_count:
        level = slide.level_count - 1
    coords = candidate_coordinates(slide, level, patch_size, stride, max_patches, coord_mode, tissue_threshold)
    if not coords:
        raise RuntimeError(f"No tissue patches found for {slide_path}")
    features = []
    downsample = float(slide.level_downsamples[level])
    with torch.no_grad():
        for coord_batch in batch_iter(coords, batch_size):
            images = [
                slide.read_region(coord, level, (patch_size, patch_size)).convert("RGB")
                for coord in coord_batch
            ]
            if encoder_name == "colorstats":
                batch_features = encode_colorstats(images)
            elif encoder_name == "conch":
                tensor = torch.stack([model.preprocess(image) for image in images], dim=0).to(device)
                batch_features = model.encode_image(tensor, proj_contrast=False, normalize=False).detach().cpu()
            else:
                tensor = pil_to_normalized_tensor(images).to(device)
                batch_features = model(tensor).detach().cpu()
            features.append(batch_features)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": torch.cat(features, dim=0).float(),
        "coords": torch.tensor(coords, dtype=torch.int64),
        "metadata": {
            "slide_path": str(slide_path),
            "encoder": encoder_name,
            "level": level,
            "level_downsample": downsample,
            "patch_size": patch_size,
            "stride": stride,
            "num_patches": len(coords),
            "coord_mode": coord_mode,
            "tissue_threshold": tissue_threshold,
        },
    }
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(output_path)
    slide.close()
    return payload["metadata"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--wsi-root", required=True)
    parser.add_argument("--features-root", default="Experiment/datasets/features")
    parser.add_argument("--encoder", choices=["uni", "conch", "colorstats"], default="uni")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-patches", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--coord-mode", choices=["thumbnail", "patch-scan"], default="thumbnail")
    parser.add_argument("--tissue-threshold", type=float, default=0.20)
    parser.add_argument("--max-slides", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import torch

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    rows = read_tsv(args.manifest)
    if args.max_slides:
        rows = rows[: args.max_slides]
    model, feature_dim = build_encoder(args.encoder, device)
    wsi_root = Path(args.wsi_root)
    output_root = Path(args.features_root) / args.encoder
    log_rows = []
    for index, row in enumerate(rows, start=1):
        output_path = output_root / f"{row['file_id']}.pt"
        if output_path.exists() and not args.overwrite:
            log_rows.append({"file_id": row["file_id"], "status": "skipped", "feature_path": str(output_path)})
            continue
        slide_path = find_slide_path(wsi_root, row)
        metadata = extract_slide_features(
            slide_path=slide_path,
            output_path=output_path,
            encoder_name=args.encoder,
            model=model,
            device=device,
            level=args.level,
            patch_size=args.patch_size,
            stride=args.stride,
            max_patches=args.max_patches,
            batch_size=args.batch_size,
            coord_mode=args.coord_mode,
            tissue_threshold=args.tissue_threshold,
        )
        log_rows.append(
            {
                "file_id": row["file_id"],
                "status": "done",
                "feature_path": str(output_path),
                "num_patches": metadata["num_patches"],
                "feature_dim": feature_dim,
            }
        )
        print(json.dumps({"index": index, "total": len(rows), **log_rows[-1]}), flush=True)
    log_path = Path("Experiment/core_code/logs") / f"extract_{args.encoder}_features_log.tsv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file_id", "status", "feature_path", "num_patches", "feature_dim"]
    with log_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(log_rows)
    print(json.dumps({"slides": len(rows), "encoder": args.encoder, "log_path": str(log_path)}, indent=2))


if __name__ == "__main__":
    main()
