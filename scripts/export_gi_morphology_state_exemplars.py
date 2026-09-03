#!/usr/bin/env python3
"""Export representative WSI patches for GI morphology-state pilot."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import openslide
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_DIR = PROJECT_ROOT / "Experiment/datasets/manifests/tasks_qc"
ATLAS_DIR = PROJECT_ROOT / "Experiment/analysis/medical_morphology_atlas"
OUT_DIR = ATLAS_DIR / "state_exemplars"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    {"name": "CRC_MSI_state00", "manifest": "crc_msi_qc_feature_complete_slide_manifest.tsv", "state": 0, "label": "MSI-H"},
    {"name": "STAD_EBV_state02", "manifest": "stad_subtype_qc_feature_complete_slide_manifest.tsv", "state": 2, "label": "EBV"},
    {"name": "STAD_EBV_state03", "manifest": "stad_subtype_qc_feature_complete_slide_manifest.tsv", "state": 3, "label": "EBV"},
    {"name": "STAD_MSI_enriched_state18", "manifest": "stad_msi_qc_feature_complete_slide_manifest.tsv", "state": 18, "label": "MSI-H"},
    {"name": "STAD_MSS_state05", "manifest": "stad_msi_qc_feature_complete_slide_manifest.tsv", "state": 5, "label": "MSS"},
    {"name": "STAD_MSS_state23", "manifest": "stad_msi_qc_feature_complete_slide_manifest.tsv", "state": 23, "label": "MSS"},
]


def read_tsv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_feature_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "features" not in payload or "coords" not in payload:
        raise ValueError(f"Feature file lacks features/coords: {path}")
    return payload


def resolve_slide_path(raw: str) -> Path:
    path = Path(raw)
    if path.exists():
        return path
    marker = "proj-Gastrointestinal/"
    if marker in raw:
        candidate = PROJECT_ROOT / raw.split(marker, 1)[1]
        if candidate.exists():
            return candidate
    candidate = PROJECT_ROOT / raw
    if candidate.exists():
        return candidate
    return path


def top_hits_for_target(target: dict, centers: np.ndarray, top_n: int = 12) -> list[dict]:
    rows = [row for row in read_tsv(MANIFEST_DIR / target["manifest"]) if row.get("task_label") == target["label"]]
    state = int(target["state"])
    center = torch.tensor(centers[state], dtype=torch.float32)
    center = torch.nn.functional.normalize(center, dim=0)
    hits = []
    for row in rows:
        payload = load_feature_payload(PROJECT_ROOT / row["feature_path"])
        slide_path = resolve_slide_path(payload.get("metadata", {}).get("slide_path", ""))
        if not slide_path.exists():
            continue
        features = torch.nn.functional.normalize(payload["features"].float(), dim=1)
        scores = features @ center
        k = min(3, scores.numel())
        vals, idxs = torch.topk(scores, k=k)
        coords = payload["coords"].cpu().numpy()
        metadata = payload.get("metadata", {})
        for score, idx in zip(vals.cpu().tolist(), idxs.cpu().tolist()):
            hits.append({
                "target": target["name"],
                "state": state,
                "cohort_manifest": target["manifest"],
                "case_submitter_id": row["case_submitter_id"],
                "file_id": row["file_id"],
                "label": row["task_label"],
                "molecular_subtype": row.get("molecular_subtype", ""),
                "msi_status": row.get("msi_status", ""),
                "score": float(score),
                "patch_index": int(idx),
                "x": int(coords[idx][0]),
                "y": int(coords[idx][1]),
                "patch_size": int(metadata.get("patch_size", 512)),
                "slide_path": str(slide_path),
            })
    return sorted(hits, key=lambda row: row["score"], reverse=True)[:top_n]


def export_patch(hit: dict, out_path: Path) -> bool:
    try:
        slide = openslide.OpenSlide(hit["slide_path"])
        patch = slide.read_region((int(hit["x"]), int(hit["y"])), 0, (int(hit["patch_size"]), int(hit["patch_size"]))).convert("RGB")
        patch.save(out_path)
        slide.close()
        return True
    except Exception as error:
        hit["export_error"] = repr(error)
        return False


def contact_sheet(image_paths: list[Path], title: str, out_path: Path) -> None:
    if not image_paths:
        return
    thumbs = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((192, 192))
        canvas = Image.new("RGB", (192, 216), "white")
        canvas.paste(img, ((192 - img.width) // 2, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 196), path.stem[-10:], fill=(0, 0, 0))
        thumbs.append(canvas)
    cols = 4
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 192, rows * 216 + 28), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), title, fill=(0, 0, 0))
    for i, thumb in enumerate(thumbs):
        x = (i % cols) * 192
        y = 28 + (i // cols) * 216
        sheet.paste(thumb, (x, y))
    sheet.save(out_path)


def main() -> None:
    centers = np.load(ATLAS_DIR / "gi_morphology_state_centers.npy")
    all_hits = []
    summary = {}
    for target in TARGETS:
        target_dir = OUT_DIR / target["name"]
        target_dir.mkdir(parents=True, exist_ok=True)
        hits = top_hits_for_target(target, centers)
        exported = []
        for rank, hit in enumerate(hits, 1):
            out_path = target_dir / f"rank{rank:02d}_{hit['case_submitter_id']}_{hit['file_id']}_x{hit['x']}_y{hit['y']}.png"
            if export_patch(hit, out_path):
                hit["image_path"] = str(out_path.relative_to(PROJECT_ROOT))
                exported.append(out_path)
            all_hits.append(hit)
        contact_sheet(exported, target["name"], target_dir / f"{target['name']}_contact_sheet.png")
        summary[target["name"]] = {
            "state": target["state"],
            "label": target["label"],
            "requested_hits": len(hits),
            "exported_images": len(exported),
            "contact_sheet": str((target_dir / f"{target['name']}_contact_sheet.png").relative_to(PROJECT_ROOT)) if exported else None,
        }

    with (OUT_DIR / "gi_morphology_state_exemplar_patches.tsv").open("w", newline="") as handle:
        fieldnames = sorted({key for row in all_hits for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_hits)
    (OUT_DIR / "gi_morphology_state_exemplar_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
