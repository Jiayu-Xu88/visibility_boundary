#!/usr/bin/env python3
"""Export the label-blind gastric state-0 exemplars used in main Figure 3.

Patients and patches are ranked only by assignment to, and cosine similarity
with, the frozen state-0 centroid. MSI labels are copied to the audit table
after selection but are never used to filter or rank candidates.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import openslide
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", REPO_ROOT)).resolve()
MANIFEST_DIR = PROJECT_ROOT / "Experiment/datasets/manifests/tasks_qc"
ATLAS_DIR = PROJECT_ROOT / "Experiment/analysis/medical_morphology_atlas"
OUT_DIR = ATLAS_DIR / "state_exemplars"
RESULT_TSV = REPO_ROOT / "results/figure3_state0_labelblind_exemplars.tsv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_SVS_BY_NAME = {
    path.name: path
    for path in (PROJECT_ROOT / "Experiment/datasets/wsi_cache").rglob("*.svs")
}

TARGETS = [
    {"name": "STAD_state00_labelblind",
     "manifest": "stad_msi_qc_feature_complete_slide_manifest.tsv",
     "state": 0, "cohort": "STAD"},
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


def label_blind_hits(target: dict, centers: np.ndarray, top_n: int = 8) -> list[dict]:
    rows = read_tsv(MANIFEST_DIR / target["manifest"])
    centers_t = torch.nn.functional.normalize(
        torch.tensor(centers, dtype=torch.float32), dim=1)
    state = int(target["state"])
    hits_by_patient: dict[str, list[dict]] = {}

    for row in rows:
        feature_path = PROJECT_ROOT / row["feature_path"]
        payload = load_feature_payload(feature_path)
        slide_path = resolve_slide_path(payload.get("metadata", {}).get("slide_path", ""))
        if not slide_path.exists():
            slide_path = LOCAL_SVS_BY_NAME.get(row.get("file_name", ""), slide_path)
        if not slide_path.exists():
            continue
        features = torch.nn.functional.normalize(payload["features"].float(), dim=1)
        similarities = features @ centers_t.T
        assignment = similarities.argmax(dim=1)
        eligible = torch.where(assignment == state)[0]
        if eligible.numel() == 0:
            continue
        local = similarities[eligible, state]
        order = torch.argsort(local, descending=True)[:2]
        coords = payload["coords"].cpu().numpy()
        metadata = payload.get("metadata", {})
        patient = row["case_submitter_id"]
        for j in order.tolist():
            idx = int(eligible[j])
            score = float(local[j])
            hit = {
                "target": target["name"],
                "cohort": target["cohort"],
                "state": state,
                "selection_used_label": False,
                "case_submitter_id": patient,
                "file_id": row["file_id"],
                "label_recorded_after_selection": row.get("task_label", ""),
                "msi_status_recorded_after_selection": row.get("msi_status", ""),
                "cosine_to_centroid": score,
                "patch_index": idx,
                "x": int(coords[idx][0]),
                "y": int(coords[idx][1]),
                "patch_size": int(metadata.get("patch_size", 512)),
                "slide_path": str(slide_path),
            }
            hits_by_patient.setdefault(patient, []).append(hit)

    ranked = {
        patient: sorted(hits, key=lambda row: row["cosine_to_centroid"], reverse=True)[:2]
        for patient, hits in hits_by_patient.items()
    }
    first = sorted((hits[0] for hits in ranked.values()),
                   key=lambda row: row["cosine_to_centroid"], reverse=True)
    second = sorted((hits[1] for hits in ranked.values() if len(hits) > 1),
                    key=lambda row: row["cosine_to_centroid"], reverse=True)
    return (first + second)[:top_n]


def export_patch(hit: dict, out_path: Path) -> bool:
    try:
        slide = openslide.OpenSlide(hit["slide_path"])
        patch = slide.read_region(
            (int(hit["x"]), int(hit["y"])), 0,
            (int(hit["patch_size"]), int(hit["patch_size"]))).convert("RGB")
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
        draw.text((4, 196), path.stem[-18:], fill=(0, 0, 0))
        thumbs.append(canvas)
    cols = 4
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 192, rows * 216 + 28), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), title, fill=(0, 0, 0))
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % cols) * 192, 28 + (i // cols) * 216))
    sheet.save(out_path)


def main() -> None:
    centers = np.load(ATLAS_DIR / "gi_morphology_state_centers.npy")
    all_hits = []
    summary = {}
    for target in TARGETS:
        target_dir = OUT_DIR / target["name"]
        target_dir.mkdir(parents=True, exist_ok=True)
        hits = label_blind_hits(target, centers)
        exported = []
        for rank, hit in enumerate(hits, 1):
            out_path = target_dir / (
                f"rank{rank:02d}_{hit['case_submitter_id']}_{hit['file_id']}"
                f"_x{hit['x']}_y{hit['y']}.png")
            if export_patch(hit, out_path):
                hit["image_path"] = str(out_path.relative_to(PROJECT_ROOT))
                exported.append(out_path)
            all_hits.append(hit)
        contact_sheet(exported, target["name"],
                      target_dir / f"{target['name']}_contact_sheet.png")
        summary[target["name"]] = {
            "state": target["state"],
            "selection": "one assigned state-0 patch per patient, ranked by cosine to frozen centroid; no label used",
            "n_selected_patients": len(hits),
            "n_exported_images": len(exported),
        }

    fieldnames = sorted({key for row in all_hits for key in row})
    with RESULT_TSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_hits)
    (OUT_DIR / "figure3_state0_labelblind_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("wrote", RESULT_TSV)


if __name__ == "__main__":
    main()
