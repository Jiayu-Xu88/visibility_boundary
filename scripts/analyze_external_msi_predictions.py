#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


LABEL_TO_INT = {"MSS": 0, "MSI-H": 1}


def read_table(path: Path) -> list[dict]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def metric_block(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(int)
    positives = labels == 1
    negatives = labels == 0
    return {
        "n": int(len(labels)),
        "n_positive": int(positives.sum()),
        "n_negative": int(negatives.sum()),
        "auroc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "pr_auc": float(average_precision_score(labels, scores)) if positives.any() else None,
        "brier": float(brier_score_loss(labels, scores)),
        "threshold": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)) if len(np.unique(labels)) == 2 else None,
        "sensitivity": float(((predictions == 1) & positives).sum() / positives.sum()) if positives.sum() else None,
        "specificity": float(((predictions == 0) & negatives).sum() / negatives.sum()) if negatives.sum() else None,
    }


def bootstrap_ci(labels: np.ndarray, scores: np.ndarray, threshold: float, n_bootstrap: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    n = len(labels)
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, n)
        sampled_labels = labels[indices]
        sampled_scores = scores[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        metrics = metric_block(sampled_labels, sampled_scores, threshold)
        for key in ["auroc", "pr_auc", "brier", "balanced_accuracy", "sensitivity", "specificity"]:
            if metrics[key] is not None:
                values[key].append(float(metrics[key]))
    return {
        key: {
            "mean": float(np.mean(items)),
            "ci95_low": float(np.quantile(items, 0.025)),
            "ci95_high": float(np.quantile(items, 0.975)),
            "n_bootstrap_valid": len(items),
        }
        for key, items in values.items()
        if items
    }


def plot_curves(labels: np.ndarray, scores: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(4.2, 4.0))
    plt.plot(fpr, tpr, color="#0072B2", linewidth=2)
    plt.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("External STAD MSI ROC")
    plt.tight_layout()
    plt.savefig(output_dir / "roc.png", dpi=300)
    plt.savefig(output_dir / "roc.pdf")
    plt.close()

    precision, recall, _ = precision_recall_curve(labels, scores)
    plt.figure(figsize=(4.2, 4.0))
    plt.plot(recall, precision, color="#D55E00", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("External STAD MSI PR")
    plt.tight_layout()
    plt.savefig(output_dir / "pr.png", dpi=300)
    plt.savefig(output_dir / "pr.pdf")
    plt.close()

    frac_pos, mean_pred = calibration_curve(labels, scores, n_bins=min(10, len(labels)), strategy="quantile")
    plt.figure(figsize=(4.2, 4.0))
    plt.plot(mean_pred, frac_pos, marker="o", color="#009E73", linewidth=2)
    plt.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    plt.xlabel("Mean predicted MSI-H probability")
    plt.ylabel("Observed MSI-H fraction")
    plt.title("External STAD MSI Calibration")
    plt.tight_layout()
    plt.savefig(output_dir / "calibration.png", dpi=300)
    plt.savefig(output_dir / "calibration.pdf")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze locked external STAD MSI prediction outputs.")
    parser.add_argument("--patient-predictions", required=True)
    parser.add_argument("--slide-predictions", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260504)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    patient_rows = read_table(Path(args.patient_predictions))
    labeled_rows = [row for row in patient_rows if row.get("label") in LABEL_TO_INT]
    if not labeled_rows:
        raise ValueError("No labeled patients found. Expected label values MSS or MSI-H.")

    labels = np.asarray([LABEL_TO_INT[row["label"]] for row in labeled_rows], dtype=int)
    scores = np.asarray([float(row["probability"]) for row in labeled_rows], dtype=float)
    overall = metric_block(labels, scores, args.threshold)
    ci = bootstrap_ci(labels, scores, args.threshold, args.bootstrap, args.seed)
    plot_curves(labels, scores, output_dir)

    slide_rows = read_table(Path(args.slide_predictions)) if args.slide_predictions else []
    patient_to_meta = {}
    for row in slide_rows:
        patient_id = row.get("external_patient_id") or row.get("case_submitter_id") or row.get("patient_id")
        if patient_id and patient_id not in patient_to_meta:
            patient_to_meta[patient_id] = row

    subgroup_rows = []
    for column in ["institution", "scanner", "msi_method", "sample_type", "is_ffpe"]:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(labeled_rows):
            meta = patient_to_meta.get(row["patient_id"], {})
            value = meta.get(column, "") or "UNKNOWN"
            groups[value].append(index)
        for value, indices in groups.items():
            subgroup_labels = labels[indices]
            subgroup_scores = scores[indices]
            metrics = metric_block(subgroup_labels, subgroup_scores, args.threshold)
            metrics.update({"subgroup_column": column, "subgroup_value": value})
            subgroup_rows.append(metrics)
    write_table(output_dir / "subgroup_metrics.tsv", subgroup_rows)

    summary = {
        "patient_predictions": args.patient_predictions,
        "slide_predictions": args.slide_predictions,
        "threshold": args.threshold,
        "overall": overall,
        "bootstrap_ci": ci,
        "n_labeled_patients": len(labeled_rows),
        "outputs": {
            "roc_png": str(output_dir / "roc.png"),
            "pr_png": str(output_dir / "pr.png"),
            "calibration_png": str(output_dir / "calibration.png"),
            "subgroup_metrics": str(output_dir / "subgroup_metrics.tsv"),
        },
    }
    (output_dir / "external_validation_analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    report = [
        "# STAD MSI Locked External Validation Analysis",
        "",
        "## Overall Patient-Level Metrics",
        "",
        f"- N patients: {overall['n']} (MSI-H={overall['n_positive']}, MSS={overall['n_negative']})",
        f"- AUROC: {overall['auroc']}",
        f"- PR-AUC: {overall['pr_auc']}",
        f"- Brier: {overall['brier']}",
        f"- Threshold: {overall['threshold']}",
        f"- Balanced accuracy: {overall['balanced_accuracy']}",
        f"- Sensitivity: {overall['sensitivity']}",
        f"- Specificity: {overall['specificity']}",
        "",
        "## Files",
        "",
        "- `roc.png` / `roc.pdf`",
        "- `pr.png` / `pr.pdf`",
        "- `calibration.png` / `calibration.pdf`",
        "- `subgroup_metrics.tsv`",
        "- `external_validation_analysis_summary.json`",
        "",
        "## Reporting Note",
        "",
        "These metrics are valid only if the manifest, labels, model, threshold, and exclusion rules were locked before inference outputs were reviewed.",
    ]
    (output_dir / "EXTERNAL_VALIDATION_ANALYSIS_REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
