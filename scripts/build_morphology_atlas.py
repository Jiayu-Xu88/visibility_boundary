#!/usr/bin/env python3
"""GI molecular morphology state atlas pilot.

This pilot asks whether STAD and CRC share H&E-derived morphology states for
MSI, and whether EBV appears as a gastric-lineage-specific state. It uses only
existing UNI patch features and task manifests.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import mannwhitneyu
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_DIR = PROJECT_ROOT / "Experiment/datasets/manifests/tasks_qc"
OUT_DIR = PROJECT_ROOT / "Experiment/analysis/medical_morphology_atlas"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260519
N_STATES = 24
PATCHES_PER_SLIDE_FIT = 96
PATCHES_PER_SLIDE_ASSIGN = 512


def read_tsv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_features(path: Path) -> torch.Tensor:
    loaded = torch.load(path, map_location="cpu")
    x = loaded["features"] if isinstance(loaded, dict) and "features" in loaded else loaded
    if x.ndim != 2:
        raise ValueError(f"Expected [patch, dim] features at {path}, got {tuple(x.shape)}")
    return torch.nn.functional.normalize(x.float(), dim=1)


def stable_indices(key: str, n: int, k: int) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=k, replace=False))


def unique_rows(*row_sets: list[dict]) -> list[dict]:
    by_file = {}
    for rows in row_sets:
        for row in rows:
            feature_path = row.get("feature_path")
            if not feature_path:
                continue
            by_file.setdefault(row["file_id"], row)
    return [by_file[key] for key in sorted(by_file)]


def sample_fit_matrix(rows: list[dict]) -> np.ndarray:
    chunks = []
    for index, row in enumerate(rows, 1):
        if index % 100 == 0:
            print(f"fit sample {index}/{len(rows)}")
        x = load_features(PROJECT_ROOT / row["feature_path"])
        idx = stable_indices(row["file_id"] + ":fit", x.shape[0], PATCHES_PER_SLIDE_FIT)
        chunks.append(x[idx].numpy().astype(np.float32))
    return np.vstack(chunks)


def slide_state_fraction(row: dict, kmeans: MiniBatchKMeans) -> np.ndarray:
    x = load_features(PROJECT_ROOT / row["feature_path"])
    idx = stable_indices(row["file_id"] + ":assign", x.shape[0], PATCHES_PER_SLIDE_ASSIGN)
    labels = kmeans.predict(x[idx].numpy().astype(np.float32))
    return np.bincount(labels, minlength=N_STATES).astype(np.float64) / len(labels)


def make_slide_table(rows: list[dict], cohort: str, kmeans: MiniBatchKMeans) -> list[dict]:
    out = []
    for index, row in enumerate(rows, 1):
        if index % 100 == 0:
            print(f"assign {cohort} {index}/{len(rows)}")
        frac = slide_state_fraction(row, kmeans)
        entry = {
            "cohort": cohort,
            "case_submitter_id": row["case_submitter_id"],
            "file_id": row["file_id"],
            "label": row["task_label"],
            "msi_status": row.get("msi_status", ""),
            "molecular_subtype": row.get("molecular_subtype", ""),
            "cv_fold": row.get("cv_fold", ""),
        }
        for state in range(N_STATES):
            entry[f"state_{state:02d}"] = float(frac[state])
        out.append(entry)
    return out


def patient_table(slides: list[dict], cohort: str, label_field: str = "label") -> list[dict]:
    by_case: dict[str, dict] = {}
    for row in slides:
        if row["cohort"] != cohort:
            continue
        item = by_case.setdefault(
            row["case_submitter_id"],
            {
                "cohort": cohort,
                "case_submitter_id": row["case_submitter_id"],
                "label": row[label_field],
                "msi_status": row["msi_status"],
                "molecular_subtype": row["molecular_subtype"],
                "cv_fold": row["cv_fold"],
                "fractions": [],
            },
        )
        item["fractions"].append([float(row[f"state_{s:02d}"]) for s in range(N_STATES)])
    out = []
    for item in by_case.values():
        frac = np.mean(item.pop("fractions"), axis=0)
        for state in range(N_STATES):
            item[f"state_{state:02d}"] = float(frac[state])
        out.append(item)
    return sorted(out, key=lambda row: row["case_submitter_id"])


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    comp = x[:, None] - y[None, :]
    return float((np.sum(comp > 0) - np.sum(comp < 0)) / comp.size)


def bh_fdr(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    q = np.empty_like(p)
    prev = 1.0
    n = len(p)
    for rank, idx in enumerate(order[::-1], 1):
        true_rank = n - rank + 1
        prev = min(prev, p[idx] * n / true_rank)
        q[idx] = prev
    return q.tolist()


def state_association(rows: list[dict], positive_labels: set[str], label_name: str) -> list[dict]:
    stats = []
    for state in range(N_STATES):
        pos = np.asarray([float(row[f"state_{state:02d}"]) for row in rows if row["label"] in positive_labels])
        neg = np.asarray([float(row[f"state_{state:02d}"]) for row in rows if row["label"] not in positive_labels])
        if len(pos) == 0 or len(neg) == 0:
            pval = 1.0
        else:
            pval = float(mannwhitneyu(pos, neg, alternative="two-sided").pvalue)
        stats.append({
            "analysis": label_name,
            "state": state,
            "positive_n": int(len(pos)),
            "negative_n": int(len(neg)),
            "positive_mean": float(pos.mean()) if len(pos) else float("nan"),
            "negative_mean": float(neg.mean()) if len(neg) else float("nan"),
            "delta_positive_minus_negative": float(pos.mean() - neg.mean()) if len(pos) and len(neg) else float("nan"),
            "mannwhitney_p": pval,
            "cliffs_delta": cliffs_delta(pos, neg),
        })
    qvals = bh_fdr([row["mannwhitney_p"] for row in stats])
    for row, qval in zip(stats, qvals):
        row["bh_fdr"] = float(qval)
    return stats


def write_tsv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def transfer_metrics(train_rows: list[dict], test_rows: list[dict], positive: str) -> dict:
    x_train = np.asarray([[float(row[f"state_{s:02d}"]) for s in range(N_STATES)] for row in train_rows])
    y_train = np.asarray([1 if row["label"] == positive else 0 for row in train_rows])
    x_test = np.asarray([[float(row[f"state_{s:02d}"]) for s in range(N_STATES)] for row in test_rows])
    y_test = np.asarray([1 if row["label"] == positive else 0 for row in test_rows])
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=SEED),
    )
    clf.fit(x_train, y_train)
    prob = clf.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "train_n": int(len(y_train)),
        "test_n": int(len(y_test)),
        "train_positive": int(y_train.sum()),
        "test_positive": int(y_test.sum()),
        "auroc": float(roc_auc_score(y_test, prob)),
        "pr_auc": float(average_precision_score(y_test, prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
    }


def summarize_states(stad_msi_stats: list[dict], crc_msi_stats: list[dict], ebv_stats: list[dict]) -> dict:
    by_stad = {row["state"]: row for row in stad_msi_stats}
    by_crc = {row["state"]: row for row in crc_msi_stats}
    by_ebv = {row["state"]: row for row in ebv_stats}
    shared_msi, stad_specific, crc_specific, ebv_specific = [], [], [], []
    for state in range(N_STATES):
        s, c, e = by_stad[state], by_crc[state], by_ebv[state]
        same_positive = s["delta_positive_minus_negative"] > 0 and c["delta_positive_minus_negative"] > 0
        same_negative = s["delta_positive_minus_negative"] < 0 and c["delta_positive_minus_negative"] < 0
        if (same_positive or same_negative) and s["bh_fdr"] < 0.10 and c["bh_fdr"] < 0.10:
            shared_msi.append(state)
        if s["bh_fdr"] < 0.10 and c["bh_fdr"] >= 0.10:
            stad_specific.append(state)
        if c["bh_fdr"] < 0.10 and s["bh_fdr"] >= 0.10:
            crc_specific.append(state)
        if e["bh_fdr"] < 0.10 and abs(e["cliffs_delta"]) >= 0.20:
            ebv_specific.append(state)
    return {
        "shared_msi_states_fdr10": shared_msi,
        "stad_msi_specific_states_fdr10": stad_specific,
        "crc_msi_specific_states_fdr10": crc_specific,
        "ebv_enriched_states_fdr10_abs_cliff_ge_0p20": ebv_specific,
    }


def main() -> None:
    stad_msi_rows = read_tsv(MANIFEST_DIR / "stad_msi_qc_feature_complete_slide_manifest.tsv")
    crc_msi_rows = read_tsv(MANIFEST_DIR / "crc_msi_qc_feature_complete_slide_manifest.tsv")
    stad_subtype_rows = read_tsv(MANIFEST_DIR / "stad_subtype_qc_feature_complete_slide_manifest.tsv")
    atlas_rows = unique_rows(stad_msi_rows, crc_msi_rows)
    print(f"unique slides for atlas: {len(atlas_rows)}")

    fit_x = sample_fit_matrix(atlas_rows)
    kmeans = MiniBatchKMeans(
        n_clusters=N_STATES,
        random_state=SEED,
        batch_size=4096,
        n_init=10,
        max_iter=200,
        reassignment_ratio=0.01,
    )
    kmeans.fit(fit_x)
    np.save(OUT_DIR / "gi_morphology_state_centers.npy", kmeans.cluster_centers_.astype(np.float32))

    slide_rows = []
    slide_rows.extend(make_slide_table(stad_msi_rows, "STAD_MSI", kmeans))
    slide_rows.extend(make_slide_table(crc_msi_rows, "CRC_MSI", kmeans))
    slide_rows.extend(make_slide_table(stad_subtype_rows, "STAD_SUBTYPE", kmeans))
    write_tsv(slide_rows, OUT_DIR / "gi_morphology_state_slide_fractions.tsv")

    stad_msi_patients = patient_table(slide_rows, "STAD_MSI")
    crc_msi_patients = patient_table(slide_rows, "CRC_MSI")
    stad_subtype_patients = patient_table(slide_rows, "STAD_SUBTYPE")
    write_tsv(stad_msi_patients + crc_msi_patients + stad_subtype_patients, OUT_DIR / "gi_morphology_state_patient_fractions.tsv")

    stad_stats = state_association(stad_msi_patients, {"MSI-H"}, "STAD_MSI_H_vs_MSS")
    crc_stats = state_association(crc_msi_patients, {"MSI-H"}, "CRC_MSI_H_vs_MSS")
    ebv_stats = state_association(stad_subtype_patients, {"EBV"}, "STAD_EBV_vs_non_EBV")
    all_stats = stad_stats + crc_stats + ebv_stats
    write_tsv(all_stats, OUT_DIR / "gi_morphology_state_associations.tsv")

    transfer = {
        "stad_to_crc_msi": transfer_metrics(stad_msi_patients, crc_msi_patients, "MSI-H"),
        "crc_to_stad_msi": transfer_metrics(crc_msi_patients, stad_msi_patients, "MSI-H"),
    }
    state_summary = summarize_states(stad_stats, crc_stats, ebv_stats)
    summary = {
        "n_states": N_STATES,
        "patches_per_slide_fit": PATCHES_PER_SLIDE_FIT,
        "patches_per_slide_assign": PATCHES_PER_SLIDE_ASSIGN,
        "slides": {
            "atlas_unique_stad_crc": len(atlas_rows),
            "stad_msi": len(stad_msi_rows),
            "crc_msi": len(crc_msi_rows),
            "stad_subtype": len(stad_subtype_rows),
        },
        "patients": {
            "stad_msi": len(stad_msi_patients),
            "crc_msi": len(crc_msi_patients),
            "stad_subtype": len(stad_subtype_patients),
        },
        "state_summary": state_summary,
        "transfer": transfer,
    }
    (OUT_DIR / "gi_molecular_morphology_atlas_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def top_rows(stats: list[dict], n: int = 6) -> list[dict]:
        return sorted(stats, key=lambda row: (row["bh_fdr"], -abs(row["cliffs_delta"])))[:n]

    report = OUT_DIR / "GI_MOLECULAR_MORPHOLOGY_ATLAS_PILOT.md"
    report.write_text(
        "# GI Molecular Morphology Atlas Pilot\n\n"
        "Unsupervised morphology states were learned from pooled STAD and CRC UNI patch features, then tested for MSI and EBV enrichment at patient level.\n\n"
        f"- States: {N_STATES}\n"
        f"- STAD MSI patients: {len(stad_msi_patients)}; CRC MSI patients: {len(crc_msi_patients)}; STAD subtype patients: {len(stad_subtype_patients)}\n"
        f"- Shared MSI states at FDR<0.10 in both STAD and CRC: {state_summary['shared_msi_states_fdr10']}\n"
        f"- STAD-specific MSI states at FDR<0.10: {state_summary['stad_msi_specific_states_fdr10']}\n"
        f"- CRC-specific MSI states at FDR<0.10: {state_summary['crc_msi_specific_states_fdr10']}\n"
        f"- EBV-enriched/depleted states at FDR<0.10 and |Cliff|>=0.20: {state_summary['ebv_enriched_states_fdr10_abs_cliff_ge_0p20']}\n\n"
        "## Cross-Cancer MSI Transfer Using State Fractions\n\n"
        "| direction | AUROC | PR-AUC | BalAcc | macro-F1 |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| STAD -> CRC | {transfer['stad_to_crc_msi']['auroc']:.4f} | {transfer['stad_to_crc_msi']['pr_auc']:.4f} | {transfer['stad_to_crc_msi']['balanced_accuracy']:.4f} | {transfer['stad_to_crc_msi']['macro_f1']:.4f} |\n"
        f"| CRC -> STAD | {transfer['crc_to_stad_msi']['auroc']:.4f} | {transfer['crc_to_stad_msi']['pr_auc']:.4f} | {transfer['crc_to_stad_msi']['balanced_accuracy']:.4f} | {transfer['crc_to_stad_msi']['macro_f1']:.4f} |\n\n"
        "## Top State Associations\n\n"
        "### STAD MSI-H vs MSS\n\n"
        "| state | delta | Cliff | FDR |\n|---:|---:|---:|---:|\n"
        + "\n".join(f"| {r['state']} | {r['delta_positive_minus_negative']:+.4f} | {r['cliffs_delta']:+.3f} | {r['bh_fdr']:.3g} |" for r in top_rows(stad_stats))
        + "\n\n### CRC MSI-H vs MSS\n\n"
        "| state | delta | Cliff | FDR |\n|---:|---:|---:|---:|\n"
        + "\n".join(f"| {r['state']} | {r['delta_positive_minus_negative']:+.4f} | {r['cliffs_delta']:+.3f} | {r['bh_fdr']:.3g} |" for r in top_rows(crc_stats))
        + "\n\n### STAD EBV vs non-EBV\n\n"
        "| state | delta | Cliff | FDR |\n|---:|---:|---:|---:|\n"
        + "\n".join(f"| {r['state']} | {r['delta_positive_minus_negative']:+.4f} | {r['cliffs_delta']:+.3f} | {r['bh_fdr']:.3g} |" for r in top_rows(ebv_stats))
        + "\n\n## Interpretation\n\n"
        "This is a morphology-state pilot, not yet a pathologist-validated atlas. A strong result would show states enriched in both STAD and CRC MSI-H, plus EBV-associated gastric-specific states. Weak transfer or no shared states would argue that MSI morphology is lineage-specific in these cohorts.\n",
    )
    print(json.dumps(summary, indent=2))
    print(report)


if __name__ == "__main__":
    main()
