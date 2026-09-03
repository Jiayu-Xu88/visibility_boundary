#!/usr/bin/env python3
"""E1: positive-control ladder for cross-lineage transfer (pre-registered in
identical pipeline at every level, only labels change).

Levels: R0 MUC-quartile (ceiling/sanity; label shares provenance with features),
R1a LYM-, R1b DEB-, R1c STR-quartile (probe-derived, image-level), R2 ICD-O-3
8480/3 mucinous adenocarcinoma (image-independent), R2m mucinous-or-signet
(8480/3 | 8490/3; standalone signet impossible: ~0 CRC cases), R3 MSI-H
(reference). Quartiles are computed within lineage. Outputs: manifest JSON and
Supplementary Table S9 TSV. CPU only.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

ROOT = Path("/home/ubuntu/shared/proj-Gastrointestinal")
MANIFEST_DIR = ROOT / "Experiment/datasets/manifests/tasks_qc"
SF_PATH = ROOT / "Experiment/analysis/medical_morphology_atlas/gi_morphology_state_patient_fractions.tsv"
PROBE = ROOT / "Publication/paper_visibility_boundary/nct_uni_tissue_probe.npz"
ICDO = ROOT / "Experiment/datasets/labels/icdo3_histology_cbioportal.csv"
OUT_DIR = ROOT / "Publication/paper_visibility_boundary"
STATE_COLS = [f"state_{i:02d}" for i in range(24)]
CV_SEED = 1234
PERM_SEED = 20260807
NPERM = 1000
NBOOT = 2000


def cv_auroc(X, y):
    if y.sum() < 5 or y.sum() > len(y) - 5:
        return None
    skf = StratifiedKFold(5, shuffle=True, random_state=CV_SEED)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    p = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]
    return round(float(roc_auc_score(y, p)), 4)


def transfer_full(Xtr, ytr, Xte, yte):
    if ytr.sum() < 5 or yte.sum() < 5:
        return None
    def fit_auc(yy):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(Xtr, yy)
        return clf.predict_proba(Xte)[:, 1]
    p = fit_auc(ytr)
    obs = roc_auc_score(yte, p)
    rng = np.random.default_rng(PERM_SEED)
    null = np.array([roc_auc_score(yte, fit_auc(rng.permutation(ytr)))
                     for _ in range(NPERM)])
    boots = []
    for _ in range(NBOOT):
        idx = rng.integers(0, len(yte), len(yte))
        if len(np.unique(yte[idx])) < 2:
            continue
        boots.append(roc_auc_score(yte[idx], p[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"auroc": round(float(obs), 4),
            "perm_p": round(float((null >= obs).mean()), 4),
            "boot95": [round(float(lo), 4), round(float(hi), 4)],
            "n_test": int(len(yte)), "n_test_pos": int(yte.sum())}


def main():
    probe = np.load(PROBE, allow_pickle=True)
    W = probe["coef"].astype(np.float32)
    b = probe["intercept"].astype(np.float32)
    classes = [str(c) for c in probe["classes"]]
    ci = {c: i for i, c in enumerate(classes)}

    by_file = {}
    for coh, fn in [("STAD", "stad_msi_qc_feature_complete_slide_manifest.tsv"),
                    ("CRC", "crc_msi_qc_feature_complete_slide_manifest.tsv")]:
        with (MANIFEST_DIR / fn).open(newline="") as h:
            for r in csv.DictReader(h, delimiter="\t"):
                if r.get("feature_path"):
                    r["_cohort"] = coh
                    by_file.setdefault(r["file_id"], r)

    acc = {}
    for i, r in enumerate(sorted(by_file.values(), key=lambda x: x["file_id"]), 1):
        u = torch.load(ROOT / r["feature_path"], map_location="cpu")
        x = F.normalize(u["features"].float(), dim=1).numpy().astype(np.float32)
        p = torch.softmax(torch.from_numpy(x @ W.T + b), dim=1).numpy()
        key = (r["_cohort"], r["case_submitter_id"])
        e = acc.setdefault(key, {"sum": np.zeros(len(classes)), "n": 0})
        e["sum"] += p.sum(0)
        e["n"] += len(p)
        if i % 200 == 0:
            print(f"probe means {i}", flush=True)
    probe_mean = {k: v["sum"] / v["n"] for k, v in acc.items()}

    sf = __import__("pandas").read_csv(SF_PATH, sep="\t")
    frames = {}
    for coh_sf, coh in [("CRC_MSI", "CRC"), ("STAD_MSI", "STAD")]:
        sub = sf[sf.cohort == coh_sf].set_index("case_submitter_id")
        pats = [p for p in sub.index if (coh, p) in probe_mean]
        frames[coh] = {"pats": pats,
                       "X": sub.loc[pats, STATE_COLS].values,
                       "msi": (sub.loc[pats, "msi_status"] == "MSI-H").astype(int).values}

    icdo = {}
    with ICDO.open(newline="") as h:
        for r in csv.DictReader(h):
            icdo[r["case_submitter_id"]] = r["icdo3_histology"]

    def quartile_label(coh, cls):
        vals = np.array([probe_mean[(coh, p)][ci[cls]] for p in frames[coh]["pats"]])
        thr = np.quantile(vals, 0.75)
        return (vals > thr).astype(int)

    def icdo_label(coh, codes):
        return np.array([1 if icdo.get(p, "") in codes else 0
                         for p in frames[coh]["pats"]])

    ladder = {
        "R0_MUC_q4": {c: quartile_label(c, "MUC") for c in ["CRC", "STAD"]},
        "R1a_LYM_q4": {c: quartile_label(c, "LYM") for c in ["CRC", "STAD"]},
        "R1b_DEB_q4": {c: quartile_label(c, "DEB") for c in ["CRC", "STAD"]},
        "R1c_STR_q4": {c: quartile_label(c, "STR") for c in ["CRC", "STAD"]},
        "R2_mucinous_icdo": {c: icdo_label(c, {"8480/3"}) for c in ["CRC", "STAD"]},
        "R2m_mucinous_or_signet": {c: icdo_label(c, {"8480/3", "8490/3"})
                                   for c in ["CRC", "STAD"]},
        "R3_MSI_H": {c: frames[c]["msi"] for c in ["CRC", "STAD"]},
    }

    results = {"meta": {
                        "quartiles": "computed within lineage",
                        "signet_note": "standalone R2-prime impossible: CRC 8490/3 count "
                                       f"= {int(icdo_label('CRC', {'8490/3'}).sum())}; merged class used",
                        "cv_seed": CV_SEED, "perm_seed": PERM_SEED,
                        "nperm": NPERM, "nboot": NBOOT},
               "levels": {}}

    rows = []
    for level, labs in ladder.items():
        e = {"n_pos": {c: int(labs[c].sum()) for c in labs},
             "within_cv": {c: cv_auroc(frames[c]["X"], labs[c]) for c in labs},
             "transfer": {}}
        for src, tgt in [("STAD", "CRC"), ("CRC", "STAD")]:
            t = transfer_full(frames[src]["X"], labs[src], frames[tgt]["X"], labs[tgt])
            e["transfer"][f"{src}_to_{tgt}"] = t
        results["levels"][level] = e
        for d, t in e["transfer"].items():
            rows.append([level, d,
                         e["within_cv"]["CRC"], e["within_cv"]["STAD"],
                         e["n_pos"]["CRC"], e["n_pos"]["STAD"],
                         t["auroc"] if t else "NA",
                         t["perm_p"] if t else "NA",
                         f"{t['boot95'][0]}-{t['boot95'][1]}" if t else "NA"])
        print(level, json.dumps(e["n_pos"]),
              json.dumps({k: (v["auroc"] if v else None) for k, v in e["transfer"].items()}),
              flush=True)

    (OUT_DIR / "e1_ladder_manifest.json").write_text(json.dumps(results, indent=2))
    with (OUT_DIR / "supplementary_table_S9_ladder.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["level", "direction", "within_cv_CRC", "within_cv_STAD",
                    "n_pos_CRC", "n_pos_STAD", "transfer_auroc", "perm_p", "boot95"])
        w.writerows(rows)
    print("wrote manifest + Table S9", flush=True)


if __name__ == "__main__":
    main()
