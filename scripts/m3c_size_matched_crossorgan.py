#!/usr/bin/env python3
"""M3c: size-matched cross-organ transfer.

The within-organ cross-center benchmark trains on roughly half a cohort, whereas
the headline cross-organ transfer trains on a full cohort. That comparison
favours the cross-organ estimate. Here the cross-organ transfer is re-run with
the training cohort randomly subsampled to the median training size of the
corresponding cross-center partitions, 50 times, so the two are matched in
training-set size (and, separately, in the number of MSI-H training events).

Also runs the positive control (ICD-O-3 mucinous histology) through the identical
within-organ cross-center benchmark, to measure directly how far the ladder's
center-robustness extrapolates from coarse morphology to MSI.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path("/home/ubuntu/shared/proj-Gastrointestinal")
SF_PATH = ROOT / "Experiment/analysis/medical_morphology_atlas/gi_morphology_state_patient_fractions.tsv"
ICDO = ROOT / "Experiment/datasets/labels/icdo3_histology_cbioportal.csv"
OUT_DIR = ROOT / "Publication/paper_visibility_boundary"
BENCH = OUT_DIR / "m3_center_benchmark_manifest.json"
STATE_COLS = [f"state_{i:02d}" for i in range(24)]
PERM_SEED = 20260807
N_REP = 50

rng = np.random.default_rng(PERM_SEED + 2)


def lr():
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)


sf = pd.read_csv(SF_PATH, sep="\t")
icdo = pd.read_csv(ICDO).set_index("case_submitter_id")
bench = json.load(open(BENCH))["M3_within_organ_cross_center"]

D = {}
for coh_sf, coh in [("CRC_MSI", "CRC"), ("STAD_MSI", "STAD")]:
    sub = sf[sf.cohort == coh_sf].copy()
    sub["tss"] = sub.case_submitter_id.str.split("-").str[1]
    muc = sub.case_submitter_id.map(icdo.icdo3_histology).eq("8480/3")
    D[coh] = {"X": sub[STATE_COLS].values,
              "y": (sub.msi_status == "MSI-H").astype(int).values,
              "muc": muc.fillna(False).astype(int).values,
              "tss": sub.tss.values,
              "has_icdo": sub.case_submitter_id.isin(icdo.index).values}

out = {"meta": {"n_rep": N_REP, "seed": PERM_SEED + 2}}

# ---- A: size-matched cross-organ transfer ----
out["A_size_matched_cross_organ"] = {}
for src, tgt in [("STAD", "CRC"), ("CRC", "STAD")]:
    sides = bench[tgt]["canonical_partition"]["sides"]
    n_train = int(np.median([s["n"] for s in sides]))
    n_pos = int(np.median([s["n_pos"] for s in sides]))
    Xs, ys = D[src]["X"], D[src]["y"]
    pos, neg = np.where(ys == 1)[0], np.where(ys == 0)[0]
    n_pos = min(n_pos, len(pos))
    n_neg = min(n_train - n_pos, len(neg))
    a = []
    for _ in range(N_REP):
        idx = np.concatenate([rng.choice(pos, n_pos, replace=False),
                              rng.choice(neg, n_neg, replace=False)])
        m = lr().fit(Xs[idx], ys[idx])
        a.append(roc_auc_score(D[tgt]["y"], m.decision_function(D[tgt]["X"])))
    a = np.array(a)
    out["A_size_matched_cross_organ"][f"{src}_to_{tgt}"] = {
        "train_n": int(n_pos + n_neg), "train_pos": int(n_pos),
        "auroc_median": round(float(np.median(a)), 4),
        "auroc_iqr": [round(float(np.percentile(a, 25)), 4),
                      round(float(np.percentile(a, 75)), 4)],
        "within_organ_cross_center_median_same_eval_set": bench[tgt]["auroc_median"]}
    print(f"{src}->{tgt} size-matched (train n={n_pos+n_neg}, pos={n_pos}): "
          f"median {np.median(a):.4f} vs within-{tgt} cross-center "
          f"{bench[tgt]['auroc_median']}", flush=True)

# ---- B: positive control through the same cross-center benchmark ----
out["B_positive_control_cross_center"] = {}
for coh in ["CRC", "STAD"]:
    ok = D[coh]["has_icdo"]
    X, ymuc, tss = D[coh]["X"][ok], D[coh]["muc"][ok], D[coh]["tss"][ok]
    codes = sorted(set(tss))
    a, tries = [], 0
    min_n = 150 if coh == "CRC" else 120
    min_pos = 15 if coh == "CRC" else 5
    while len(a) < N_REP * 2 and tries < 5000:
        tries += 1
        order = rng.permutation(codes)
        grp, cnt = {}, [0, 0]
        for t in order:
            side = 0 if cnt[0] <= cnt[1] else 1
            grp[t] = side
            cnt[side] += int((tss == t).sum())
        mask = np.array([grp[t] for t in tss])
        if any((mask == s).sum() < min_n or ymuc[mask == s].sum() < min_pos for s in (0, 1)):
            continue
        for tr, te in [(0, 1), (1, 0)]:
            m = lr().fit(X[mask == tr], ymuc[mask == tr])
            a.append(roc_auc_score(ymuc[mask == te], m.decision_function(X[mask == te])))
    a = np.array(a)
    out["B_positive_control_cross_center"][coh] = {
        "n_evals": len(a),
        "auroc_median": round(float(np.median(a)), 4) if len(a) else None,
        "auroc_iqr": [round(float(np.percentile(a, 25)), 4),
                      round(float(np.percentile(a, 75)), 4)] if len(a) else None,
        "msi_cross_center_median": bench[coh]["auroc_median"]}
    print(f"{coh} mucinous cross-center: n={len(a)} median "
          f"{np.median(a) if len(a) else float('nan'):.4f} vs MSI "
          f"{bench[coh]['auroc_median']}", flush=True)

with open(OUT_DIR / "m3c_size_matched_manifest.json", "w") as h:
    json.dump(out, h, indent=1)
print("DONE", flush=True)
