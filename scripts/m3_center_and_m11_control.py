#!/usr/bin/env python3
"""Reviewer round-4 computations.

Part A (M3): within-organ cross-center transfer benchmark.
  TCGA-CRC (and TCGA-STAD) patients are split into two disjoint
  tissue-source-site (TSS) groups; the unified configuration
  (LogisticRegression, balanced, C=1.0, max_iter=2000, K=24 state fractions,
  frozen fractions file) is trained on group A and evaluated on group B and
  vice versa, with a train-label-shuffle-and-refit permutation null (identical
  to the transfer-reconciliation protocol, perm seed 20260807).
  50 random TSS-level partitions per organ; each side must hold >=150 (CRC) /
  >=120 (STAD) patients and >=20 (CRC) / >=15 (STAD) MSI-H patients.

  INTERPRETATION RULES FIXED BEFORE EXECUTION (recorded in manifest):
    median within-organ cross-center AUROC >= 0.65 in both directions
      -> center effects do not collapse within-organ state-fraction transfer;
         the lineage-boundary framing stands and is strengthened.
    median in (0.58, 0.65) -> partial center sensitivity; soften the
      center-confound counter-argument accordingly.
    median <= 0.58 -> the cross-organ 0.53 is a marginal decrement over the
      within-organ cross-center baseline; the lineage-boundary framing is not
      supported and the manuscript must be reframed.

Part B (m11): specificity control for the R^2 reconstruction contrast.
  Train a mucinous-histology classifier (ICD-O-3 8480/3) and an MSI-H
  classifier on the SAME representation (patient-level mean 1024-d UNI
  features, unified logistic config, 5-fold out-of-fold decision scores,
  cv seed 1234), then RidgeCV-reconstruct each score from (a) the 24-d state
  fractions and (b) the 1024-d means (5-fold cross-validated R^2, protocol of
  e2c_mil_projection.py). If the 24-d representation reconstructs the mucinous
  score far better than the MSI score, the R^2 gap for MSI is not a generic
  compression artefact.
"""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

ROOT = Path("/home/ubuntu/shared/proj-Gastrointestinal")
SF_PATH = ROOT / "Experiment/analysis/medical_morphology_atlas/gi_morphology_state_patient_fractions.tsv"
MANIFEST_DIR = ROOT / "Experiment/datasets/manifests/tasks_qc"
ICDO = ROOT / "Experiment/datasets/labels/icdo3_histology_cbioportal.csv"
OUT_DIR = ROOT / "Publication/paper_visibility_boundary"
CACHE = OUT_DIR / "patient_mean1024_cache.npz"
STATE_COLS = [f"state_{i:02d}" for i in range(24)]
CV_SEED = 1234
PERM_SEED = 20260807
N_PARTITIONS = 50
NPERM_PART = 200
NPERM_CANON = 1000

rng = np.random.default_rng(PERM_SEED)


def lr():
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)


def fit_auroc(Xtr, ytr, Xte, yte):
    m = lr().fit(Xtr, ytr)
    return roc_auc_score(yte, m.decision_function(Xte))


# ---------------- Part A: within-organ cross-center benchmark ----------------
sf = pd.read_csv(SF_PATH, sep="\t")
results = {"meta": {"cv_seed": CV_SEED, "perm_seed": PERM_SEED,
                    "n_partitions": N_PARTITIONS, "nperm_per_partition": NPERM_PART,
                    "nperm_canonical": NPERM_CANON,
                    "interpretation_rules_fixed_before_run": {
                        "stands": "median >= 0.65 both directions",
                        "soften": "median in (0.58, 0.65)",
                        "reframe": "median <= 0.58"}},
           "M3_within_organ_cross_center": {}}

for coh_sf, coh, min_n, min_pos in [("CRC_MSI", "CRC", 150, 20),
                                    ("STAD_MSI", "STAD", 120, 15)]:
    sub = sf[sf.cohort == coh_sf].copy()
    sub["tss"] = sub.case_submitter_id.str.split("-").str[1]
    X = sub[STATE_COLS].values
    y = (sub.msi_status == "MSI-H").astype(int).values
    tss_arr = sub.tss.values
    tss_codes = sorted(set(tss_arr))
    part_results = []
    canonical = None
    tries = 0
    while len(part_results) < N_PARTITIONS and tries < 5000:
        tries += 1
        order = rng.permutation(tss_codes)
        grp = {}
        cnt = [0, 0]
        for t in order:
            side = 0 if cnt[0] <= cnt[1] else 1
            grp[t] = side
            cnt[side] += int((tss_arr == t).sum())
        mask = np.array([grp[t] for t in tss_arr])
        ok = True
        for side in (0, 1):
            m = mask == side
            if m.sum() < min_n or y[m].sum() < min_pos:
                ok = False
        if not ok:
            continue
        entry = {"partition": len(part_results),
                 "sides": [{"n": int((mask == s).sum()), "n_pos": int(y[mask == s].sum()),
                            "n_tss": int(sum(1 for t in grp if grp[t] == s))}
                           for s in (0, 1)],
                 "directions": {}}
        nperm = NPERM_CANON if canonical is None else NPERM_PART
        for name, tr, te in [("A_to_B", 0, 1), ("B_to_A", 1, 0)]:
            Xtr, ytr = X[mask == tr], y[mask == tr]
            Xte, yte = X[mask == te], y[mask == te]
            obs = fit_auroc(Xtr, ytr, Xte, yte)
            null = np.array([fit_auroc(Xtr, rng.permutation(ytr), Xte, yte)
                             for _ in range(nperm)])
            p = float((1 + (null >= obs).sum()) / (1 + nperm))
            entry["directions"][name] = {"auroc": round(float(obs), 4),
                                         "perm_p": round(p, 4), "nperm": nperm}
        if canonical is None:
            canonical = entry
            entry["canonical"] = True
        part_results.append(entry)
        if len(part_results) % 10 == 0:
            print(f"{coh}: {len(part_results)}/{N_PARTITIONS} partitions", flush=True)
    aurocs = np.array([e["directions"][d]["auroc"]
                       for e in part_results for d in ("A_to_B", "B_to_A")])
    sig = np.array([e["directions"][d]["perm_p"] < 0.05
                    for e in part_results for d in ("A_to_B", "B_to_A")])
    results["M3_within_organ_cross_center"][coh] = {
        "n_valid_partitions": len(part_results),
        "auroc_median": round(float(np.median(aurocs)), 4),
        "auroc_iqr": [round(float(np.percentile(aurocs, 25)), 4),
                      round(float(np.percentile(aurocs, 75)), 4)],
        "auroc_range": [round(float(aurocs.min()), 4), round(float(aurocs.max()), 4)],
        "frac_perm_p_lt_05": round(float(sig.mean()), 4),
        "canonical_partition": canonical,
        "partitions": part_results}
    print(coh, "median", np.median(aurocs), "IQR",
          np.percentile(aurocs, [25, 75]), "sig frac", sig.mean(), flush=True)

with open(OUT_DIR / "m3_center_benchmark_manifest.json", "w") as h:
    json.dump(results, h, indent=1)
print("Part A written", flush=True)

# ---------------- Part B: m11 reconstruction-specificity control ----------------
def read_manifest():
    by_file = {}
    for coh, fn in [("STAD", "stad_msi_qc_feature_complete_slide_manifest.tsv"),
                    ("CRC", "crc_msi_qc_feature_complete_slide_manifest.tsv")]:
        with (MANIFEST_DIR / fn).open(newline="") as h:
            for r in csv.DictReader(h, delimiter="\t"):
                if r.get("feature_path"):
                    r["_cohort"] = coh
                    by_file.setdefault(r["file_id"], r)
    return sorted(by_file.values(), key=lambda x: x["file_id"])


if CACHE.exists():
    z = np.load(CACHE, allow_pickle=True)
    pats_by_coh = {c: list(z[f"{c}_pats"]) for c in ("CRC", "STAD")}
    mean_by_coh = {c: z[f"{c}_mean"] for c in ("CRC", "STAD")}
else:
    slides = read_manifest()
    acc = {}
    for i, r in enumerate(slides, 1):
        u = torch.load(ROOT / r["feature_path"], map_location="cpu")
        x = F.normalize(u["features"].float(), dim=1).numpy().astype(np.float32)
        key = (r["_cohort"], r["case_submitter_id"])
        e = acc.setdefault(key, {"sum": np.zeros(x.shape[1]), "n": 0})
        e["sum"] += x.sum(0)
        e["n"] += len(x)
        if i % 200 == 0:
            print(f"features {i}/{len(slides)}", flush=True)
    pats_by_coh, mean_by_coh = {}, {}
    for coh in ("CRC", "STAD"):
        pats = sorted(p for (c, p) in acc if c == coh)
        pats_by_coh[coh] = pats
        mean_by_coh[coh] = np.vstack([acc[(coh, p)]["sum"] / acc[(coh, p)]["n"]
                                      for p in pats])
    np.savez_compressed(CACHE,
                        CRC_pats=np.array(pats_by_coh["CRC"]),
                        CRC_mean=mean_by_coh["CRC"],
                        STAD_pats=np.array(pats_by_coh["STAD"]),
                        STAD_mean=mean_by_coh["STAD"])
    print("feature cache written", flush=True)

icdo = pd.read_csv(ICDO).set_index("case_submitter_id")
sf_crc = sf[sf.cohort == "CRC_MSI"].set_index("case_submitter_id")
pats = [p for p in pats_by_coh["CRC"] if p in sf_crc.index and p in icdo.index]
M = mean_by_coh["CRC"][[pats_by_coh["CRC"].index(p) for p in pats]]
F24 = sf_crc.loc[pats, STATE_COLS].values
y_msi = (sf_crc.loc[pats, "msi_status"] == "MSI-H").astype(int).values
y_muc = (icdo.loc[pats, "icdo3_histology"] == "8480/3").astype(int).values
print("m11 cohort:", len(pats), "mucinous", y_muc.sum(), "MSI-H", y_msi.sum(), flush=True)

out_b = {"n_patients": len(pats), "n_mucinous": int(y_muc.sum()),
         "n_msih": int(y_msi.sum()), "targets": {}}
skf = StratifiedKFold(5, shuffle=True, random_state=CV_SEED)
kf = KFold(5, shuffle=True, random_state=CV_SEED)
for tname, y in [("mucinous_score", y_muc), ("msi_score", y_msi)]:
    score = np.zeros(len(y))
    for tr, te in skf.split(M, y):
        m = lr().fit(M[tr], y[tr])
        score[te] = m.decision_function(M[te])
    auroc = roc_auc_score(y, score)
    rec = {}
    for rname, X in [("K24_fractions", F24), ("mean1024", M)]:
        model = RidgeCV(alphas=np.logspace(-3, 3, 13))
        rec[rname] = round(float(np.mean(
            cross_val_score(model, X, score, cv=kf, scoring="r2"))), 4)
    out_b["targets"][tname] = {"oof_auroc": round(float(auroc), 4), "r2": rec}
    print(tname, "auroc", round(auroc, 3), rec, flush=True)

with open(OUT_DIR / "m11_reconstruction_control_manifest.json", "w") as h:
    json.dump(out_b, h, indent=1)
print("DONE", flush=True)
