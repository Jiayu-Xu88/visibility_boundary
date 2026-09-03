#!/usr/bin/env python3
"""Difference-in-differences for the label-contrast claim (DEV-001).

The manuscript states that on gastric patients MSI-H loses ~0.14 AUROC on
crossing the organ boundary while the mucinous positive control loses nothing.
Until now that was a comparison of two point estimates. This estimates the
difference of differences with an interval and a permutation null.

For each draw b:
  a random tissue-source-site partition of the evaluation organ is generated and
  BOTH labels are transferred across that same centre boundary (paired);
  the source organ is subsampled to the matched training size and BOTH labels
  are transferred across the organ boundary.
  DiD_b = (MSI_xorg - MSI_xcen) - (control_xorg - control_xcen)

A negative DiD means MSI-H pays an organ penalty the control does not.
The null shuffles MSI-H labels only, leaving the control untouched.
Unified configuration throughout: LogisticRegression(balanced, C=1.0,
max_iter=2000) on frozen UNI K=24 patient-level state fractions.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path("/home/ubuntu/shared/proj-Gastrointestinal")
P = ROOT / "Publication/paper_visibility_boundary"
SF = ROOT / "Experiment/analysis/medical_morphology_atlas/gi_morphology_state_patient_fractions.tsv"
ICDO = ROOT / "Experiment/datasets/labels/icdo3_histology_cbioportal.csv"
STATE = [f"state_{i:02d}" for i in range(24)]
SEED = 20260807
NBOOT, NPERM = 400, 500

rng = np.random.default_rng(SEED)


def lr():
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)


def auroc(Xtr, ytr, Xte, yte):
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return np.nan
    return roc_auc_score(yte, lr().fit(Xtr, ytr).decision_function(Xte))


sf = pd.read_csv(SF, sep="\t")
icdo = pd.read_csv(ICDO).set_index("case_submitter_id")
D = {}
for coh_sf, coh in [("CRC_MSI", "CRC"), ("STAD_MSI", "STAD")]:
    s = sf[sf.cohort == coh_sf].copy()
    s["tss"] = s.case_submitter_id.str.split("-").str[1]
    D[coh] = {
        "X": s[STATE].values,
        "msi": (s.msi_status == "MSI-H").astype(int).values,
        "muc": s.case_submitter_id.map(icdo.icdo3_histology).eq("8480/3").fillna(False).astype(int).values,
        "tss": s.tss.values,
    }
    print(coh, "n", len(s), "MSI-H", D[coh]["msi"].sum(), "mucinous", D[coh]["muc"].sum(), flush=True)

MINPOS = {"msi": {"CRC": 20, "STAD": 15}, "muc": {"CRC": 15, "STAD": 5}}
MINN = {"CRC": 150, "STAD": 120}


def partition(coh, r):
    """Random split of the organ's TSS codes into two size-balanced groups."""
    t = D[coh]["tss"]
    for _ in range(400):
        grp, cnt = {}, [0, 0]
        for code in r.permutation(sorted(set(t))):
            side = 0 if cnt[0] <= cnt[1] else 1
            grp[code] = side
            cnt[side] += int((t == code).sum())
        m = np.array([grp[c] for c in t])
        ok = all((m == s).sum() >= MINN[coh] and
                 D[coh]["msi"][m == s].sum() >= MINPOS["msi"][coh] and
                 D[coh]["muc"][m == s].sum() >= MINPOS["muc"][coh] for s in (0, 1))
        if ok:
            return m
    return None


def cross_centre(coh, mask, key, y=None):
    y = D[coh][key] if y is None else y
    X = D[coh]["X"]
    vals = [auroc(X[mask == a], y[mask == a], X[mask == b], y[mask == b])
            for a, b in ((0, 1), (1, 0))]
    return float(np.nanmean(vals))


def cross_organ(src, tgt, key, r, n_train, y_src=None):
    ys = D[src][key] if y_src is None else y_src
    Xs = D[src]["X"]
    pos, neg = np.where(ys == 1)[0], np.where(ys == 0)[0]
    npos = min(max(1, int(round(n_train * ys.mean()))), len(pos))
    nneg = min(n_train - npos, len(neg))
    idx = np.concatenate([r.choice(pos, npos, replace=False),
                          r.choice(neg, nneg, replace=False)])
    return auroc(Xs[idx], ys[idx], D[tgt]["X"], D[tgt][key])


def did_draw(tgt, src, r, y_src_msi=None, y_tgt_msi=None):
    m = partition(tgt, r)
    if m is None:
        return None
    n_train = int(np.median([(m == s).sum() for s in (0, 1)]))
    msi_t = D[tgt]["msi"] if y_tgt_msi is None else y_tgt_msi
    xc_msi = cross_centre(tgt, m, "msi", y=msi_t)
    xc_muc = cross_centre(tgt, m, "muc")
    xo_msi = cross_organ(src, tgt, "msi", r, n_train, y_src=y_src_msi)
    xo_muc = cross_organ(src, tgt, "muc", r, n_train)
    if any(np.isnan(v) for v in (xc_msi, xc_muc, xo_msi, xo_muc)):
        return None
    return dict(xc_msi=xc_msi, xc_muc=xc_muc, xo_msi=xo_msi, xo_muc=xo_muc,
                d_msi=xo_msi - xc_msi, d_muc=xo_muc - xc_muc,
                did=(xo_msi - xc_msi) - (xo_muc - xc_muc))


out = {"meta": {"seed": SEED, "n_bootstrap": NBOOT, "n_permutation": NPERM,
                "config": "LogisticRegression(balanced, C=1.0, max_iter=2000) on frozen UNI K=24 patient state fractions",
                "estimand": "(cross-organ minus cross-centre) for MSI-H minus the same for the mucinous control, both labels on the same partitions",
                "interpretation": "DiD < 0 means MSI-H pays an organ penalty the positive control does not"}}

for tgt, src in [("STAD", "CRC"), ("CRC", "STAD")]:
    draws = []
    r = np.random.default_rng(SEED + hash(tgt) % 1000)
    while len(draws) < NBOOT:
        d = did_draw(tgt, src, r)
        if d:
            draws.append(d)
        if len(draws) % 100 == 0 and draws and len(draws) != len(set(range(len(draws)))):
            pass
    arr = {k: np.array([d[k] for d in draws]) for k in draws[0]}
    obs = {k: float(np.median(v)) for k, v in arr.items()}
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in arr.items()}

    nulls = []
    rp = np.random.default_rng(SEED + 7)
    while len(nulls) < NPERM:
        ys = rp.permutation(D[src]["msi"])
        yt = rp.permutation(D[tgt]["msi"])
        d = did_draw(tgt, src, rp, y_src_msi=ys, y_tgt_msi=yt)
        if d:
            nulls.append(d["did"])
    nulls = np.array(nulls)
    p_two = float((1 + (np.abs(nulls) >= abs(obs["did"])).sum()) / (1 + len(nulls)))

    out[f"evaluated_on_{tgt}"] = {
        "source_organ": src, "n_bootstrap_draws": len(draws),
        "cross_centre_MSI_median": obs["xc_msi"], "cross_centre_MSI_ci95": ci["xc_msi"],
        "cross_centre_control_median": obs["xc_muc"], "cross_centre_control_ci95": ci["xc_muc"],
        "cross_organ_MSI_median": obs["xo_msi"], "cross_organ_MSI_ci95": ci["xo_msi"],
        "cross_organ_control_median": obs["xo_muc"], "cross_organ_control_ci95": ci["xo_muc"],
        "delta_MSI": obs["d_msi"], "delta_MSI_ci95": ci["d_msi"],
        "delta_control": obs["d_muc"], "delta_control_ci95": ci["d_muc"],
        "difference_in_differences": obs["did"], "did_ci95": ci["did"],
        "did_excludes_zero": bool(ci["did"][1] < 0 or ci["did"][0] > 0),
        "permutation_p_two_sided": p_two,
        "permutation_null_median": float(np.median(nulls)),
    }
    print(f"{tgt}: dMSI {obs['d_msi']:+.3f} {ci['d_msi']}  dCTRL {obs['d_muc']:+.3f} {ci['d_muc']}"
          f"  DiD {obs['did']:+.3f} {ci['did']}  p={p_two:.4f}", flush=True)

fp = P / "did_label_contrast_manifest.json"
fp.write_text(json.dumps(out, indent=1))
print("written", fp)
