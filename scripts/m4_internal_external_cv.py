#!/usr/bin/env python3
"""M4 substitute: leave-one-center-out (internal-external) cross-validation.

No external gastric cohort with MSI labels is publicly available (see the
documented cohort search in m4_cohort_search_manifest.json). The accepted
substitute when external data cannot be obtained is internal-external
cross-validation: each contributing tissue-source site (TSS) is held out in
turn, the model is fitted on the remaining sites, and the held-out predictions
are pooled. This re-tests within-lineage MSI visibility under the same kind of
center shift that an external cohort would impose.

Reported per lineage:
  - pooled leave-one-center-out AUROC (all held-out predictions pooled),
    with a patient bootstrap 95% CI and a label-shuffle permutation null;
  - the random 5-fold CV value on the identical features, for comparison
    (the difference is the optimism attributable to same-center folds);
  - per-center AUROC for sites with >=5 events and >=20 patients.
The mucinous positive control is run through the identical procedure so that
the loss under center shift can be calibrated against a label known to be
robust.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path("/home/ubuntu/shared/proj-Gastrointestinal")
SF_PATH = ROOT / "Experiment/analysis/medical_morphology_atlas/gi_morphology_state_patient_fractions.tsv"
ICDO = ROOT / "Experiment/datasets/labels/icdo3_histology_cbioportal.csv"
OUT = ROOT / "Publication/paper_visibility_boundary/m4_internal_external_cv_manifest.json"
STATE_COLS = [f"state_{i:02d}" for i in range(24)]
CV_SEED = 1234
PERM_SEED = 20260807
NBOOT = 2000
NPERM = 1000

rng = np.random.default_rng(PERM_SEED)


def lr():
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)


def loco_scores(X, y, tss):
    """Leave-one-center-out held-out decision scores."""
    out = np.full(len(y), np.nan)
    for t in sorted(set(tss)):
        te = tss == t
        tr = ~te
        if len(set(y[tr])) < 2:
            continue
        out[te] = lr().fit(X[tr], y[tr]).decision_function(X[te])
    return out


def boot_ci(y, sc, n=NBOOT):
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(set(y[b])) < 2:
            continue
        vals.append(roc_auc_score(y[b], sc[b]))
    return [round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


sf = pd.read_csv(SF_PATH, sep="\t")
icdo = pd.read_csv(ICDO).set_index("case_submitter_id")
res = {"meta": {"cv_seed": CV_SEED, "perm_seed": PERM_SEED, "nboot": NBOOT,
                "nperm": NPERM,
                "design": "leave-one-tissue-source-site-out, held-out scores pooled"}}

for coh_sf, coh in [("CRC_MSI", "CRC"), ("STAD_MSI", "STAD")]:
    sub = sf[sf.cohort == coh_sf].copy()
    sub["tss"] = sub.case_submitter_id.str.split("-").str[1]
    X = sub[STATE_COLS].values
    tss = sub.tss.values
    labels = {"MSI_H": (sub.msi_status == "MSI-H").astype(int).values}
    muc = sub.case_submitter_id.map(icdo.icdo3_histology).eq("8480/3")
    labels["mucinous_control"] = muc.fillna(False).astype(int).values

    res[coh] = {"n_patients": int(len(sub)), "n_centers": int(len(set(tss)))}
    for lname, y in labels.items():
        sc = loco_scores(X, y, tss)
        ok = ~np.isnan(sc)
        auroc = roc_auc_score(y[ok], sc[ok])
        # permutation null: shuffle labels, repeat the whole LOCO procedure
        null = []
        for _ in range(NPERM):
            yp = rng.permutation(y)
            s2 = loco_scores(X, yp, tss)
            o2 = ~np.isnan(s2)
            if len(set(yp[o2])) < 2:
                continue
            null.append(roc_auc_score(yp[o2], s2[o2]))
        null = np.array(null)
        p = float((1 + (null >= auroc).sum()) / (1 + len(null)))
        # random 5-fold CV comparator on identical features
        skf = StratifiedKFold(5, shuffle=True, random_state=CV_SEED)
        rand = cross_val_predict(lr(), X, y, cv=skf, method="decision_function")
        rand_auroc = roc_auc_score(y, rand)
        per_center = {}
        for t in sorted(set(tss)):
            m = (tss == t) & ok
            if m.sum() >= 20 and 5 <= y[m].sum() <= m.sum() - 5:
                per_center[t] = round(float(roc_auc_score(y[m], sc[m])), 4)
        res[coh][lname] = {
            "n_pos": int(y.sum()),
            "loco_pooled_auroc": round(float(auroc), 4),
            "loco_boot95": boot_ci(y[ok], sc[ok]),
            "loco_perm_p": round(p, 4),
            "loco_perm_null_mean": round(float(null.mean()), 4),
            "random_5fold_cv_auroc": round(float(rand_auroc), 4),
            "optimism_random_minus_loco": round(float(rand_auroc - auroc), 4),
            "per_center_auroc_where_evaluable": per_center}
        print(f"{coh} {lname}: LOCO {auroc:.4f} (perm p {p:.4f}) vs random-5fold "
              f"{rand_auroc:.4f}; optimism {rand_auroc-auroc:+.4f}", flush=True)

with open(OUT, "w") as h:
    json.dump(res, h, indent=1)
print("DONE", flush=True)
