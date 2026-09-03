#!/usr/bin/env python3
"""M3 comparator: random patient-level splits with the same group sizes as the
TSS partitions, isolating the center effect from the training-set-size effect.
For each organ: 50 random half-splits (matched to the TSS partitions' typical
size), unified config, both directions. If random-split AUROC >> TSS-split
AUROC, the decrement is center structure, not sample size."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path("/home/ubuntu/shared/proj-Gastrointestinal")
SF_PATH = ROOT / "Experiment/analysis/medical_morphology_atlas/gi_morphology_state_patient_fractions.tsv"
OUT_DIR = ROOT / "Publication/paper_visibility_boundary"
STATE_COLS = [f"state_{i:02d}" for i in range(24)]
PERM_SEED = 20260807
N_SPLITS = 50

rng = np.random.default_rng(PERM_SEED + 1)


def lr():
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)


sf = pd.read_csv(SF_PATH, sep="\t")
out = {"meta": {"n_splits": N_SPLITS, "seed": PERM_SEED + 1,
                "purpose": "random half-splits matched in size to TSS partitions"}}
for coh_sf, coh in [("CRC_MSI", "CRC"), ("STAD_MSI", "STAD")]:
    sub = sf[sf.cohort == coh_sf]
    X = sub[STATE_COLS].values
    y = (sub.msi_status == "MSI-H").astype(int).values
    n = len(y)
    aurocs = []
    for _ in range(N_SPLITS):
        idx = rng.permutation(n)
        half = n // 2
        for a, b in [(idx[:half], idx[half:]), (idx[half:], idx[:half])]:
            if y[a].sum() < 5 or y[b].sum() < 5:
                continue
            m = lr().fit(X[a], y[a])
            aurocs.append(roc_auc_score(y[b], m.decision_function(X[b])))
    aurocs = np.array(aurocs)
    out[coh] = {"auroc_median": round(float(np.median(aurocs)), 4),
                "auroc_iqr": [round(float(np.percentile(aurocs, 25)), 4),
                              round(float(np.percentile(aurocs, 75)), 4)],
                "n_evals": len(aurocs)}
    print(coh, out[coh], flush=True)

with open(OUT_DIR / "m3b_random_split_comparator.json", "w") as h:
    json.dump(out, h, indent=1)
print("DONE", flush=True)
