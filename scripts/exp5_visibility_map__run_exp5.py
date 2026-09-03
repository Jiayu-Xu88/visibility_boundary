"""Exp5: Continuous cross-lineage visibility map.

Turns the binary "is there a shared state?" question into a continuous map:
for each (phenotype, lineage) we quantify within-lineage VISIBILITY (best single
morphology-state AUROC + mutual information) and, for MSI, cross-lineage
TRANSFER AUROC. Produces a phenotype x axis table the paper can plot to show the
visibility boundary is graded, not all-or-nothing.

Uses the EXISTING frozen UNI pooled K=24 patient fractions (no recompute).
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atlas_utils as A

OUT = Path(__file__).resolve().parent
COLS = [f"state_{s:02d}" for s in range(A.N_STATES)]
pat = pd.read_csv(A.PATIENT_FRAC, sep="\t")


def best_state_auroc(df, y):
    aus = [max(a, 1 - a) for a in (roc_auc_score(y, df[c].values) for c in COLS)]
    return float(np.max(aus)), int(np.argmax(aus))


def cv_auroc(df, y):
    X = df[COLS].values
    skf = StratifiedKFold(5, shuffle=True, random_state=A.SEED)
    probs = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1,
            class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=A.SEED))
        clf.fit(X[tr], y[tr]); probs[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, probs))


def mi(df, y):
    return float(mutual_info_classif(df[COLS].values, y, random_state=A.SEED).sum())


def transfer(tr, te):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced",
        solver="lbfgs", max_iter=5000, random_state=A.SEED))
    clf.fit(tr[COLS].values, (tr.label == "MSI-H").astype(int).values)
    p = clf.predict_proba(te[COLS].values)[:, 1]
    return float(roc_auc_score((te.label == "MSI-H").astype(int).values, p))


def main():
    rows = []
    # --- MSI in each lineage (with cross-lineage transfer) ---
    crc = pat[pat.cohort == "CRC_MSI"]; stad = pat[pat.cohort == "STAD_MSI"]
    yC = (crc.label == "MSI-H").astype(int).values
    yS = (stad.label == "MSI-H").astype(int).values
    bC, sC = best_state_auroc(crc, yC); bS, sS = best_state_auroc(stad, yS)
    rows.append(dict(phenotype="MSI-H", lineage="CRC", n=len(crc), n_pos=int(yC.sum()),
        best_state_auroc=bC, best_state=sC, within_cv_auroc=cv_auroc(crc, yC), mutual_info=mi(crc, yC),
        cross_lineage_transfer_auroc=transfer(stad, crc), transfer_direction="STAD->CRC"))
    rows.append(dict(phenotype="MSI-H", lineage="STAD", n=len(stad), n_pos=int(yS.sum()),
        best_state_auroc=bS, best_state=sS, within_cv_auroc=cv_auroc(stad, yS), mutual_info=mi(stad, yS),
        cross_lineage_transfer_auroc=transfer(crc, stad), transfer_direction="CRC->STAD"))

    # --- STAD molecular subtypes (one-vs-rest), no cross-lineage counterpart ---
    sub = pat[pat.cohort == "STAD_SUBTYPE"]
    for st in ["EBV", "MSI", "GS", "CIN"]:
        y = (sub.molecular_subtype == st).astype(int).values
        if y.sum() < 5:
            continue
        b, s = best_state_auroc(sub, y)
        rows.append(dict(phenotype=f"subtype:{st}", lineage="STAD", n=len(sub), n_pos=int(y.sum()),
            best_state_auroc=b, best_state=s, within_cv_auroc=cv_auroc(sub, y), mutual_info=mi(sub, y),
            cross_lineage_transfer_auroc=None, transfer_direction=None))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "visibility_map.csv", index=False)
    print(df.to_string(index=False))
    (OUT / "exp5_results.json").write_text(df.to_json(orient="records", indent=2))
    print(f"\nwrote {OUT/'visibility_map.csv'}")


if __name__ == "__main__":
    main()
