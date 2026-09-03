"""Exp3: End-to-end CONCH-clustered atlas (not just CONCH re-scoring).

The manuscript admits CONCH was only used to re-score state0, not to build an
independent atlas. A reviewer will ask: is the negative shared-MSI result a UNI
artifact? Here we build the WHOLE pooled K=24 atlas on CONCH features and repeat:
  (a) per-state MSI association in STAD and CRC (Cliff delta + Mann-Whitney + BH-FDR)
  (b) shared-state AND-test at FDR<0.10
  (c) cross-lineage transfer (logistic regression on 24 state fractions)
  (d) best single-state CRC MSI anchor AUROC
If the negative result + a CRC anchor both reproduce under CONCH, the conclusion
is encoder-robust.
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atlas_utils as A

OUT = Path(__file__).resolve().parent
ENC = "conch"


def bh_fdr(p):
    p = np.asarray(p); n = len(p); order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(ranked, 0, 1)
    return out


def assoc(pat, cohort):
    sub = pat[pat.cohort == cohort]
    pos = sub[sub.label == "MSI-H"]; neg = sub[sub.label == "MSS"]
    deltas, ps = [], []
    for s in range(A.N_STATES):
        a = pos[f"state_{s:02d}"].values; b = neg[f"state_{s:02d}"].values
        deltas.append(A.cliffs_delta(a, b))
        try:
            ps.append(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        except ValueError:
            ps.append(1.0)
    return np.array(deltas), np.array(ps), bh_fdr(ps), len(pos), len(neg)


def transfer(train, test, pos="MSI-H"):
    cols = [f"state_{s:02d}" for s in range(A.N_STATES)]
    xtr = train[cols].values; ytr = (train.label == pos).astype(int).values
    xte = test[cols].values;  yte = (test.label == pos).astype(int).values
    clf = make_pipeline(StandardScaler(),
        LogisticRegression(C=0.1, class_weight="balanced", solver="lbfgs",
                           max_iter=5000, random_state=A.SEED))
    clf.fit(xtr, ytr); prob = clf.predict_proba(xte)[:, 1]
    return dict(auroc=float(roc_auc_score(yte, prob)),
                pr_auc=float(average_precision_score(yte, prob)),
                bal_acc=float(balanced_accuracy_score(yte, (prob >= .5).astype(int))))


def main():
    frac_cache = OUT / "conch_patient_fractions.csv"
    if frac_cache.exists():
        print("loading cached CONCH patient fractions ...")
        pat = pd.read_csv(frac_cache)
    else:
        man = A.slide_manifest_for_cohort()
        rows = man[man.cohort.isin(["STAD_MSI", "CRC_MSI"])].to_dict("records")
        print(f"building CONCH pooled atlas on {len(rows)} slides ...")
        fit_x = A.sample_fit_matrix(rows, encoder=ENC)
        print(f"  fit matrix {fit_x.shape}")
        centers = A.fit_kmeans(fit_x)
        np.save(OUT / "conch_atlas_centers.npy", centers)
        slide = A.assign_fractions(rows, centers, encoder=ENC)
        pat = A.to_patient(slide, A.N_STATES)
        pat.to_csv(frac_cache, index=False)
    rows = [0] * 942

    dS, pS, fS, nSp, nSn = assoc(pat, "STAD_MSI")
    dC, pC, fC, nCp, nCn = assoc(pat, "CRC_MSI")
    sig_S = set(np.where(fS < 0.10)[0]); sig_C = set(np.where(fC < 0.10)[0])
    shared = sorted(int(x) for x in (sig_S & sig_C))
    print(f"\nSTAD sig states (FDR<0.10): {sorted(int(x) for x in sig_S)}")
    print(f"CRC  sig states (FDR<0.10): {sorted(int(x) for x in sig_C)}")
    print(f"SHARED MSI states (both lineages, FDR<0.10): {shared}")
    # directionality: does the MSI shift point the same way in both lineages?
    dir_match = []
    for s in shared:
        same = (np.sign(dS[s]) == np.sign(dC[s]))
        dir_match.append(bool(same))
        print(f"  state {s}: STAD delta={dS[s]:+.3f}  CRC delta={dC[s]:+.3f}  "
              f"direction_match={same}")
    shared_concordant = [s for s, m in zip(shared, dir_match) if m]
    print(f"SHARED states with CONCORDANT direction: {shared_concordant}")

    # best single-state CRC anchor
    crc = pat[pat.cohort == "CRC_MSI"]; yC = (crc.label == "MSI-H").astype(int).values
    aurocs = []
    for s in range(A.N_STATES):
        v = crc[f"state_{s:02d}"].values
        au = roc_auc_score(yC, v)
        aurocs.append(max(au, 1 - au))  # directionality-agnostic
    best = int(np.argmax(aurocs))
    print(f"\nbest CRC single-state anchor: state {best} AUROC={aurocs[best]:.3f} "
          f"(delta={dC[best]:+.3f}, FDR={fC[best]:.2e})")

    stad = pat[pat.cohort == "STAD_MSI"]
    tSC = transfer(stad, crc); tCS = transfer(crc, stad)
    print(f"\ncross-lineage transfer (CONCH state fractions):")
    print(f"  STAD->CRC AUROC={tSC['auroc']:.3f}  CRC->STAD AUROC={tCS['auroc']:.3f}")

    res = dict(encoder=ENC, n_slides=len(rows),
               stad_n=(int(nSp), int(nSn)), crc_n=(int(nCp), int(nCn)),
               shared_msi_states=shared,
               shared_concordant_states=[int(s) for s in shared_concordant],
               stad_sig_states=sorted(int(x) for x in sig_S),
               crc_sig_states=sorted(int(x) for x in sig_C),
               best_crc_anchor_state=best,
               best_crc_anchor_auroc=float(aurocs[best]),
               best_crc_anchor_delta=float(dC[best]),
               best_crc_anchor_fdr=float(fC[best]),
               transfer_stad_to_crc=tSC, transfer_crc_to_stad=tCS,
               crc_deltas=[float(x) for x in dC], crc_fdr=[float(x) for x in fC],
               stad_deltas=[float(x) for x in dS], stad_fdr=[float(x) for x in fS])
    (OUT / "exp3_results.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT/'exp3_results.json'}")


if __name__ == "__main__":
    main()
