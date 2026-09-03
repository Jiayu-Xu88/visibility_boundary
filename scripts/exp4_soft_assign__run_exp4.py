"""Exp4: Soft-assignment robustness of the negative shared-MSI result.

The manuscript's atlas uses hard K-means argmin assignment. A reviewer can ask
whether the negative result is a hard-boundary artifact. Here we keep the SAME
frozen UNI K=24 centroids but replace argmin with a temperature-softmax soft
assignment (each patch contributes fractional mass to every state), recompute
patient soft-fractions, and repeat the shared-state AND-test, the best CRC anchor,
and cross-lineage transfer. Robust negative => not a hard-assignment artifact.
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atlas_utils as A

OUT = Path(__file__).resolve().parent
CENTERS = np.load(A.CENTERS)  # frozen UNI pooled K=24 centroids [24,1024]


def bh_fdr(p):
    p = np.asarray(p); n = len(p); order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(ranked, 0, 1); return out


def soft_fraction(x, centers, T):
    # squared euclidean distance to each centroid, softmax(-d2/T) per patch
    d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # [N,K]
    d2 -= d2.min(axis=1, keepdims=True)
    w = np.exp(-d2 / T)
    w /= w.sum(axis=1, keepdims=True) + 1e-12
    return w.mean(axis=0)  # [K] soft fraction


def build_soft(rows, T):
    recs = []
    for r in rows:
        x = A.load_features(r["feature_path"])
        idx = A.stable_indices(r["file_id"] + ":assign", x.shape[0], A.PATCHES_ASSIGN)
        frac = soft_fraction(x[idx].astype(np.float32), CENTERS, T)
        rec = dict(cohort=r["cohort"], case_submitter_id=r["case_submitter_id"],
                   label=r["label"])
        for s in range(A.N_STATES):
            rec[f"state_{s:02d}"] = frac[s]
        recs.append(rec)
    df = pd.DataFrame(recs)
    cols = [f"state_{s:02d}" for s in range(A.N_STATES)]
    return df.groupby(["cohort", "case_submitter_id", "label"])[cols].mean().reset_index()


def assoc(pat, cohort):
    sub = pat[pat.cohort == cohort]; pos = sub[sub.label == "MSI-H"]; neg = sub[sub.label == "MSS"]
    d, p = [], []
    for s in range(A.N_STATES):
        a = pos[f"state_{s:02d}"].values; b = neg[f"state_{s:02d}"].values
        d.append(A.cliffs_delta(a, b)); p.append(mannwhitneyu(a, b).pvalue)
    return np.array(d), bh_fdr(p)


def transfer(tr, te):
    cols = [f"state_{s:02d}" for s in range(A.N_STATES)]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced",
        solver="lbfgs", max_iter=5000, random_state=A.SEED))
    clf.fit(tr[cols].values, (tr.label == "MSI-H").astype(int).values)
    prob = clf.predict_proba(te[cols].values)[:, 1]
    return float(roc_auc_score((te.label == "MSI-H").astype(int).values, prob))


def main():
    man = A.slide_manifest_for_cohort()
    rows = man[man.cohort.isin(["STAD_MSI", "CRC_MSI"])].to_dict("records")
    out = {}
    for T in [0.5, 1.0, 2.0]:
        print(f"\n=== soft assignment T={T} ===")
        pat = build_soft(rows, T)
        dS, fS = assoc(pat, "STAD_MSI"); dC, fC = assoc(pat, "CRC_MSI")
        sig_S = set(np.where(fS < .10)[0]); sig_C = set(np.where(fC < .10)[0])
        shared = sorted(int(x) for x in sig_S & sig_C)
        concordant = [s for s in shared if np.sign(dS[s]) == np.sign(dC[s])]
        crc = pat[pat.cohort == "CRC_MSI"]; yC = (crc.label == "MSI-H").astype(int).values
        au = [max(a, 1 - a) for a in (roc_auc_score(yC, crc[f"state_{s:02d}"].values) for s in range(A.N_STATES))]
        best = int(np.argmax(au))
        stad = pat[pat.cohort == "STAD_MSI"]
        tSC = transfer(stad, crc); tCS = transfer(crc, stad)
        print(f"  shared(both FDR<.10)={shared} concordant={concordant}")
        print(f"  best CRC anchor: state {best} AUROC={au[best]:.3f}")
        print(f"  transfer STAD->CRC={tSC:.3f} CRC->STAD={tCS:.3f}")
        out[f"T={T}"] = dict(shared=shared, concordant=[int(s) for s in concordant],
                             best_crc_anchor_state=best, best_crc_anchor_auroc=float(au[best]),
                             transfer_stad_to_crc=tSC, transfer_crc_to_stad=tCS)
    (OUT / "exp4_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT/'exp4_results.json'}")


if __name__ == "__main__":
    main()
