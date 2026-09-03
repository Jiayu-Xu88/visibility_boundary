"""Exp2: Optimal-transport cross-lineage state matching.

Question the original AND-test could not answer: maybe a shared MSI morphology
program exists but lands in *different* pooled centroids in STAD vs CRC, so the
strict "same centroid significant in both" test misses it. Here we build
lineage-SPECIFIC K=24 atlases, match STAD<->CRC states by morphology (centroid
distance) via optimal transport, and test whether MSI-association TRANSFERS along
the morphological matching. Two outcomes, both informative:
  - transported STAD MSI-association correlates with CRC's  -> shared program
  - it does not                                              -> genuine lineage-dependence
Also reports the morphological twin of the CRC MSI anchor state.
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atlas_utils as A

OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(0)


def sinkhorn(cost, eps=0.05, iters=500):
    K = np.exp(-cost / eps)
    n, m = cost.shape
    u = np.ones(n) / n; v = np.ones(m) / m
    a = np.ones(n) / n; b = np.ones(m) / m
    for _ in range(iters):
        u = a / (K @ v + 1e-12)
        v = b / (K.T @ u + 1e-12)
    return u[:, None] * K * v[None, :]


def lineage_atlas(rows, tag):
    rows = list(rows)
    fit_x = A.sample_fit_matrix(rows)
    centers = A.fit_kmeans(fit_x)
    slide = A.assign_fractions(rows, centers)
    pat = A.to_patient(slide, A.N_STATES)
    # MSI association (Cliff delta) per state
    pos = pat[pat.label == "MSI-H"]; neg = pat[pat.label == "MSS"]
    deltas = np.array([A.cliffs_delta(pos[f"state_{s:02d}"], neg[f"state_{s:02d}"])
                       for s in range(A.N_STATES)])
    print(f"  [{tag}] patients={len(pat)} MSI-H={len(pos)} MSS={len(neg)} "
          f"max|delta|={np.nanmax(np.abs(deltas)):.3f} at state {int(np.nanargmax(np.abs(deltas)))}")
    return centers, deltas, pat


def main():
    man = A.slide_manifest_for_cohort()
    stad = man[man.cohort == "STAD_MSI"].to_dict("records")
    crc = man[man.cohort == "CRC_MSI"].to_dict("records")
    print(f"STAD slides={len(stad)} CRC slides={len(crc)}")

    cS, dS, patS = lineage_atlas(stad, "STAD")
    cC, dC, patC = lineage_atlas(crc, "CRC")

    # cost = pairwise euclidean between centroids (morphology distance)
    cost = np.linalg.norm(cS[:, None, :] - cC[None, :, :], axis=2)
    # normalize cost for sinkhorn stability
    costn = cost / cost.mean()

    # --- Soft OT coupling ---
    P = sinkhorn(costn, eps=0.05)
    # transport STAD deltas onto CRC index
    w = P / (P.sum(axis=0, keepdims=True) + 1e-12)
    dS_to_C = (w * dS[:, None]).sum(axis=0)   # weighted STAD delta at each CRC state
    rho, p = spearmanr(dS_to_C, dC)
    print(f"\nOT-transported STAD->CRC MSI-association vs CRC MSI-association: "
          f"Spearman rho={rho:.3f} p={p:.3g}")

    # permutation null: shuffle CRC state labels
    null = []
    for _ in range(2000):
        perm = rng.permutation(A.N_STATES)
        null.append(spearmanr(dS_to_C, dC[perm]).correlation)
    null = np.array(null)
    p_perm = (np.sum(np.abs(null) >= abs(rho)) + 1) / (len(null) + 1)
    print(f"  permutation null mean={np.nanmean(null):.3f} "
          f"95%=[{np.nanpercentile(null,2.5):.3f},{np.nanpercentile(null,97.5):.3f}] "
          f"two-sided p={p_perm:.4f}")

    # --- Hard 1-1 matching (Hungarian) for interpretability ---
    ri, ci = linear_sum_assignment(cost)
    match = {int(i): int(j) for i, j in zip(ri, ci)}
    # CRC MSI anchor = CRC state with max positive delta
    crc_anchor = int(np.nanargmax(dC))
    # which STAD state maps to it
    stad_twin = int(ci.tolist().index(crc_anchor)) if crc_anchor in ci.tolist() else None
    stad_twin = int(ri[list(ci).index(crc_anchor)]) if crc_anchor in list(ci) else None
    print(f"\nCRC MSI anchor = CRC-atlas state {crc_anchor} (delta={dC[crc_anchor]:+.3f})")
    if stad_twin is not None:
        print(f"  morphological twin in STAD = STAD-atlas state {stad_twin} "
              f"(STAD MSI delta={dS[stad_twin]:+.3f}, centroid dist={cost[stad_twin,crc_anchor]:.2f})")
    # top-3 nearest STAD states to the CRC anchor + their STAD MSI deltas
    near = np.argsort(cost[:, crc_anchor])[:3]
    print("  3 nearest STAD states to CRC anchor (morphology) and their STAD MSI deltas:")
    for s in near:
        print(f"    STAD state {int(s)}: dist={cost[s,crc_anchor]:.2f}  STAD MSI delta={dS[s]:+.3f}")

    res = dict(
        stad_patients=int(len(patS)), crc_patients=int(len(patC)),
        ot_transport_spearman_rho=float(rho), ot_transport_p=float(p),
        perm_null_mean=float(np.nanmean(null)),
        perm_null_ci=[float(np.nanpercentile(null,2.5)), float(np.nanpercentile(null,97.5))],
        perm_two_sided_p=float(p_perm),
        crc_anchor_state=crc_anchor, crc_anchor_delta=float(dC[crc_anchor]),
        stad_twin_state=stad_twin,
        stad_twin_delta=float(dS[stad_twin]) if stad_twin is not None else None,
        nearest_stad_states=[int(s) for s in near],
        nearest_stad_deltas=[float(dS[s]) for s in near],
        stad_deltas=[float(x) for x in dS], crc_deltas=[float(x) for x in dC],
    )
    (OUT / "exp2_results.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT/'exp2_results.json'}")


if __name__ == "__main__":
    main()
