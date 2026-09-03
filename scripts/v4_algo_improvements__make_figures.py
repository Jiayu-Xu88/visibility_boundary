import os
"""Generate the 3 new manuscript figures from saved v4 results (no recompute)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

V4 = Path(os.environ.get("PROJECT_ROOT",".")) / "Experiment/analysis/v4_algo_improvements"
OUT = Path(os.environ.get("PROJECT_ROOT",".")) / "Publication/paper_visibility_boundary/figures/v4"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.bbox": "tight"})

# ---------- Figure Y: continuous visibility map ----------
vm = pd.read_csv(V4 / "exp5_visibility_map/visibility_map.csv")
vm["lab"] = vm.phenotype + "\n(" + vm.lineage + ")"
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(vm)); w = 0.38
ax.bar(x - w/2, vm.within_cv_auroc, w, label="within-lineage 5-fold CV AUROC", color="#2c7fb8")
tr = vm.cross_lineage_transfer_auroc
ax.bar(x + w/2, tr.fillna(0), w, label="cross-lineage transfer AUROC (MSI only)", color="#d95f0e")
ax.axhline(0.5, ls="--", c="gray", lw=1, label="chance")
ax.set_xticks(x); ax.set_xticklabels(vm.lab, fontsize=8)
ax.set_ylabel("AUROC"); ax.set_ylim(0.4, 0.82)
ax.set_title("H&E molecular visibility is graded across phenotype and lineage")
for i, v in enumerate(tr):
    if pd.isna(v):
        ax.text(i + w/2, 0.41, "n/a", ha="center", fontsize=7, color="gray")
ax.legend(fontsize=8, loc="upper right")
fig.savefig(OUT / "figY_visibility_map.png"); plt.close(fig)

# ---------- OT panel: transported assoc + anchor twin ----------
ot = json.loads((V4 / "exp2_ot_matching/exp2_results.json").read_text())
dS = np.array(ot["stad_deltas"]); dC = np.array(ot["crc_deltas"])
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axes[0]
ax.scatter(dC, dS[np.argsort(dC)] * 0 + 0, alpha=0)  # placeholder to keep layout
ax.cla()
# scatter: each CRC state delta vs its OT-nearest STAD twin delta is summarized by rho
ax.scatter(dC, np.interp(np.argsort(np.argsort(dC)), np.arange(len(dS)), np.sort(dS)),
           s=10, alpha=0)  # not used; keep simple bars instead
ax.cla()
ax.bar(np.arange(len(dC)), dC, color="#d95f0e")
ax.set_title(f"CRC states: MSI-H association\n(anchor {ot['crc_anchor_state']}, δ={ot['crc_anchor_delta']:+.3f})",
             fontsize=9)
ax.set_xlabel("CRC state index"); ax.set_ylabel("Cliff δ (MSI-H vs MSS)")
ax.axhline(0, c="k", lw=0.6)
ax = axes[1]
labels = ["CRC anchor", "STAD morph. twin", "STAD near#2", "STAD near#3"]
vals = [ot["crc_anchor_delta"], ot["stad_twin_delta"]] + ot["nearest_stad_deltas"][:2]
# stad_twin may equal nearest[0]; show twin + next two distinct
vals = [ot["crc_anchor_delta"]] + ot["nearest_stad_deltas"][:3]
labels = ["CRC anchor"] + [f"STAD twin#{i+1}" for i in range(3)]
colors = ["#d95f0e", "#2c7fb8", "#2c7fb8", "#2c7fb8"]
ax.bar(labels, vals, color=colors)
ax.axhline(0, c="k", lw=0.6)
ax.set_ylabel("Cliff δ (MSI-H vs MSS)")
ax.set_title(f"CRC anchor's STAD morphological twins carry ~0 MSI signal\n"
             f"OT-transported ρ={ot['ot_transport_spearman_rho']:.3f}, perm p={ot['perm_two_sided_p']:.2f}",
             fontsize=9)
plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
fig.tight_layout(w_pad=3)
fig.savefig(OUT / "figOT_cross_lineage_matching.png"); plt.close(fig)

# ---------- CONCH concept panel ----------
c1 = json.loads((V4 / "exp1_conch_concepts/exp1_results.json").read_text())
ct = pd.DataFrame(c1["concept_tests"]).sort_values("cliffs_delta")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
colors = ["#d95f0e" if g == "MSI" else "#7f7f7f" for g in ct.group]
ax.barh(ct.concept, ct.cliffs_delta, color=colors)
ax.axvline(0, c="k", lw=0.6)
ax.set_xlabel("Cliff δ (state0 vs background CONCH similarity)")
ax.set_title(f"Zero-shot concept enrichment of CRC_MSI_state0\n"
             f"(n={c1['n_state0']} state0 / {c1['n_bg']} background patches)")
ax.tick_params(axis="y", labelsize=8)
ax = axes[1]
top = pd.Series(c1["state0_top1_distribution"]).sort_values(ascending=True)
top = top[top > 0.01]
ax.barh(top.index, top.values, color="#d95f0e")
ax.set_xlabel("fraction of state0 patches (top-1 concept)")
ax.set_title("state0 dominant histologic concept")
ax.tick_params(axis="y", labelsize=8)
fig.savefig(OUT / "figCONCH_state0_concepts.png"); plt.close(fig)

print("wrote:")
for f in sorted(OUT.glob("*.png")):
    print(" ", f)
