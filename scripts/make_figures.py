#!/usr/bin/env python3
"""Main figures, rendered from the approved figure plan.

Each figure defends one conclusion (FIGURE_DESIGN_BRIEF.yaml, approved):
  F1  prediction transfers across the organ boundary; the low-dimensional
      morphology representation does not.                    hero: b
  F2  that failing representation is nonetheless real in both organs. hero: b
  F3  the null is neither method insensitivity nor centre structure. two co-heroes
  F4  the transferable signal is distributed and tumour-associated.  hero: a

House contract: 178 mm double column, 7 pt sans, left and bottom spines only,
ticks outward, no tinted panel backgrounds, editable Type 42 text, 600 dpi.
Captions live in the manuscript, never on the canvas.
"""
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

P = Path("/home/ubuntu/shared/proj-Gastrointestinal/Publication/paper_visibility_boundary")
REPRO = P / "github_repo/reproducibility"
ATLAS = Path("/home/ubuntu/shared/proj-Gastrointestinal/Experiment/analysis/medical_morphology_atlas")
EX_DIR = ATLAS / "state_exemplars"

J = lambda f: json.load(open(REPRO / f))
lad = J("e1_ladder_manifest.json")["levels"]
ext = J("coupling_extensions_manifest.json")
lad["RT_tstage"] = {"transfer": ext["S2_new_rungs"]["tstage_T34"]["transfer"]}
cpl = ext["S1_coupling"]
bench = J("m3_center_benchmark_manifest.json")["M3_within_organ_cross_center"]
sm = J("m3c_size_matched_manifest.json")
cv = J("m4_internal_external_cv_manifest.json")
m11 = J("m11_reconstruction_control_manifest.json")["targets"]
e2c = J("e2c_mil_projection_manifest.json")
audit_full = J("e2_representation_audit_manifest.json")
audit = audit_full["E2a"]
surgen = J("surgen_state0_bootstrap_ci.json")["by_label_view"]["dmmr_msi_composite"]["auroc"]
did = json.load(open(P / "did_label_contrast_manifest.json"))

MM = 1 / 25.4
W = 178 * MM
BLUE, RED, GREY, AMBER = "#2C6FBB", "#C1352F", "#8C8C8C", "#DE8A2B"
PALE, INK = "#B9CBE3", "#1A1A1A"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.3,
    "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.pad": 1.8, "ytick.major.pad": 1.8,
    "legend.frameon": False, "lines.linewidth": 0.9,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
})


def clean(ax, left=True):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_visible(left)
    if not left:
        ax.tick_params(axis="y", length=0)


def letter(ax, ch, dx=-0.06, dy=1.0):
    ax.text(dx, dy, ch, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="left")


def flabel(fig, ch, y, x=0.012):
    """Panel letter in FIGURE coordinates, so every letter in a figure sits at
    the same distance from the page edge."""
    fig.text(x, y, ch, fontsize=9, fontweight="bold", va="top", ha="left")


def tile_row(fig, ims, left, right, top, n_cols, n_rows, gap=0.06,
             edge=None, edge_lw=0.5):
    """Lay a grid of square image tiles that spans exactly left..right.

    imshow keeps an equal aspect, so tiles placed by a gridspec shrink and
    centre themselves; positioning them explicitly is what makes the row line up
    with the data panels.
    """
    w_fig, h_fig = fig.get_size_inches()
    tw = (right - left) / (n_cols + (n_cols - 1) * gap)
    th = tw * w_fig / h_fig
    axes = []
    for r in range(n_rows):
        for c in range(n_cols):
            x = left + c * tw * (1 + gap)
            y = top - (r + 1) * th - r * th * gap
            ax = fig.add_axes([x, y, tw, th])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(edge_lw)
                sp.set_color(edge or INK)
            k = r * n_cols + c
            if k < len(ims) and ims[k] is not None:
                ax.imshow(ims[k], aspect="auto")
            else:
                ax.set_facecolor("#F2F2F2")
            axes.append(ax)
    return axes, th


def tiles(folder, k=4, thresh=225):
    out = []
    for f in sorted(Path(folder).glob("rank*.png")):
        im = Image.open(f)
        if np.asarray(im.convert("L")).mean() < thresh:
            out.append(im)
        if len(out) == k:
            break
    return out


def save(fig, name):
    for ext_ in ("pdf", "png"):
        fig.savefig(P / f"figures/{name}.{ext_}")
    bbox_check(fig, name)
    plt.close(fig)


# ===================================================================== F1 ===
def figure1():
    """Hero: b. a is the design bridge that makes b interpretable."""
    fig = plt.figure(figsize=(W, 92 * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.72, 1.0], hspace=0.34,
                          left=0.055, right=0.965, top=0.94, bottom=0.115)
    axA = fig.add_subplot(gs[0]); axB = fig.add_subplot(gs[1])

    # the schematic's boxes run to x = 0.93 in its own coordinates; ending the
    # axis there makes the schematic span the same width as the data panel
    axA.set_axis_off(); axA.set_xlim(0, 0.93); axA.set_ylim(0, 1)

    def box(xy, w, h, txt, ec=INK, fc="none", fs=6.2, bold=False):
        axA.add_patch(Rectangle(xy, w, h, lw=0.6, ec=ec, fc=fc))
        axA.text(xy[0] + w / 2, xy[1] + h / 2, txt, ha="center", va="center",
                 fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.35)

    def arrow(a, b, c=INK):
        axA.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=5.5,
                                      lw=0.6, color=c, shrinkA=0, shrinkB=0))

    box((0.00, 0.60), 0.180, 0.30, "TCGA colorectal\n534 patients, 71 MSI-H", fs=6.0)
    box((0.00, 0.14), 0.180, 0.30, "TCGA gastric\n376 patients, 60 MSI-H", fs=6.0)
    box((0.220, 0.37), 0.200, 0.30, "frozen foundation-model\npatch features", fs=5.8)
    arrow((0.182, 0.75), (0.218, 0.60)); arrow((0.182, 0.29), (0.218, 0.44))
    box((0.450, 0.60), 0.215, 0.30, "supervised\nmultiple-instance classifier", ec=BLUE)
    box((0.450, 0.14), 0.215, 0.30, "K = 24 morphology states\nper patient", ec=RED)
    arrow((0.422, 0.60), (0.448, 0.75), BLUE); arrow((0.422, 0.44), (0.448, 0.29), RED)
    box((0.715, 0.60), 0.215, 0.30, "transfer to gastric\nAUROC 0.73", ec=BLUE,
        fc="#EAF1F9", fs=6.6, bold=True)
    box((0.715, 0.14), 0.215, 0.30, "transfer to gastric\nAUROC 0.54", ec=RED,
        fc="#F8ECEB", fs=6.6, bold=True)
    arrow((0.667, 0.75), (0.713, 0.75), BLUE); arrow((0.667, 0.29), (0.713, 0.29), RED)
    axA.text(0.450, 0.955, "route 1  prediction", fontsize=6.5, color=BLUE, fontweight="bold")
    axA.text(0.450, 0.035, "route 2  the representation used to interpret it",
             fontsize=6.5, color=RED, fontweight="bold")
    flabel(fig, "a", 0.985)

    MUC = lad["R2_mucinous_icdo"]["transfer"]["CRC_to_STAD"]
    rows = [("supervised classifier", 0.73, None, BLUE),
            ("mucinous histology", MUC["auroc"], MUC["boot95"], BLUE),
            ("MSI-H, matched training size",
             sm["A_size_matched_cross_organ"]["CRC_to_STAD"]["auroc_median"], None, RED),
            ("MSI-H, full training cohort", 0.593, [0.512, 0.672], RED),
            ("MSI-H, across gastric centres",
             bench["STAD"]["auroc_median"], bench["STAD"]["auroc_iqr"], GREY),
            ("MSI-H, leave-one-centre-out",
             cv["STAD"]["MSI_H"]["loco_pooled_auroc"], cv["STAD"]["MSI_H"]["loco_boot95"], GREY)]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, ci, c) in zip(y, rows):
        if ci:
            axB.plot(ci, [yi, yi], color=c, lw=0.9, solid_capstyle="butt", zorder=2)
        axB.plot(v, yi, "o", ms=4.0, mfc=c, mec=c, zorder=3)
        axB.text(v, yi + 0.22, f"{v:.2f}", fontsize=6.3, color=c, ha="center", va="bottom")
    axB.axvline(0.5, lw=0.6, ls=(0, (1, 2)), color=GREY, zorder=0)
    axB.axvline(0.62, lw=0.6, ls=(0, (3, 2)), color=GREY, zorder=0)
    axB.text(0.503, -0.86, "chance", fontsize=5.9, color=GREY, ha="left")
    axB.text(0.623, -0.86, "equivalence bound", fontsize=5.9, color=GREY, ha="left")
    axB.set_yticks(y); axB.set_yticklabels([r[0] for r in rows])
    axB.set_ylim(-1.15, len(rows) - 0.35); axB.set_xlim(0.48, 0.90)
    axB.set_xlabel("AUROC on the same 376 gastric patients", labelpad=1)
    clean(axB, left=False)
    for yy, txt, c in [(4.5, "transfers", BLUE), (2.5, "does not", RED),
                       (0.5, "within-gastric reference", GREY)]:
        axB.text(0.898, yy, txt, fontsize=6.1, color=c, ha="right",
                 va="center", style="italic")
    flabel(fig, "b", 0.545)
    # pin the schematic to the data panel's axes box so the two panels share a
    # left and right edge; the y tick labels would otherwise offset only one
    _pb, _pa = axB.get_position(), axA.get_position()
    axA.set_position([_pb.x0, _pa.y0, _pb.width, _pa.height])
    save(fig, "Figure_1")
    print("Figure 1")


# ===================================================================== F2 ===
def figure2():
    """Hero: b. a defines the object, c and d make it inspectable."""
    fig = plt.figure(figsize=(W, 186 * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.50, 0.78], hspace=0.50,
                          top=0.972, bottom=0.640, left=0.085, right=0.965)
    # tile geometry drives where the image panels sit, so widening the tiles
    # can never push one panel onto another again
    L, Rr, GAP = 0.085, 0.965, 0.06
    _tw = (Rr - L) / (8 + 7 * GAP)
    _th = _tw * fig.get_size_inches()[0] / fig.get_size_inches()[1]
    _rowh = 2 * _th + GAP * _th
    TOP_C = 0.560
    TOP_D = TOP_C - _rowh - 0.070
    axA = fig.add_subplot(gs[0]); axB = fig.add_subplot(gs[1])

    assoc = pd.read_csv(REPRO / "gi_morphology_state_associations.tsv", sep="\t")
    mat, sig = [], []
    for key in ("CRC_MSI_H_vs_MSS", "STAD_MSI_H_vs_MSS"):
        sub = assoc[assoc.analysis == key].sort_values("state")
        mat.append(sub.cliffs_delta.to_numpy())
        sig.append((sub.bh_fdr.to_numpy() < 0.10).astype(int))
    mat, sig = np.vstack(mat), np.vstack(sig)
    im = axA.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-0.45, vmax=0.45)
    axA.set_yticks([0, 1]); axA.set_yticklabels(["colorectal", "gastric"])
    axA.set_xticks(range(0, 24, 2)); axA.set_xticklabels(range(0, 24, 2))
    axA.set_xlabel("morphology state", labelpad=1)
    for i in range(2):
        for j in range(24):
            if sig[i, j]:
                axA.text(j, i, "∗", ha="center", va="center", fontsize=5.2)
    axA.add_patch(Rectangle((-0.5, -0.5), 1, 2, fill=False, ec=INK, lw=1.0, zorder=4))
    axA.text(0, -0.80, "state 0", fontsize=6.0, ha="center", va="bottom")
    for sp in axA.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=axA, fraction=0.022, pad=0.010, aspect=8)
    cb.outline.set_linewidth(0.5); cb.ax.tick_params(labelsize=5.8, width=0.5, length=1.8)
    cb.ax.set_title("Cliff $\\delta$", fontsize=6.2, pad=3)
    flabel(fig, "a", 0.995)

    # hero: does the within-lineage signal survive centre shift?
    for i, (coh, name) in enumerate([("CRC", "colorectal"), ("STAD", "gastric")]):
        rnd_, loco = cv[coh]["MSI_H"]["random_5fold_cv_auroc"], cv[coh]["MSI_H"]["loco_pooled_auroc"]
        ci = cv[coh]["MSI_H"]["loco_boot95"]
        col = RED if rnd_ - loco > 0.05 else BLUE
        axB.plot([i - 0.28, i + 0.28], [rnd_, loco], "-o", color=col, lw=1.0,
                 ms=4.0, mfc=col, mec=col, zorder=3)
        axB.plot([i + 0.28] * 2, ci, color=col, lw=0.9, solid_capstyle="butt", zorder=2)
        axB.text(i + 0.34, loco, f"{loco - rnd_:+.3f}", fontsize=6.2, color=col,
                 va="center", ha="left", fontweight="bold")
        axB.text(i - 0.34, rnd_, f"{rnd_:.2f}", fontsize=6.2, color=GREY,
                 va="center", ha="right")
    axB.plot([2.60], [surgen["mean"]], "D", ms=4.2, mfc="white", mec=INK, mew=0.9, zorder=3)
    axB.plot([2.60] * 2, [surgen["ci_low_2_5"], surgen["ci_high_97_5"]], color=INK,
             lw=0.9, solid_capstyle="butt", zorder=2)
    axB.text(2.76, surgen["mean"], "SurGen\nexternal", fontsize=6.1, va="center", ha="left")
    axB.axhline(0.5, lw=0.6, ls=(0, (1, 2)), color=GREY, zorder=0)
    axB.set_xticks([-0.28, 0.28, 0.72, 1.28, 2.60])
    axB.set_xticklabels(["random\nfolds", "held-out\ncentres", "random\nfolds",
                         "held-out\ncentres", "frozen\nstate 0"], fontsize=5.9)
    axB.set_xlim(-0.72, 3.45); axB.set_ylim(0.48, 0.80)
    axB.set_ylabel("within-lineage AUROC", labelpad=1)
    axB.text(0.0, 1.02, "colorectal", ha="center", fontsize=6.4, style="italic",
             transform=axB.get_xaxis_transform())
    axB.text(1.0, 1.02, "gastric", ha="center", fontsize=6.4, style="italic",
             transform=axB.get_xaxis_transform())
    clean(axB)
    flabel(fig, "b", 0.800)

    # c: the same state in both organs.  d: the four gastric states that shift with MSI-H.
    ims_c = tiles(EX_DIR / "CRC_MSI_state00", 8) + tiles(EX_DIR / "STAD_state00", 8)
    axes_c, th_c = tile_row(fig, ims_c, L, Rr, TOP_C, 8, 2, gap=GAP)
    for r, name in enumerate(["colorectal", "gastric"]):
        axes_c[r * 8].set_ylabel(name, fontsize=5.9, labelpad=2)
    flabel(fig, "c", TOP_C + 0.030)

    gastric = [(22, "state 22 \u2191", RED, "tumour"), (18, "state 18 \u2191", RED, "mixed"),
               (23, "state 23 \u2193", BLUE, "muscle"),
               (5, "state 5 \u2193", BLUE, "normal mucosa")]
    top_d = TOP_D
    ims_d = [None] * 16
    for gi, (st, name, col, tissue) in enumerate(gastric):
        got = tiles(EX_DIR / f"STAD_state{st:02d}", 4)
        for k in range(4):
            rr, cc = divmod(k, 2)
            if k < len(got):
                ims_d[rr * 8 + gi * 2 + cc] = got[k]
    axes_d, th_d = tile_row(fig, ims_d, L, Rr, top_d, 8, 2, gap=GAP)
    tw_d = _tw
    for gi, (st, name, col, tissue) in enumerate(gastric):
        for k in range(4):
            rr, cc = divmod(k, 2)
            ax = axes_d[rr * 8 + gi * 2 + cc]
            for sp in ax.spines.values():
                sp.set_color(col)
        axes_d[gi * 2].set_title(name, fontsize=6.0, loc="left", pad=2, color=col)
        fig.text(L + gi * 2 * tw_d * 1.06, top_d - 2 * th_d - th_d * 0.06 - 0.012,
                 tissue, fontsize=5.7, color=GREY, va="top")
    flabel(fig, "d", TOP_D + 0.030)
    fig.text(0.5, TOP_D - _rowh - 0.048,
             "c  state 0 in both organs        d  gastric states that shift with"
             " MSI-H;  \u2191 enriched, \u2193 depleted", ha="center", fontsize=5.9, color=GREY)

    save(fig, "Figure_2")
    print("Figure 2")


# ===================================================================== F3 ===
def figure3():
    """Two co-heroes: a excludes method insensitivity, b addresses batch collinearity."""
    fig = plt.figure(figsize=(W, 62 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.26, 1.02, 0.62], wspace=0.40)
    axA = fig.add_subplot(gs[0]); axB = fig.add_subplot(gs[1]); axC = fig.add_subplot(gs[2])

    order = [("R0_MUC_q4", "mucus", GREY), ("R1b_DEB_q4", "debris", GREY),
             ("R1c_STR_q4", "stroma", GREY), ("R1a_LYM_q4", "lymphocyte", GREY),
             (None, None, None),
             ("R2_mucinous_icdo", "mucinous", BLUE),
             ("R2m_mucinous_or_signet", "muc./signet", BLUE),
             ("RT_tstage", "T3/T4", BLUE), ("R3_MSI_H", "MSI-H", RED)]
    xs, labs = [], []
    for i, (key, lab, c) in enumerate(order):
        if key is None:
            axA.axvline(i, lw=0.5, color=GREY, ls=(0, (2, 2)))
            continue
        t = lad[key]["transfer"]["STAD_to_CRC"]
        lo, hi = t["boot95"]
        axA.plot([i, i], [lo, hi], lw=0.9, color=c, solid_capstyle="butt", zorder=2)
        axA.plot(i, t["auroc"], "o", ms=4.0, mfc=c, mec=c, zorder=3)
        if t["perm_p"] < 0.05:
            axA.text(i, hi + 0.010, "∗" * (2 if t["perm_p"] < 0.01 else 1),
                     ha="center", fontsize=6.2, color=c)
        xs.append(i); labs.append(lab)
    axA.axhline(0.5, lw=0.6, ls=(0, (1, 2)), color=GREY, zorder=0)
    axA.set_xticks(xs); axA.set_xticklabels(labs, rotation=28, ha="right")
    axA.set_xlim(-0.7, len(order) - 0.3); axA.set_ylim(0.42, 0.90)
    axA.set_ylabel("cross-organ transfer AUROC", labelpad=1)
    axA.text(1.5, 1.01, "probe-derived", fontsize=6.0, color=GREY, ha="center",
             va="bottom", style="italic", transform=axA.get_xaxis_transform())
    axA.text(6.5, 1.01, "pathology records", fontsize=6.0, color=INK, ha="center",
             va="bottom", style="italic", transform=axA.get_xaxis_transform())
    clean(axA)

    XORG = {"CRC": sm["A_size_matched_cross_organ"]["STAD_to_CRC"]["auroc_median"],
            "STAD": sm["A_size_matched_cross_organ"]["CRC_to_STAD"]["auroc_median"]}
    MUC_XORG = {"CRC": lad["R2_mucinous_icdo"]["transfer"]["STAD_to_CRC"]["auroc"],
                "STAD": lad["R2_mucinous_icdo"]["transfer"]["CRC_to_STAD"]["auroc"]}
    MUC_XCEN = {c: sm["B_positive_control_cross_center"][c]["auroc_median"]
                for c in ("CRC", "STAD")}
    for off, coh in [(0.0, "CRC"), (1.0, "STAD")]:
        for lab, xcen, xorg, col, fill, ls in [
                ("MSI-H", bench[coh]["auroc_median"], XORG[coh], RED, None, "-"),
                ("mucinous", MUC_XCEN[coh], MUC_XORG[coh], BLUE, "white", (0, (3, 1.6)))]:
            axB.plot([off - 0.20, off + 0.20], [xcen, xorg], "o", color=col, ls=ls,
                     lw=0.9, ms=3.8, mfc=fill or col, mec=col, mew=0.9, zorder=3)
            axB.text(off + 0.25, xorg, f"{xorg - xcen:+.2f}", fontsize=6.1, color=col,
                     va="center", ha="left", fontweight="bold")
            pass  # series are named once in the panel corner below
    axB.text(0.98, 0.985, "MSI-H", fontsize=6.0, color=RED, fontweight="bold",
             transform=axB.transAxes, va="top", ha="right")
    axB.text(0.98, 0.905, "mucinous", fontsize=6.0, color=BLUE,
             transform=axB.transAxes, va="top", ha="right")
    axB.set_xticks([-0.20, 0.20, 0.80, 1.20])
    axB.set_xticklabels(["centres", "organs"] * 2, fontsize=5.8)
    axB.set_xlim(-0.45, 1.52); axB.set_ylim(0.48, 0.80)
    axB.set_yticks([0.50, 0.55, 0.60, 0.65, 0.70, 0.75])
    axB.set_ylabel("transfer AUROC", labelpad=1)
    axB.text(0.0, 1.02, "colorectal", ha="center", fontsize=6.4, style="italic",
             transform=axB.get_xaxis_transform())
    axB.text(1.0, 1.02, "gastric", ha="center", fontsize=6.4, style="italic",
             transform=axB.get_xaxis_transform())
    axB.spines["bottom"].set_bounds(-0.45, 1.45)
    axB.spines["left"].set_bounds(0.50, 0.75)
    clean(axB)

    # c: the change each label pays, and their difference, with 95% CIs
    rows = []
    for coh in ("CRC", "STAD"):
        d = did[f"evaluated_on_{coh}"]
        rows.append((coh, [("\u0394 MSI-H", d["delta_MSI"], d["delta_MSI_ci95"], RED),
                           ("\u0394 control", d["delta_control"], d["delta_control_ci95"], BLUE),
                           ("DiD", d["difference_in_differences"], d["did_ci95"], INK)]))
    yy = 0
    ylabels = []
    for coh, items in rows:
        for lab, v, ci, col in items:
            axC.plot(ci, [yy, yy], lw=0.9, color=col, solid_capstyle="butt")
            axC.plot(v, yy, "o", ms=3.4, mfc=col if col is not INK else "white",
                     mec=col, mew=0.9)
            ylabels.append(lab)
            yy -= 1
        yy -= 0.7
    axC.axvline(0, lw=0.6, ls=(0, (1, 2)), color=GREY, zorder=0)
    axC.set_yticks([0, -1, -2, -3.7, -4.7, -5.7])
    axC.set_yticklabels(ylabels, fontsize=5.9)
    axC.set_ylim(-6.5, 0.8)
    axC.text(-0.40, 0.62, "colorectal", fontsize=6.0, style="italic")
    axC.text(-0.40, -3.10, "gastric", fontsize=6.0, style="italic")
    axC.set_xlabel("change in transfer AUROC", labelpad=1)
    axC.set_xlim(-0.42, 0.28)
    clean(axC)

    for ax, ch in ((axA, "a"), (axB, "b"), (axC, "c")):
        fig.text(ax.get_position().x0 - 0.045, 0.955, ch, fontsize=9,
                 fontweight="bold", va="bottom", ha="left")
    save(fig, "Figure_3")
    print("Figure 3")


# ===================================================================== F4 ===
def figure4():
    """Hero: a. b excludes a non-tissue artefact, c bounds what is shared."""
    fig = plt.figure(figsize=(W, 58 * MM))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.16, 0.82, 0.92, 0.78], wspace=0.50)
    axA = fig.add_subplot(gs[0]); axL = fig.add_subplot(gs[1])
    axB = fig.add_subplot(gs[2]); axC = fig.add_subplot(gs[3])

    scores = [("mucinous", m11["mucinous_score"]["r2"]["K24_fractions"],
               m11["mucinous_score"]["r2"]["mean1024"], GREY),
              ("linear MSI", m11["msi_score"]["r2"]["K24_fractions"],
               m11["msi_score"]["r2"]["mean1024"], GREY),
              ("MIL, transfers", e2c["ridge_cv_r2_of_mil_logit"]["K24"],
               e2c["ridge_cv_r2_of_mil_logit"]["mean1024"], BLUE)]
    yy = np.arange(len(scores))[::-1]
    for j, (lab, r24, r1024, c) in zip(yy, scores):
        axA.plot([r24, r1024], [j, j], lw=0.8, color=GREY, zorder=1)
        axA.plot(r1024, j, "o", ms=4.0, mfc="white", mec=INK, mew=0.8, zorder=3)
        axA.plot(r24, j, "o", ms=4.0, mfc=c, mec=c, zorder=3)
        axA.text(r1024 + 0.05, j, f"{r24 / r1024:.0%} kept", fontsize=6.1, color=c,
                 ha="left", va="center", fontweight="bold" if c == BLUE else "normal")
    # a generic patient-level score keeps 62-69% of its full-representation R2;
    # mark where the transferring score would have landed under that floor
    mil_full = e2c["ridge_cv_r2_of_mil_logit"]["mean1024"]
    lo, hi = 0.62 * mil_full, 0.69 * mil_full
    axA.plot([lo, hi], [0, 0], lw=4.4, color="#C4C4C4", solid_capstyle="butt", zorder=1)
    for edge in (lo, hi):
        axA.plot([edge, edge], [-0.16, 0.16], lw=1.0, color=INK, zorder=4)
    axA.annotate("where a generic score\nwould land (62-69% kept)",
                 xy=((lo + hi) / 2, -0.14), xytext=((lo + hi) / 2, -0.62),
                 fontsize=5.8, color=INK, ha="center", va="top",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=INK))
    axA.set_yticks(yy); axA.set_yticklabels([s[0] for s in scores])
    axA.set_xlim(0.0, 1.46); axA.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axA.set_ylim(-1.15, len(scores) - 0.30)
    axA.set_xlabel("cross-validated $R^2$ of reconstruction", labelpad=1)
    clean(axA, left=False)
    axA.plot([], [], "o", ms=4.0, mfc="white", mec=INK, mew=0.8, label="1,024-d")
    axA.plot([], [], "o", ms=4.0, mfc=GREY, mec=GREY, label="24 states")
    axA.legend(loc="lower center", bbox_to_anchor=(0.46, 1.00), ncol=2,
               handletextpad=0.3, columnspacing=1.0, borderpad=0)


    # b: the compression ladder — how much survives each representation size
    lad2 = audit_full["E2b"]
    order = ["mean1024", "K128", "K64", "K24", "top1"]
    xs = range(len(order))
    for coh, col, lab in (("CRC", GREY, "within colorectal"),
                          ("STAD", "#B0B0B0", "within gastric")):
        axL.plot(xs, [lad2[k]["within_cv"][coh] for k in order], "-o", ms=2.8,
                 lw=0.9, color=col, mfc=col, mec=col)
    axL.plot(xs, [lad2[k]["transfer_S2C"]["auroc"] for k in order], "-o", ms=2.8,
             lw=1.1, color=RED, mfc=RED, mec=RED)
    axL.text(4.15, lad2["top1"]["within_cv"]["CRC"] + 0.012, "within-lineage",
             fontsize=5.6, color=GREY, va="bottom", ha="right")
    axL.text(4.15, lad2["top1"]["transfer_S2C"]["auroc"] - 0.012, "transfer",
             fontsize=5.6, color=RED, va="top", ha="right")
    axL.axhline(0.5, lw=0.6, ls=(0, (1, 2)), color=GREY, zorder=0)
    axL.set_xticks(list(xs))
    axL.set_xticklabels(["1,024-d", "128", "64", "24", "top-1"], fontsize=5.6,
                        rotation=30, ha="right")
    axL.set_ylim(0.44, 0.88)
    axL.set_ylabel("AUROC", labelpad=1)
    axL.set_xlabel("representation size", labelpad=1)
    clean(axL)

    att = e2c["attention_tissue_classes"]
    cls = ["TUM", "STR", "MUS", "DEB", "MUC", "NORM", "BACK", "ADI", "LYM"]
    cls_label = {"TUM": "tumour", "STR": "stroma", "MUS": "muscle", "DEB": "debris",
                 "MUC": "mucus", "NORM": "normal", "BACK": "background",
                 "ADI": "adipose", "LYM": "lymphocyte"}
    x = np.arange(len(cls)); w = 0.38
    axB.bar(x - w / 2, [att["high_attention_top5pct"][c] for c in cls], w,
            color=BLUE, label="top 5% attention")
    axB.bar(x + w / 2, [att["low_attention_bottom50pct"][c] for c in cls], w,
            color=PALE, label="bottom 50%")
    axB.text(0, att["high_attention_top5pct"]["TUM"] + 0.02, "0.51", ha="center",
             fontsize=6.2, color=BLUE, fontweight="bold")
    axB.set_xticks(x)
    axB.set_xticklabels([cls_label[c] for c in cls], fontsize=5.1, rotation=55, ha="right")
    axB.set_ylim(0, 0.60); axB.set_ylabel("mean class probability", labelpad=1)
    axB.set_xlabel("tissue class", labelpad=1)
    axB.legend(loc="upper right", bbox_to_anchor=(1.02, 1.04), fontsize=5.4,
               handlelength=0.9, handletextpad=0.35, borderpad=0.1)
    clean(axB)

    obs = audit["cross"]["cos_wSTAD_wCRC"]; flo = audit["cross"]["floor_2p5_97p5"]
    bars = [("CRC", audit["CRC"]["splithalf_cos_median"], GREY),
            ("gastric", audit["STAD"]["splithalf_cos_median"], GREY),
            ("cross-organ", obs, BLUE)]
    for i, (lab, v, c) in enumerate(bars):
        axC.bar(i, v, 0.58, color=c)
        axC.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=6.2, color=c)
    axC.axhspan(flo[0], flo[1], color="#E8E8E8", zorder=0)
    axC.set_xlim(-0.62, 3.30)
    axC.text(2.62, (flo[0] + flo[1]) / 2, "permutation\nfloor", fontsize=5.8, color=GREY,
             va="center", ha="left")
    axC.axhline(0, color=INK, lw=0.6)
    axC.set_xticks(range(3))
    axC.set_xticklabels(["colorectal", "gastric", "cross-organ"],
                        fontsize=5.6, rotation=30, ha="right")
    axC.text(0.5, 1.03, "within-lineage ceiling", fontsize=5.8, color=GREY, ha="center",
             va="bottom", transform=axC.get_xaxis_transform())
    axC.set_ylim(-0.25, 0.75); axC.set_ylabel("cosine of MSI directions", labelpad=1)
    clean(axC)
    for ax, ch in ((axA, "a"), (axL, "b"), (axB, "c"), (axC, "d")):
        fig.text(ax.get_position().x0 - 0.042, 0.94, ch, fontsize=9,
                 fontweight="bold", va="bottom", ha="left")
    save(fig, "Figure_4")
    print("Figure 4")


def _quad(t, r):
    """Corners of a text's rotated glyph box, in display coordinates."""
    box = t.get_window_extent(r)
    rot = t.get_rotation()
    cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
    if not rot:
        w, h = box.width, box.height
    else:
        t.set_rotation(0)
        flat = t.get_window_extent(t.figure.canvas.get_renderer())
        w, h = flat.width, flat.height
        t.set_rotation(rot)
    a = np.deg2rad(rot)
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[cx + dx * ca - dy * sa, cy + dx * sa + dy * ca]
                     for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2),
                                    (w / 2, h / 2), (-w / 2, h / 2))])


def _hits(p, q, pad=-0.6):
    """Separating-axis overlap test, shrunk by `pad` px so a touch is not a hit."""
    for poly in (p, q):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            axis = axis / (np.linalg.norm(axis) or 1)
            a, b = p @ axis, q @ axis
            if a.min() > b.max() + pad or b.min() > a.max() + pad:
                return False
    return True


def bbox_check(fig, name):
    """Report visible text that overlaps other text, or that leaves the canvas."""
    r = fig.canvas.get_renderer()
    skip = set()
    for ax in fig.axes:
        skip.add(id(ax.xaxis.label)); skip.add(id(ax.yaxis.label)); skip.add(id(ax.title))
        # matplotlib keeps tick labels for ticks outside the view limits; those
        # are never drawn, so comparing them invents collisions
        for axis, (v0, v1) in ((ax.xaxis, sorted(ax.get_xlim())),
                               (ax.yaxis, sorted(ax.get_ylim()))):
            for tick, loc in zip(axis.get_major_ticks(), axis.get_majorticklocs()):
                if not (v0 <= loc <= v1):
                    skip.add(id(tick.label1)); skip.add(id(tick.label2))
    texts = [t for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible() and id(t) not in skip]
    quads = [(t, _quad(t, r)) for t in texts]
    hits = [(a.get_text(), b.get_text())
            for i, (a, qa) in enumerate(quads) for b, qb in quads[i + 1:]
            if _hits(qa, qb)]
    # no canvas-bounds test: savefig runs with bbox="tight" and re-crops after
    if hits:
        print(f"  {name} OVERLAPS {len(hits)}")
        for a, b in hits[:12]:
            print(f"    {a[:40]!r} x {b[:40]!r}")
    else:
        print(f"  {name} bbox check clean")


for f in (figure1, figure2, figure3, figure4):
    f()
for k in (1, 2, 3, 4):
    (P / "overleaf_submission/figures" / f"Figure_{k}.pdf").write_bytes(
        (P / f"figures/Figure_{k}.pdf").read_bytes())
print("installed into overleaf_submission/figures")
