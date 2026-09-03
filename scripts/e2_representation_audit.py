#!/usr/bin/env python3
"""E2 representation audit (pre-registered): E2a direction-angle, E2b compression
ladder, E2d equivalence bounds. CPU only."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import roc_auc_score

ROOT = Path("/home/ubuntu/shared/proj-Gastrointestinal")
MANIFEST_DIR = ROOT / "Experiment/datasets/manifests/tasks_qc"
SF_PATH = ROOT / "Experiment/analysis/medical_morphology_atlas/gi_morphology_state_patient_fractions.tsv"
HIGHK = ROOT / "Experiment/analysis/medical_morphology_atlas_highk_sensitivity"
LAB = ROOT / "Experiment/datasets/labels"
OUT_DIR = ROOT / "Publication/paper_visibility_boundary"
STATE_COLS = [f"state_{i:02d}" for i in range(24)]
CV_SEED = 1234
PERM_SEED = 20260807
C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3]
N_SPLITHALF = 200
N_FLOOR = 1000
NPERM_LADDER = 1000
NBOOT_EQ = 10000
NPERM_EQ = 1000
NSIM_POWER = 500
MLH1_PROBES = ["cg21490561", "cg14751544", "cg23658326"]


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


def lr(C=1.0):
    return LogisticRegression(max_iter=4000, class_weight="balanced", C=C)


def main():
    slides = read_manifest()
    c64 = np.load(HIGHK / "K64/gi_morphology_state_centers.npy")
    c128 = np.load(HIGHK / "K128/gi_morphology_state_centers.npy")

    mean_acc, f64_acc, f128_acc = {}, {}, {}
    for i, r in enumerate(slides, 1):
        u = torch.load(ROOT / r["feature_path"], map_location="cpu")
        x = F.normalize(u["features"].float(), dim=1).numpy().astype(np.float32)
        key = (r["_cohort"], r["case_submitter_id"])
        e = mean_acc.setdefault(key, {"sum": np.zeros(x.shape[1]), "n": 0})
        e["sum"] += x.sum(0)
        e["n"] += len(x)
        a64 = ((x[:, None, :] - c64[None]) ** 2).sum(-1).argmin(1)
        a128 = ((x[:, None, :] - c128[None]) ** 2).sum(-1).argmin(1)
        f64_acc.setdefault(key, []).append(np.bincount(a64, minlength=64) / len(a64))
        f128_acc.setdefault(key, []).append(np.bincount(a128, minlength=128) / len(a128))
        if i % 200 == 0:
            print(f"loaded {i}/{len(slides)}", flush=True)

    sf = __import__("pandas").read_csv(SF_PATH, sep="\t")
    D = {}
    for coh_sf, coh in [("CRC_MSI", "CRC"), ("STAD_MSI", "STAD")]:
        sub = sf[sf.cohort == coh_sf].set_index("case_submitter_id")
        pats = [p for p in sub.index if (coh, p) in mean_acc]
        M = np.vstack([mean_acc[(coh, p)]["sum"] / mean_acc[(coh, p)]["n"] for p in pats])
        F64 = np.vstack([np.mean(f64_acc[(coh, p)], axis=0) for p in pats])
        F128 = np.vstack([np.mean(f128_acc[(coh, p)], axis=0) for p in pats])
        F24 = sub.loc[pats, STATE_COLS].values
        onehot = np.eye(24)[F24.argmax(1)]
        D[coh] = {"pats": pats, "mean1024": M, "K128": F128, "K64": F64,
                  "K24": F24, "top1": onehot,
                  "msi": (sub.loc[pats, "msi_status"] == "MSI-H").astype(int).values}
    results = {"meta": {"c_grid": C_GRID,
                        "cv_seed": CV_SEED, "perm_seed": PERM_SEED}}

    # ---------------- E2a ----------------
    rng = np.random.default_rng(PERM_SEED)

    def choose_C(X, y):
        best, bestC = -1, 1.0
        for C in C_GRID:
            skf = StratifiedKFold(5, shuffle=True, random_state=CV_SEED)
            a = cross_val_score(lr(C), X, y, cv=skf, scoring="roc_auc").mean()
            if a > best:
                best, bestC = a, C
        return bestC, round(float(best), 4)

    w = {}
    e2a = {}
    for coh in ["CRC", "STAD"]:
        X, y = D[coh]["mean1024"], D[coh]["msi"]
        C, cvauc = choose_C(X, y)
        m = lr(C).fit(X, y)
        w[coh] = m.coef_[0] / np.linalg.norm(m.coef_[0])
        e2a[coh] = {"chosen_C": C, "within_cv_auroc": cvauc,
                    "n": int(len(y)), "n_pos": int(y.sum())}
        print(f"E2a {coh}: C={C} cv={cvauc}", flush=True)

    for coh in ["CRC", "STAD"]:
        X, y = D[coh]["mean1024"], D[coh]["msi"]
        C = e2a[coh]["chosen_C"]
        cosines = []
        for _ in range(N_SPLITHALF):
            idx = rng.permutation(len(y))
            pos = idx[y[idx] == 1]
            neg = idx[y[idx] == 0]
            h1 = np.concatenate([pos[:len(pos)//2], neg[:len(neg)//2]])
            h2 = np.concatenate([pos[len(pos)//2:], neg[len(neg)//2:]])
            w1 = lr(C).fit(X[h1], y[h1]).coef_[0]
            w2 = lr(C).fit(X[h2], y[h2]).coef_[0]
            cosines.append(float(np.dot(w1, w2) /
                                 (np.linalg.norm(w1) * np.linalg.norm(w2))))
        e2a[coh]["splithalf_cos_median"] = round(float(np.median(cosines)), 4)
        e2a[coh]["splithalf_cos_iqr"] = [round(float(np.percentile(cosines, 25)), 4),
                                         round(float(np.percentile(cosines, 75)), 4)]
        print(f"E2a splithalf {coh}: {e2a[coh]['splithalf_cos_median']}", flush=True)

    obs_cos = float(np.dot(w["STAD"], w["CRC"]))
    floor = []
    Xs, ys = D["STAD"]["mean1024"], D["STAD"]["msi"]
    Cs = e2a["STAD"]["chosen_C"]
    for i in range(N_FLOOR):
        wp = lr(Cs).fit(Xs, rng.permutation(ys)).coef_[0]
        floor.append(float(np.dot(wp / np.linalg.norm(wp), w["CRC"])))
        if (i + 1) % 200 == 0:
            print(f"floor {i+1}/{N_FLOOR}", flush=True)
    floor = np.array(floor)
    ws_on_crc = roc_auc_score(D["CRC"]["msi"], D["CRC"]["mean1024"] @ w["STAD"])
    wc_on_stad = roc_auc_score(D["STAD"]["msi"], D["STAD"]["mean1024"] @ w["CRC"])
    e2a["cross"] = {"cos_wSTAD_wCRC": round(obs_cos, 4),
                    "floor_mean": round(float(floor.mean()), 4),
                    "floor_2p5_97p5": [round(float(np.percentile(floor, 2.5)), 4),
                                       round(float(np.percentile(floor, 97.5)), 4)],
                    "floor_p": round(float((np.abs(floor) >= abs(obs_cos)).mean()), 4),
                    "transfer_auroc_STAD_to_CRC": round(float(ws_on_crc), 4),
                    "transfer_auroc_CRC_to_STAD": round(float(wc_on_stad), 4)}
    results["E2a"] = e2a
    print("E2a cross:", json.dumps(e2a["cross"]), flush=True)

    # ---------------- E2b ----------------
    reps = ["mean1024", "K128", "K64", "K24", "top1"]
    e2b = {}
    for rep in reps:
        Xc, yc = D["CRC"][rep], D["CRC"]["msi"]
        Xs_, ys_ = D["STAD"][rep], D["STAD"]["msi"]
        skf = StratifiedKFold(5, shuffle=True, random_state=CV_SEED)
        cvc = roc_auc_score(yc, cross_val_predict(lr(), Xc, yc, cv=skf,
                                                  method="predict_proba")[:, 1])
        cvs = roc_auc_score(ys_, cross_val_predict(lr(), Xs_, ys_, cv=skf,
                                                   method="predict_proba")[:, 1])
        clf = lr().fit(Xs_, ys_)
        p = clf.predict_proba(Xc)[:, 1]
        obs = roc_auc_score(yc, p)
        nperm = NPERM_LADDER if rep != "mean1024" else 200
        prng = np.random.default_rng(PERM_SEED)
        null = np.array([roc_auc_score(yc, lr().fit(Xs_, prng.permutation(ys_))
                                       .predict_proba(Xc)[:, 1]) for _ in range(nperm)])
        e2b[rep] = {"dim": int(D["CRC"][rep].shape[1]),
                    "within_cv": {"CRC": round(float(cvc), 4), "STAD": round(float(cvs), 4)},
                    "transfer_S2C": {"auroc": round(float(obs), 4),
                                     "perm_p": round(float((null >= obs).mean()), 4),
                                     "nperm": nperm}}
        print("E2b", rep, json.dumps(e2b[rep]), flush=True)
    e2b["MIL_reference"] = {"note": "existing cancer-holdout benchmarks: ABMIL 0.758, "
                                    "TransMIL-lite 0.771 on held-out TCGA-STAD (trained on "
                                    "remaining GI cohorts); cited, not recomputed"}
    results["E2b"] = e2b

    # ---------------- E2d ----------------
    def mlh1_status(coh_m, patients):
        m = __import__("pandas").read_parquet(
            ROOT / f"Experiment/datasets/methylation/{coh_m}/methylation_matrix_full.parquet",
            columns=MLH1_PROBES)
        beta = m.mean(axis=1)
        return {p: (1 if beta.get(p, np.nan) > 0.20 else 0)
                for p in patients if p in beta.index}

    def pheno_frames(coh, coh_m, extra_csv):
        pats = D[coh]["pats"]
        X = D[coh]["K24"]
        out = {"MSI-H": (X, D[coh]["msi"])}
        ml = mlh1_status(coh_m, pats)
        idx = [i for i, p in enumerate(pats) if p in ml]
        out["MLH1-hyper"] = (X[idx], np.array([ml[pats[i]] for i in idx]))
        ex = __import__("pandas").read_csv(extra_csv).set_index("case_submitter_id")
        for col, name in [("KRAS_mut", "KRAS"), ("hypermutated", "TMB-high")]:
            idx = [i for i, p in enumerate(pats)
                   if p in ex.index and not __import__("pandas").isna(ex.loc[p, col])]
            y = ex.loc[[pats[i] for i in idx], col].astype(int).values
            out[name] = (X[idx], y)
        return out

    crc_ph = pheno_frames("CRC", "CRC", LAB / "crc_extra_phenotypes_cbioportal.csv")
    stad_ph = pheno_frames("STAD", "STAD", LAB / "stad_extra_phenotypes_cbioportal.csv")
    e2d = {}
    for pheno in ["MSI-H", "MLH1-hyper", "TMB-high", "KRAS"]:
        for src, tgt, fs, ft in [("STAD", "CRC", stad_ph, crc_ph),
                                 ("CRC", "STAD", crc_ph, stad_ph)]:
            Xtr, ytr = fs[pheno]
            Xte, yte = ft[pheno]
            clf = lr().fit(Xtr, ytr)
            p = clf.predict_proba(Xte)[:, 1]
            obs = roc_auc_score(yte, p)
            brng = np.random.default_rng(PERM_SEED)
            boots = []
            for _ in range(NBOOT_EQ):
                idx = brng.integers(0, len(yte), len(yte))
                if len(np.unique(yte[idx])) < 2:
                    continue
                boots.append(roc_auc_score(yte[idx], p[idx]))
            ub95 = float(np.percentile(boots, 95))
            null = np.array([roc_auc_score(yte, lr().fit(Xtr, brng.permutation(ytr))
                                           .predict_proba(Xte)[:, 1])
                             for _ in range(NPERM_EQ)])
            crit = float(np.percentile(null, 95))
            npos, nneg = int(yte.sum()), int(len(yte) - yte.sum())
            delta = np.sqrt(2) * __import__("scipy.stats", fromlist=["norm"]).norm.ppf(0.65)
            power_hits = 0
            for _ in range(NSIM_POWER):
                spos = brng.normal(delta, 1, npos)
                sneg = brng.normal(0, 1, nneg)
                a = roc_auc_score(np.r_[np.ones(npos), np.zeros(nneg)], np.r_[spos, sneg])
                power_hits += a > crit
            e2d[f"{pheno}::{src}_to_{tgt}"] = {
                "auroc": round(float(obs), 4),
                "upper95_onesided": round(ub95, 4),
                "excludes_0.62": bool(ub95 < 0.62),
                "null_crit95": round(crit, 4),
                "power_detect_0.65": round(power_hits / NSIM_POWER, 3),
                "n_test": len(yte), "n_test_pos": npos}
            print("E2d", pheno, src, "->", tgt,
                  json.dumps(e2d[f"{pheno}::{src}_to_{tgt}"]), flush=True)
    results["E2d"] = e2d

    (OUT_DIR / "e2_representation_audit_manifest.json").write_text(
        json.dumps(results, indent=2))
    print("wrote manifest", flush=True)


if __name__ == "__main__":
    main()
