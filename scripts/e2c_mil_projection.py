#!/usr/bin/env python3
"""E2c: interpretability projection of the CRC-trained MIL transfer signal
(pre-registered). (1) CRC-trained TransMIL-lite (UNI, 5 folds) scores every STAD
patient; report the cross-organ AUROC of the fold-mean score. (2) Ridge-regress
the patient-level scores onto K=24 / K=64 / K=128 state fractions and the 1024-d
patient mean vector (5-fold CV R^2): if R^2(24) is low while R^2(1024) is high,
the transferable MIL signal lives outside the state basis. (3) CRC-trained ABMIL
attention on STAD: compare the NCT tissue-class distribution of high-attention
(top 5% within bag) versus low-attention (bottom 50%) patches. CPU only."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_auc_score

ROOT = Path("/home/ubuntu/shared/proj-Gastrointestinal")
CORE = ROOT / "Experiment/core_code"
MANIFEST = ROOT / "Experiment/datasets/manifests/tasks_qc/stad_msi_qc_feature_complete_slide_manifest.tsv"
SF_PATH = ROOT / "Experiment/analysis/medical_morphology_atlas/gi_morphology_state_patient_fractions.tsv"
K24_C = ROOT / "Experiment/analysis/medical_morphology_atlas/gi_morphology_state_centers.npy"
K64_C = ROOT / "Experiment/analysis/medical_morphology_atlas_highk_sensitivity/K64/gi_morphology_state_centers.npy"
K128_C = ROOT / "Experiment/analysis/medical_morphology_atlas_highk_sensitivity/K128/gi_morphology_state_centers.npy"
PROBE = ROOT / "Publication/paper_visibility_boundary/nct_uni_tissue_probe.npz"
OUT = ROOT / "Publication/paper_visibility_boundary/e2c_mil_projection_manifest.json"
sys.path.insert(0, str(CORE / "src"))

torch.set_num_threads(8)


def main():
    from mil_models import TransMILLite, ABMIL

    def load_ckpts(prefix, cls):
        models = []
        for f in range(5):
            sd = torch.load(CORE / f"checkpoints/{prefix}_fold{f}.pth", map_location="cpu")
            cfg = sd.get("config", {})
            kw = {}
            for k in ["feature_dim", "num_classes", "hidden_dim"]:
                if k in cfg:
                    kw[k] = cfg[k]
            kw.setdefault("feature_dim", 1024)
            kw.setdefault("num_classes", 2)
            m = cls(**kw)
            m.load_state_dict(sd["model"])
            m.eval()
            models.append(m)
        return models

    transmil = load_ckpts("crc_msi_qc_transmil_lite_uni", TransMILLite)
    abmil = load_ckpts("crc_msi_qc_abmil_uni", ABMIL)
    print("checkpoints loaded", flush=True)

    probe = np.load(PROBE, allow_pickle=True)
    Wp = probe["coef"].astype(np.float32)
    bp = probe["intercept"].astype(np.float32)
    classes = [str(c) for c in probe["classes"]]

    rows = []
    with MANIFEST.open(newline="") as h:
        for r in csv.DictReader(h, delimiter="\t"):
            if r.get("feature_path"):
                rows.append(r)
    by_pat = {}
    for r in rows:
        by_pat.setdefault(r["case_submitter_id"], []).append(r)

    c24, c64, c128 = np.load(K24_C), np.load(K64_C), np.load(K128_C)
    sf = __import__("pandas").read_csv(SF_PATH, sep="\t")
    sub = sf[sf.cohort == "STAD_MSI"].set_index("case_submitter_id")

    pats, scores, reps24, reps64, reps128, means, msis = [], [], [], [], [], [], []
    hi_cls = np.zeros(len(classes))
    lo_cls = np.zeros(len(classes))
    hi_n = lo_n = 0
    for i, (pat, rs) in enumerate(sorted(by_pat.items()), 1):
        if pat not in sub.index:
            continue
        raw_list, norm_list = [], []
        for r in rs:
            u = torch.load(ROOT / r["feature_path"], map_location="cpu")
            raw_list.append(u["features"].float())
            norm_list.append(F.normalize(u["features"].float(), dim=1))
        raw = torch.cat(raw_list)
        norm = torch.cat(norm_list)
        with torch.no_grad():
            probs = []
            for m in transmil:
                logits = m(raw)
                probs.append(torch.softmax(logits.squeeze(0), -1)[..., 1].item()
                             if logits.numel() == 2 else float(torch.sigmoid(logits).view(-1)[0]))
            score = float(np.mean(probs))
            att_all = []
            for m in abmil:
                att, _ = m.attention_scores(raw)
                att_all.append(att.view(-1).numpy())
            att = np.mean(att_all, axis=0)
        xn = norm.numpy().astype(np.float32)
        pcls = torch.softmax(torch.from_numpy(xn @ Wp.T + bp), dim=1).numpy()
        k_hi = max(1, int(0.05 * len(att)))
        order = np.argsort(att)
        hi_idx = order[-k_hi:]
        lo_idx = order[: len(att) // 2]
        hi_cls += pcls[hi_idx].sum(0); hi_n += len(hi_idx)
        lo_cls += pcls[lo_idx].sum(0); lo_n += len(lo_idx)
        pats.append(pat)
        scores.append(score)
        a24 = ((xn[:, None, :] - c24[None]) ** 2).sum(-1).argmin(1)
        a64 = ((xn[:, None, :] - c64[None]) ** 2).sum(-1).argmin(1)
        a128 = ((xn[:, None, :] - c128[None]) ** 2).sum(-1).argmin(1)
        reps24.append(np.bincount(a24, minlength=24) / len(a24))
        reps64.append(np.bincount(a64, minlength=64) / len(a64))
        reps128.append(np.bincount(a128, minlength=128) / len(a128))
        means.append(xn.mean(0))
        msis.append(int(sub.loc[pat, "msi_status"] == "MSI-H"))
        if i % 50 == 0:
            print(f"patients {i}", flush=True)

    y = np.array(msis)
    s = np.array(scores)
    eps = 1e-6
    logit = np.log(np.clip(s, eps, 1 - eps) / np.clip(1 - s, eps, 1 - eps))
    auroc = roc_auc_score(y, s)

    def cv_r2(X):
        kf = KFold(5, shuffle=True, random_state=1234)
        model = RidgeCV(alphas=np.logspace(-3, 3, 13))
        return round(float(np.mean(cross_val_score(model, X, logit, cv=kf, scoring="r2"))), 4)

    res = {"meta": {
                    "models": "CRC-trained TransMIL-lite UNI (5 folds, fold-mean prob); "
                              "ABMIL UNI (5 folds, mean attention)",
                    "n_stad_patients": int(len(y)), "n_msi_h": int(y.sum())},
           "crc_trained_mil_on_stad_auroc": round(float(auroc), 4),
           "ridge_cv_r2_of_mil_logit": {"K24": cv_r2(np.vstack(reps24)),
                                        "K64": cv_r2(np.vstack(reps64)),
                                        "K128": cv_r2(np.vstack(reps128)),
                                        "mean1024": cv_r2(np.vstack(means))},
           "attention_tissue_classes": {
               "high_attention_top5pct": {c: round(float(hi_cls[i] / hi_n), 4)
                                          for i, c in enumerate(classes)},
               "low_attention_bottom50pct": {c: round(float(lo_cls[i] / lo_n), 4)
                                             for i, c in enumerate(classes)}}}
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
