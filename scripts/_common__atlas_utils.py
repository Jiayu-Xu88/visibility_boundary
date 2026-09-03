import os
"""Shared utilities for v4 algorithm-improvement experiments.

Mirrors the original atlas recipe in core_code/pipeline/run_gi_molecular_morphology_atlas.py:
  - deterministic MD5-seeded per-slide patch sampling
  - MiniBatchKMeans K=24
  - per-slide state fraction = bincount(assign)/n ; per-patient = mean over slides
All read-only against existing caches; writes only under v4_algo_improvements/.
"""
import hashlib
import numpy as np
import pandas as pd
import torch
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "."))
ATLAS_DIR = PROJECT_ROOT / "Experiment/analysis/medical_morphology_atlas"
SLIDE_FRAC = ATLAS_DIR / "gi_morphology_state_slide_fractions.tsv"
PATIENT_FRAC = ATLAS_DIR / "gi_morphology_state_patient_fractions.tsv"
CENTERS = ATLAS_DIR / "gi_morphology_state_centers.npy"

PATCHES_FIT = 96
PATCHES_ASSIGN = 512
N_STATES = 24
SEED = 20260519


def stable_indices(key: str, n: int, k: int) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    h = hashlib.md5(key.encode()).hexdigest()
    rng = np.random.default_rng(int(h[:16], 16))
    return np.sort(rng.choice(n, size=k, replace=False))


def load_features(feature_path, encoder="uni"):
    """Load [N,D] float32 features for a slide. encoder in {uni,conch}."""
    p = PROJECT_ROOT / feature_path if not str(feature_path).startswith("/") else Path(feature_path)
    if encoder != "uni":
        # swap the encoder subdir: .../features/uni/<id>.pt -> .../features/<enc>/<id>.pt
        p = Path(str(p).replace("/features/uni/", f"/features/{encoder}/"))
    obj = torch.load(p, map_location="cpu")
    x = obj["features"]
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return x.float().numpy()


def slide_manifest_for_cohort():
    """Return DataFrame: cohort,case_submitter_id,file_id,label,feature_path (uni)."""
    sf = pd.read_csv(SLIDE_FRAC, sep="\t")
    # map file_id -> uni feature path
    sf["feature_path"] = sf["file_id"].apply(
        lambda fid: f"Experiment/datasets/features/uni/{fid}.pt")
    return sf[["cohort", "case_submitter_id", "file_id", "label", "msi_status",
               "molecular_subtype", "feature_path"]].copy()


def sample_fit_matrix(rows, encoder="uni"):
    mats = []
    for r in rows:
        x = load_features(r["feature_path"], encoder)
        idx = stable_indices(r["file_id"] + ":fit", x.shape[0], PATCHES_FIT)
        mats.append(x[idx])
    return np.concatenate(mats, axis=0).astype(np.float32)


def assign_fractions(rows, centers, encoder="uni"):
    """Per-slide state fraction via argmin to centers. Returns DataFrame."""
    c2 = 0.5 * (centers * centers).sum(axis=1)
    recs = []
    for r in rows:
        x = load_features(r["feature_path"], encoder)
        idx = stable_indices(r["file_id"] + ":assign", x.shape[0], PATCHES_ASSIGN)
        xs = x[idx]
        a = (xs @ centers.T - c2[None, :]).argmax(axis=1)
        frac = np.bincount(a, minlength=centers.shape[0]).astype(float) / len(a)
        rec = dict(cohort=r["cohort"], case_submitter_id=r["case_submitter_id"],
                   file_id=r["file_id"], label=r["label"])
        for s in range(centers.shape[0]):
            rec[f"state_{s:02d}"] = frac[s]
        recs.append(rec)
    return pd.DataFrame(recs)


def to_patient(df_slide, n_states):
    cols = [f"state_{s:02d}" for s in range(n_states)]
    g = df_slide.groupby(["cohort", "case_submitter_id", "label"])[cols].mean().reset_index()
    return g


def cliffs_delta(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    gt = 0; lt = 0
    # vectorized
    for v in pos:
        gt += np.sum(v > neg); lt += np.sum(v < neg)
    return (gt - lt) / (len(pos) * len(neg))


def fit_kmeans(fit_x, k=N_STATES, seed=SEED):
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096,
                         n_init=10, max_iter=200, reassignment_ratio=0.01)
    km.fit(fit_x)
    return km.cluster_centers_.astype(np.float32)
