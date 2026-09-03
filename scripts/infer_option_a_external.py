"""
External inference for Option A (STAD-trained MSI head) on TCGA-COAD and TCGA-READ.

For each (seed, lambda) checkpoint produced by train_option_a_msi_aux.py:
  1. Build OptionAModel with same hyperparams (feat_dim=1024, n_meth=500).
  2. Load checkpoint state_dict.
  3. Iterate manifest rows: read patient_id + file_id, load UNI .pt features,
     forward through model, get patient-level logit (mean of slide logits within
     patient). y_score = sigmoid(mean_logit).
  4. Write predictions parquet: patient_id, y_true, logit, y_score, lambda,
     seed, cohort.

Inputs (per cohort):
  --manifest <Experiment/datasets/manifests/external/<cohort>_msi_manifest.tsv>
  --features-root <Experiment/datasets/features/uni_external/<cohort>/uni>
  --ckpt-root <Experiment/core_code/checkpoints/option_a_external_ckpt>
  --seeds 7 1234 2025 4242 31337
  --lambdas 0.0 0.5
  --out-dir <Experiment/results/external_validation/option_a/<cohort>>

Outputs:
  <out-dir>/seed_<S>/option_a_<cohort>_lambda<L>_predictions_external.parquet
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Experiment" / "core_code" / "src"))

from train_option_a_msi_aux import OptionAModel  # type: ignore


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f, delimiter='\t'):
            rows.append(r)
    return rows


def msi_binary(status: str) -> int:
    s = status.strip().upper()
    if s in ("MSI-H", "MSI_H", "MSIH"):
        return 1
    if s in ("MSS", "MSI-L", "MSI_L", "MSIL"):
        return 0
    raise ValueError(f"unknown MSI status: {status!r}")


def infer_one_ckpt(ckpt_path: Path, manifest_rows: list[dict],
                   features_root: Path, device: str) -> pd.DataFrame:
    model = OptionAModel(feat_dim=1024, n_meth=500).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    by_pid: dict[str, list[float]] = {}
    y_by_pid: dict[str, int] = {}
    n_missing = 0
    with torch.no_grad():
        for r in manifest_rows:
            pid = r["patient_id"]
            fid = r["file_id"]
            y = msi_binary(r["msi_status"])
            feat_p = features_root / f"{fid}.pt"
            if not feat_p.exists():
                n_missing += 1
                continue
            payload = torch.load(feat_p, map_location="cpu", weights_only=False)
            feats = payload["features"].float().to(device)
            logit, _meth, _a = model(feats)
            by_pid.setdefault(pid, []).append(float(logit.item()))
            y_by_pid[pid] = y
    rows = []
    for pid, logits in by_pid.items():
        mean_logit = float(np.mean(logits))
        score = 1.0 / (1.0 + np.exp(-mean_logit))
        rows.append({"patient_id": pid, "y_true": y_by_pid[pid],
                     "logit": mean_logit, "y_score": score,
                     "n_slides": len(logits)})
    df = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
    return df, n_missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--features-root", type=Path, required=True)
    ap.add_argument("--ckpt-root", type=Path, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--lambdas", nargs="+", type=float, required=True)
    ap.add_argument("--cohort", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = load_manifest(args.manifest)
    print(f"[i] cohort={args.cohort} manifest_rows={len(manifest_rows)} device={device}")

    for seed in args.seeds:
        seed_out = args.out_dir / f"seed_{seed}"
        seed_out.mkdir(parents=True, exist_ok=True)
        for lam in args.lambdas:
            tag = f"option_a_{args.cohort}_lambda{lam:.2f}"
            ckpt = args.ckpt_root / f"seed_{seed}" / f"option_a_STAD_lambda{lam:.2f}_best.pt"
            if not ckpt.exists():
                print(f"  [skip] missing ckpt: {ckpt}")
                continue
            df, n_missing = infer_one_ckpt(ckpt, manifest_rows, args.features_root, device)
            df["lambda"] = lam
            df["seed"] = seed
            df["cohort"] = args.cohort
            out_path = seed_out / f"{tag}_predictions_external.parquet"
            df.to_parquet(out_path)
            try:
                from sklearn.metrics import roc_auc_score
                auc = float(roc_auc_score(df["y_true"], df["y_score"])) if df["y_true"].nunique() == 2 else float("nan")
            except Exception:
                auc = float("nan")
            print(f"  seed={seed} lam={lam:.2f}  patients={len(df)}  pos={int(df['y_true'].sum())}  auc={auc:.4f}  miss={n_missing}  -> {out_path.name}")


if __name__ == "__main__":
    sys.exit(main())
