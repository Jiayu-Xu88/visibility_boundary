"""
Option A: methylation-supervised auxiliary task for MSI classification.

Pre-registered design (decided 2026-05-13, before any runs):
  * Task: patient-level MSI binary classification (MSI-H vs MSS/MSI-L)
  * Backbone: gated ABMIL over UNI 1024-d patch features
  * Heads:
      msi_head      : linear -> sigmoid (BCE loss)
      meth_head     : linear -> K-d (MSE on logit-beta, K=500 top-variable CpGs)
  * Total loss: L = L_msi + lambda * L_meth
      lambda in {0.0, 0.1, 0.5, 1.0}
      lambda=0.0 is the pre-registered baseline B_msi
      (NB: when lambda=0 we still construct the meth head, we just zero its
       gradient via lambda=0.0; this keeps initialization identical, removing
       a known nuisance variable.)
  * Cohort: STAD fold-0 and CRC fold-0, run independently
  * Splits: re-use fold0_{train,val}_patient_ids.txt (STAD) and
            crc_fold0_{train,val}_patient_ids.txt (CRC)
  * Metric: patient-level AUROC on val
  * Pre-registered gate: paired patient-level bootstrap on
      delta_AUC = AUC(lambda) - AUC(lambda=0)
      PASS iff observed delta >= +0.02 AND CI lower > 0 (one-sided)
  * Seed: 1234 fixed across all runs; no seed cherry-pick.

Each lambda * cohort run produces:
  <out>/option_a_<cohort>_lambda<lambda>_summary.json
  <out>/option_a_<cohort>_lambda<lambda>_predictions_val.parquet
      columns: patient_id, y_true (0/1 MSI-H), y_score, lambda, cohort
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]


# -----------------------------------------------------------------------------
# Model: gated ABMIL backbone + MSI head + (optional) methylation head
# -----------------------------------------------------------------------------

class GatedABMIL(nn.Module):
    def __init__(self, feat_dim=1024, attn_dim=256):
        super().__init__()
        self.U = nn.Linear(feat_dim, attn_dim)
        self.V = nn.Linear(feat_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, feats: torch.Tensor):  # feats: (n, d)
        u = torch.tanh(self.U(feats))
        v = torch.sigmoid(self.V(feats))
        a = self.w(u * v).squeeze(-1)             # (n,)
        a = torch.softmax(a, dim=0)
        z = (a.unsqueeze(-1) * feats).sum(0)      # (d,)
        return z, a


class OptionAModel(nn.Module):
    def __init__(self, feat_dim=1024, attn_dim=256, n_meth=500):
        super().__init__()
        self.pool = GatedABMIL(feat_dim, attn_dim)
        self.msi_head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(),
            nn.Dropout(0.25), nn.Linear(256, 1),
        )
        self.meth_head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(),
            nn.Dropout(0.25), nn.Linear(256, n_meth),
        )

    def forward(self, feats):
        z, a = self.pool(feats)                   # (d,), (n,)
        logit_msi = self.msi_head(z).squeeze(-1)  # scalar
        pred_meth = self.meth_head(z)             # (K,)
        return logit_msi, pred_meth, a


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def load_msi_labels(csv_path: Path) -> dict[str, int]:
    df = pd.read_csv(csv_path)
    pid_col = "case_submitter_id"
    status_col = "msi_status"
    out = {}
    for r in df.itertuples(index=False):
        pid = getattr(r, pid_col)
        s = getattr(r, status_col)
        if pd.isna(s):
            continue
        s = str(s).strip().upper()
        if s in ("MSI-H", "MSI_H", "MSIH"):
            out[pid] = 1
        elif s in ("MSS", "MSI-L", "MSI_L", "MSIL"):
            out[pid] = 0
    return out


def load_manifest_rows(tsv_path: Path) -> list[dict]:
    rows = []
    with tsv_path.open() as f:
        for r in csv.DictReader(f, delimiter='\t'):
            rows.append(r)
    return rows


def build_items(manifest_rows, train_ids: set, val_ids: set,
                features_root: Path, msi_map: dict[str, int],
                meth_targets: dict[str, np.ndarray]):
    """Return list of dicts with slide-level data references + patient labels."""
    out = []
    for r in manifest_rows:
        pid = r["case_submitter_id"]
        if pid not in msi_map:
            continue
        if pid in train_ids:
            split = "train"
        elif pid in val_ids:
            split = "val"
        else:
            continue
        uuid = Path(r["feature_path"]).stem
        feat_p = features_root / f"{uuid}.pt"
        if not feat_p.exists():
            continue
        out.append({
            "patient_id": pid,
            "slide_uuid": uuid,
            "feat_path": feat_p,
            "y_msi": int(msi_map[pid]),
            "has_meth": pid in meth_targets,
            "split": split,
        })
    return out


class SlideDataset(torch.utils.data.Dataset):
    def __init__(self, items, meth_targets):
        self.items = items
        self.meth_targets = meth_targets

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        f = torch.load(it["feat_path"], map_location="cpu", weights_only=False)
        feats = f["features"].float()                            # (n, 1024)
        if it["has_meth"]:
            y_meth = torch.from_numpy(self.meth_targets[it["patient_id"]]).float()
            has_meth = True
        else:
            y_meth = torch.zeros(self.meth_K, dtype=torch.float32)
            has_meth = False
        return {
            "feats": feats,
            "y_msi": torch.tensor(it["y_msi"], dtype=torch.float32),
            "y_meth": y_meth,
            "has_meth": torch.tensor(has_meth, dtype=torch.bool),
            "patient_id": it["patient_id"],
            "slide_uuid": it["slide_uuid"],
        }


# -----------------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, lambda_meth, train: bool, grad_accum=4):
    model.train(train)
    sum_loss = sum_msi = sum_meth = 0.0
    n = 0
    per_slide_scores = []   # list of (pid, slide_uuid, y_msi, score)
    if train:
        optimizer.zero_grad(set_to_none=True)
    step = 0
    for batch in loader:
        feats = batch["feats"][0].to(device)
        y_msi = batch["y_msi"][0].to(device)
        y_meth = batch["y_meth"][0].to(device)
        has_meth = bool(batch["has_meth"][0].item())

        with torch.set_grad_enabled(train):
            logit_msi, pred_meth, a = model(feats)
            L_msi = F.binary_cross_entropy_with_logits(logit_msi, y_msi)
            if has_meth and lambda_meth > 0:
                L_meth = F.mse_loss(pred_meth, y_meth)
                loss = L_msi + lambda_meth * L_meth
            else:
                L_meth = torch.tensor(0.0, device=device)
                loss = L_msi
            if train:
                (loss / grad_accum).backward()
                step += 1
                if step % grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        sum_loss += float(loss.detach())
        sum_msi  += float(L_msi.detach())
        sum_meth += float(L_meth.detach())
        n += 1
        per_slide_scores.append({
            "patient_id": batch["patient_id"][0],
            "slide_uuid": batch["slide_uuid"][0],
            "y_true": int(batch["y_msi"][0].item()),
            "logit": float(logit_msi.detach().cpu().item()),
        })
    if train and step % grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {
        "loss": sum_loss / max(n, 1),
        "msi_bce": sum_msi / max(n, 1),
        "meth_mse": sum_meth / max(n, 1),
        "n_slides": n,
    }, per_slide_scores


def aggregate_patient_scores(per_slide: list[dict]) -> pd.DataFrame:
    """Mean of slide logits within patient -> patient-level score (sigmoid afterwards)."""
    by_pid = {}
    y_by_pid = {}
    for d in per_slide:
        by_pid.setdefault(d["patient_id"], []).append(d["logit"])
        y_by_pid[d["patient_id"]] = d["y_true"]
    rows = []
    for pid, logits in by_pid.items():
        mean_logit = float(np.mean(logits))
        score = 1.0 / (1.0 + np.exp(-mean_logit))
        rows.append({"patient_id": pid, "y_true": y_by_pid[pid],
                     "logit": mean_logit, "y_score": score})
    return pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)


def patient_auc(df: pd.DataFrame) -> float:
    if df["y_true"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(df["y_true"], df["y_score"]))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uni-features-root", type=Path, required=True)
    ap.add_argument("--slide-manifest",   type=Path, required=True)
    ap.add_argument("--targets",          type=Path, required=True,
                    help="targets_logit_beta.npz (top-K=500)")
    ap.add_argument("--fold0-train-ids",  type=Path, required=True)
    ap.add_argument("--fold0-val-ids",    type=Path, required=True)
    ap.add_argument("--msi-labels-csv",   type=Path, required=True)
    ap.add_argument("--cohort",           type=str, required=True,
                    choices=["STAD", "CRC"])
    ap.add_argument("--lambda-meth",      type=float, required=True)
    ap.add_argument("--out-dir",          type=Path, required=True)
    ap.add_argument("--epochs",           type=int, default=40)
    ap.add_argument("--lr",               type=float, default=1e-4)
    ap.add_argument("--weight-decay",     type=float, default=1e-4)
    ap.add_argument("--grad-accum",       type=int, default=4)
    ap.add_argument("--patience",         type=int, default=8)
    ap.add_argument("--seed",             type=int, default=1234)
    ap.add_argument("--feat-dim",         type=int, default=1024)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ids = set(x.strip() for x in args.fold0_train_ids.read_text().splitlines() if x.strip())
    val_ids   = set(x.strip() for x in args.fold0_val_ids.read_text().splitlines() if x.strip())
    msi_map   = load_msi_labels(args.msi_labels_csv)

    targets_npz = np.load(args.targets)
    targets = {k: targets_npz[k] for k in targets_npz.files}
    K = next(iter(targets.values())).shape[0]

    rows = load_manifest_rows(args.slide_manifest)
    items = build_items(rows, train_ids, val_ids,
                         args.uni_features_root, msi_map, targets)
    train_items = [it for it in items if it["split"] == "train"]
    val_items   = [it for it in items if it["split"] == "val"]
    train_pids  = sorted({it["patient_id"] for it in train_items})
    val_pids    = sorted({it["patient_id"] for it in val_items})
    n_train_pos = sum(1 for it in train_items if it["y_msi"]==1)
    n_val_pos   = sum(1 for p in val_pids if msi_map[p]==1)
    n_train_meth = sum(1 for p in train_pids if p in targets)
    n_val_meth   = sum(1 for p in val_pids   if p in targets)
    print(f"[i] cohort={args.cohort}  device={device}  K={K}  lambda={args.lambda_meth}")
    print(f"[i] train slides={len(train_items)}  patients={len(train_pids)}  MSI-H slides={n_train_pos}  meth-covered patients={n_train_meth}/{len(train_pids)}")
    print(f"[i]  val  slides={len(val_items)}  patients={len(val_pids)}  MSI-H patients={n_val_pos}  meth-covered={n_val_meth}/{len(val_pids)}")

    train_ds = SlideDataset(train_items, targets)
    train_ds.meth_K = K
    val_ds   = SlideDataset(val_items,   targets)
    val_ds.meth_K = K
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True,
                                               num_workers=2, persistent_workers=True)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=1, shuffle=False,
                                               num_workers=2, persistent_workers=True)

    model = OptionAModel(feat_dim=args.feat_dim, n_meth=K).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[i] params={n_params:,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    tag = f"option_a_{args.cohort}_lambda{args.lambda_meth:.2f}"
    log_path = args.out_dir / f"{tag}_train_log.tsv"
    log_f = log_path.open("w")
    log_f.write("epoch\ttrain_loss\ttrain_msi\ttrain_meth\tval_loss\tval_msi\tval_meth\tval_auc\ttime_sec\n")

    best_auc = -1.0
    best_epoch = -1
    best_val_pred = None
    patience_left = args.patience

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr, _ = run_epoch(model, train_loader, optimizer, device,
                          args.lambda_meth, train=True, grad_accum=args.grad_accum)
        with torch.no_grad():
            va, vps = run_epoch(model, val_loader, optimizer, device,
                                 args.lambda_meth, train=False)
        df_val = aggregate_patient_scores(vps)
        auc = patient_auc(df_val)
        dt = time.time() - t0
        log_f.write(f"{ep}\t{tr['loss']:.4f}\t{tr['msi_bce']:.4f}\t{tr['meth_mse']:.4f}\t"
                    f"{va['loss']:.4f}\t{va['msi_bce']:.4f}\t{va['meth_mse']:.4f}\t"
                    f"{auc:.4f}\t{dt:.1f}\n"); log_f.flush()
        improved = auc > best_auc + 1e-6
        if improved:
            torch.save(model.state_dict(), args.out_dir / f"{tag}_best.pt")
            best_auc, best_epoch = auc, ep
            best_val_pred = df_val.copy()
            best_val_pred["lambda"] = args.lambda_meth
            best_val_pred["cohort"] = args.cohort
            patience_left = args.patience
        else:
            patience_left -= 1
        print(f"  ep {ep:3d}  tr_loss={tr['loss']:.4f}  val_loss={va['loss']:.4f}  "
              f"val_auc={auc:.4f}  ({dt:.1f}s){'  *' if improved else ''}")
        if patience_left <= 0:
            print(f"[i] early stop @ ep {ep};  best ep={best_epoch}  best_auc={best_auc:.4f}")
            break
    log_f.close()

    # Save artifacts
    summary = {
        "cohort": args.cohort,
        "lambda_meth": args.lambda_meth,
        "best_epoch": best_epoch,
        "best_val_auc": best_auc,
        "K": K,
        "n_train_slides": len(train_items),
        "n_train_patients": len(train_pids),
        "n_train_msi_h_slides": n_train_pos,
        "n_val_patients": len(val_pids),
        "n_val_msi_h": n_val_pos,
        "n_train_meth_covered": n_train_meth,
        "n_val_meth_covered": n_val_meth,
        "seed": args.seed,
        "epochs_run": ep,
    }
    (args.out_dir / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    best_val_pred.to_parquet(args.out_dir / f"{tag}_predictions_val.parquet")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
