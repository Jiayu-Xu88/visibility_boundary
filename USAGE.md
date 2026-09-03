# Usage manual

Step-by-step guide to run the code. The repository ships **code + small artifacts**; the large
cached features and raw whole-slide images are **not distributed here** — their locations are
recorded in [`DATA.md`](DATA.md). With the cached features downloaded, every number, figure, and
table in the paper can be regenerated without the raw images.

---

## 0. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Requires a CUDA GPU for feature extraction and MIL training (CPU is fine for the atlas
statistics and the v4 analyses if features are already cached). Tested with Python 3.12,
PyTorch 2.11 (CUDA 12.8).

## 1. Point the code at your data

The code resolves paths relative to a project root. Set it once:

```bash
export PROJECT_ROOT=/path/to/your/workdir
```

Lay the downloaded data out under `$PROJECT_ROOT` as recorded in `DATA.md`, e.g.
`datasets/features/uni/<file_id>.pt`, `datasets/labels/`, `datasets/manifests/`.
(A few scripts still contain an absolute path in their header constant — set `PROJECT_ROOT`
or edit that line; see "Known issues" below.)

## 1b. What reproduces at each data tier

| Tier | Needs | What runs |
|---|---|---|
| **Clone only** | nothing extra | All patient-level statistics from the shipped state-fraction tables and frozen centroids: the visibility map (`exp5`), state-phenotype associations, cross-lineage transfer AUROCs. Verified: reproduces the paper's Table values (e.g. CRC MSI best-state AUROC 0.700; transfer 0.534/0.456). |
| **+ cached features** (~1.4 GB, see DATA.md) | UNI/CONCH patch embeddings | Atlas rebuilding, OT matching (`exp2`), CONCH atlas (`exp3`), soft assignment (`exp4`), MIL training. Scripts fail with a clear missing-file path until features are placed under `$PROJECT_ROOT/Experiment/datasets/features/`. |
| **+ raw WSIs + gated weights** | TCGA slides, HF access to UNI/CONCH | Feature extraction, CONCH zero-shot concept labeling (`exp1`), exemplar patch export. |

## 2. Get the data (see DATA.md for exact locations)

- **Cached UNI/CONCH patch features** — download from the Zenodo archive (DOI in `DATA.md`) into
  `datasets/features/`. This is what lets you skip WSI feature extraction.
- **Labels** — already included here under `results/labels/`. The per-task QC
  slide manifests are available from the authors on request.
- **Frozen K=24 atlas centroids** — included: `results/gi_morphology_state_centers.npy`.
- **Raw WSIs** — only needed if you want to re-extract features or re-export exemplar patches;
  obtain from TCGA/GDC and SurGen by accession (see `DATA.md`). Not required for the main results.

## 3. Reproduce the results

**(a) Morphology-state atlas + shared-MSI test + cross-lineage transfer (main UNI result)**
```bash
python scripts/build_morphology_atlas.py
python scripts/run_gi_molecular_morphology_atlas_sensitivity.py   # K=16/32/48 sensitivity
```
Outputs: atlas centroids, per-state association statistics, transfer AUROCs.

**(b) New analyses (this paper's added experiments)** — in `scripts/`:
```bash
python scripts/exp2_ot_matching__run_exp2.py       # optimal-transport cross-lineage matching
python scripts/exp3_conch_atlas__run_exp3.py       # end-to-end CONCH atlas replication
python scripts/exp4_soft_assign__run_exp4.py       # soft-assignment robustness
python scripts/exp5_visibility_map__run_exp5.py    # continuous visibility map
python scripts/exp1_conch_concepts__run_exp1.py    # CONCH zero-shot concept labeling (needs WSIs + CONCH weights)
python scripts/v4_algo_improvements__make_figures.py   # render the new figures
```
`_common__atlas_utils.py` holds shared helpers (feature loading, sampling, clustering) used by
exp2–5. Note the flattened naming `folder__file.py` reflects the original directory layout.

**(c) Supervised MIL benchmarks (reference)** — in `scripts/`:
```bash
bash scripts/run_crc_msi_qc_uni_5fold.sh       # CRC MSI, UNI
bash scripts/run_gi_msi_qc_uni_5fold.sh        # combined GI MSI
bash scripts/run_stad_msi_qc_abmil_5fold.sh    # STAD MSI
bash scripts/run_stad_subtype_qc_uni_5fold.sh  # STAD subtype
```

**(d) External SurGen evaluation** — `scripts/infer_option_a_external.py`,
`scripts/analyze_external_msi_predictions.py` (frozen-centroid inference on SurGen; see `DATA.md`).

**(e) Feature extraction from raw WSIs (optional)** — `scripts/extract_wsi_features.py`
(requires WSIs, openslide, and gated UNI/CONCH weights).

## 4. Model weights

UNI and CONCH are gated on Hugging Face (`MahmoodLab/UNI`, `MahmoodLab/conch`). Request access
and authenticate (`huggingface-cli login`) before any feature extraction or CONCH zero-shot step.
The atlas statistics and v4 analyses run on cached features and do not need the weights.

## 5. Map of entry points

| Goal | Script |
|---|---|
| Build the pooled K=24 atlas, associations, transfer | `scripts/build_morphology_atlas.py` |
| K-sensitivity (16/32/48) | `scripts/run_gi_molecular_morphology_atlas_sensitivity.py` |
| Export state exemplar patches | `scripts/export_gi_morphology_state_exemplars.py` |
| WSI feature extraction | `scripts/extract_wsi_features.py` |
| Supervised MIL 5-fold runs | `scripts/run_*_5fold.sh` |
| External SurGen inference / analysis | `scripts/infer_option_a_external.py`, `scripts/analyze_external_msi_predictions.py` |
| OT / CONCH-atlas / soft / visibility-map / concept | `scripts/exp2..5`, `exp1` |

## Known issues

- **Hardcoded absolute paths:** 13 files under `scripts/` embed a project-root path; set
  `PROJECT_ROOT` or edit the header constant before running.
- **GPU required** for feature extraction and MIL training; the atlas/statistics/v4 analyses
  run on CPU when features are cached.