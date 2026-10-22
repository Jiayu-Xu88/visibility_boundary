# Data availability and locations

This repository ships **code and small artifacts only**. Below is where every dataset lives.

## Included in this repo (small)

- `results/gi_morphology_state_centers.npy` — frozen K=24 morphology-atlas centroids (UNI).
- `results/*_state_fractions.tsv` — patient- and slide-level morphology-state fractions.
- `results/labels/` — MSI status (MANTIS/MSIsensor) and STAD subtype labels (TCGA barcodes).
- Per-task QC slide manifests, recording which slides entered each analysis, are not shipped here; they are available from the authors on request.

## Large cached data — recorded here, NOT distributed in this repository

These files are too large for GitHub and are **not uploaded**. They are recorded below so the
provenance and sizes are documented. To reproduce from cached features rather than from raw
WSIs, obtain them from the authors on request, or re-extract them from the public WSIs with
`scripts/extract_wsi_features.py` (see USAGE.md).

| Artifact | Approx. size | How to obtain |
|---|---|---|
| Frozen **UNI** patch embeddings (`datasets/features/uni/`) | ~0.9 GB | from authors on request, or re-extract from TCGA WSIs |
| Frozen **CONCH** patch embeddings (`datasets/features/conch/`) | ~0.5 GB | from authors on request, or re-extract |
| UNI external (CPTAC) embeddings | ~0.6 GB | from authors on request |
| Supervised MIL checkpoints (`core_code/checkpoints/`) | ~2.9 GB | not needed to reproduce reported results |
| Raw WSIs, methylation, SurGen raw | 21 / 11 / 23 GB | public — see below; not re-hosted |

## Public — cite by accession, do NOT re-host

- **TCGA-COAD / TCGA-READ / TCGA-STAD** diagnostic WSIs: NIH GDC Data Portal
  (https://portal.gdc.cancer.gov/).
- **MSI labels**: TCGA PanCancer Atlas MANTIS/MSIsensor scores via cBioPortal
  (https://www.cbioportal.org/), primary source Bonneville et al., *JCO Precis Oncol* 2017
  (doi:10.1200/PO.17.00073).
- **STAD molecular subtype**: TCGA gastric taxonomy (Nature 2014, doi:10.1038/nature13480).
- **SurGen SR1482** external cohort: EBI BioStudies accession **S-BIAD1285**
  (doi:10.6019/S-BIAD1285); dataset paper Myles et al., *GigaScience* 2025
  (doi:10.1093/gigascience/giaf086); pre-extracted UNI embeddings on Zenodo
  (doi:10.5281/zenodo.14047723), released by the SurGen authors.

## Not part of this paper

- `methylation/` (11 GB) — not used here.