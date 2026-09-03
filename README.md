# Microsatellite-instability prediction can transfer across gastric and colorectal cohorts without a shared low-dimensional morphology program

Code, the frozen intermediates that make the results recomputable without the
whole-slide images, and the machine-readable supplementary results.

## Where each reported value comes from

Values below were read from the analysis outputs at the time this file was
written; the script named is the one that produces them.

| Reported quantity | Value | Script |
|---|---|---|
| Supervised classifier transfers to gastric | AUROC 0.73 | `scripts/e2c_mil_projection.py` |
| Image-independent mucinous control, same 24 states | AUROC 0.72 | `scripts/e1_positive_control_ladder.py` |
| Morphology-state MSI-H, matched training size | AUROC 0.54 | `scripts/m3c_size_matched_crossorgan.py` |
| Morphology-state MSI-H, full training cohort | AUROC 0.593 | `scripts/reconcile_transfer_auroc.py` |
| Within-gastric, across contributing centres | AUROC 0.678 | `scripts/m3_center_and_m11_control.py` |
| Within-gastric, leave-one-centre-out | AUROC 0.694 | `scripts/m4_internal_external_cv.py` |
| Within-colorectal, random folds to leave-one-centre-out | 0.708 to 0.590 | `scripts/m4_internal_external_cv.py` |
| External colorectal cohort, frozen state 0 (SurGen SR1482) | AUROC 0.639 [0.549, 0.725] | `scripts/analyze_external_msi_predictions.py` |
| Label contrast, difference in differences (gastric) | -0.19 [-0.39, +0.02], p = 0.15 | `scripts/did_label_contrast.py` |
| Reconstruction of the transferring score from 24 states | R2 0.14 of 0.81 (17% kept) | `scripts/e2c_mil_projection.py` |
| Generic compression floor measured on control scores | 62-69% kept | `scripts/m3_center_and_m11_control.py` |
| Attention on tumour epithelium, top 5% vs bottom 50% | 0.51 vs 0.10 | `scripts/e2c_mil_projection.py` |
| Cross-organ cosine of MSI directions | 0.27 (ceilings 0.60/0.51, floor +/-0.17) | `scripts/e2_representation_audit.py` |

## Layout

```
scripts/   the analyses above, plus the atlas construction, feature extraction
           and MIL training they sit on
results/   the machine-readable supplementary results (Supplementary Tables
           S11-S17 and the manifests named in Supplementary Note S3); the frozen
           atlas centroids and patient-level state fractions; labels/; and the
           pre-registration document
```

## Reproducing

The patient-level results recompute from
`results/gi_morphology_state_patient_fractions.tsv` and `results/labels/`
without the whole-slide images. Only feature extraction
(`scripts/extract_wsi_features.py`) and MIL training
(`scripts/train_mil_classifier.py`) need the slides.

`scripts/make_figures.py` reads the full set of analysis manifests, not all of
which are distributed here; run the analyses first to regenerate them.

`DATA.md` gives the accession and DOI for every public dataset; `USAGE.md` gives
the run order.

## License

MIT, see `LICENSE`.
