# Shared cross-organ MSI signal in pathology foundation-model space

Code and lightweight source data supporting the manuscript analyses of
cross-organ microsatellite-instability (MSI) prediction in colorectal and
gastric cancer.

## Main findings represented in this repository

| Analysis | Reported value | Machine-readable source |
|---|---:|---|
| CRC-trained TransMIL-lite evaluated in gastric cancer | AUROC 0.727 | `results/e2c_mil_projection_manifest.json` |
| Continuous direction, CRC to gastric | AUROC 0.660 | `results/e2_representation_audit_manifest.json` |
| Continuous direction, gastric to CRC | AUROC 0.636 | `results/e2_representation_audit_manifest.json` |
| Cross-organ direction alignment | cosine 0.266; permutation p = 0.006 | `results/e2_representation_audit_manifest.json` |
| Transferred-logit reconstruction | R^2 0.810 from 1,024D; 0.137 from K = 24 | `results/e2c_mil_projection_manifest.json` |
| K = 24 morphology states, CRC to gastric | AUROC 0.593 | `results/e1_ladder_manifest.json` |
| Frozen CRC state 0 evaluated in SurGen | AUROC 0.638 | `results/surgen_state0_bootstrap_ci.json` |
| Mucinous positive control | AUROC 0.722 / 0.700 | `results/e1_ladder_manifest.json` |

The reconstruction R^2 values quantify fidelity to the transferred model output;
they are not estimates of the proportion of MSI biology retained. Continuous
directions, K = 24 transfer, score reconstruction, and the reverse-direction
representation ladder are separate analyses and should not be interpreted as a
single controlled compression experiment.

## Repository layout

- `scripts/`: feature extraction, MIL training, morphology-state construction,
  final statistical analyses, and figure generation.
- `results/`: lightweight derived tables and machine-readable outputs underlying
  the reported figures and numerical claims.
- `DATA.md`: public data sources, accessions, and the boundary between distributed
  and non-distributed data.
- `USAGE.md`: environment setup and the main reproduction routes.

Raw whole-slide images and large patch-embedding caches are not duplicated on
GitHub. Public image sources and instructions for regenerating the embeddings
are documented in `DATA.md`.

## Reproducibility boundary

The repository supports two levels of reproduction:

1. The lightweight result files allow the reported values and plots to be
   inspected without downloading whole-slide images.
2. The analysis scripts reproduce the models and statistics when the public
   slides and required foundation-model features are available.

The colorectal state-0 exemplar workflow is label-directed: the candidate pool
is restricted to MSI-H before centroid ranking. The gastric state-0 workflow is
label-blind. These images illustrate the states and are not independent evidence
of an MSI association.

## License

MIT; see `LICENSE`.
