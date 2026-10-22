# Usage

## Environment

Create a Python environment and install the declared dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The reported analyses used Python 3.12. A CUDA-capable environment is required
for feature extraction and MIL training; the patient-level statistical analyses
can run on CPU once features have been prepared.

## Data layout

Set `PROJECT_ROOT` to a directory containing the study data under `Experiment/`:

```bash
export PROJECT_ROOT=/path/to/proj-Gastrointestinal
```

Public cohorts, accessions, foundation-model requirements, and the expected
locations of large non-distributed files are documented in `DATA.md`.

## Main reproduction routes

### Supervised prediction

- `scripts/extract_wsi_features.py`: extract patch-level foundation-model features.
- `scripts/train_mil_classifier.py`: train the supervised MIL models.
- `scripts/e2c_mil_projection.py`: evaluate CRC-to-gastric supervised transfer
  and transferred-score reconstruction.

### Continuous representation analysis

Run:

```bash
python scripts/e2_representation_audit.py
```

This script performs nested within-lineage evaluation, source-fitted
cross-direction scoring, and the direction-alignment permutation with
regularisation reselected in each replicate. The final machine-readable output
is `results/e2_representation_audit_manifest.json`.

### Morphology-state analysis

- `scripts/build_morphology_atlas.py`: construct the morphology-state atlas.
- `scripts/e1_positive_control_ladder.py`: evaluate MSI and mucinous cross-organ
  transfer at the morphology-state level.
- `scripts/m3_center_and_m11_control.py` and
  `scripts/m4_internal_external_cv.py`: centre-held-out and related controls.
- `scripts/analyze_external_msi_predictions.py`: evaluate frozen CRC state 0 in
  the SurGen cohort.

### Exemplar selection

- `scripts/export_gi_morphology_state_exemplars.py` performs the label-directed
  exploratory exporter used for the colorectal state-0 illustrations.
- `scripts/export_gastric_state0_labelblind_exemplars.py` performs the gastric
  label-blind state-0 selection.

Do not describe the colorectal exemplar selection as label-blind.

### Main figures

Run:

```bash
python scripts/make_foundation_space_figures.py
```

By default, figures are written to `generated_figures/`. Set `FIGURE_OUTPUT_DIR`
to choose another output directory. Histology exemplars require local slide
access through `PROJECT_ROOT`; the statistical panels use the lightweight files
in `results/`.

## Statistical interpretation

Target-patient bootstrap intervals hold the fitted source prediction vector
fixed and resample target patients. Source-label permutation analyses refit the
source model after every shuffle. These procedures address different questions
and their p values and confidence intervals need not lead to identical binary
decisions.
