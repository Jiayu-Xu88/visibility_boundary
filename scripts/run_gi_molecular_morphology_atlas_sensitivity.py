#!/usr/bin/env python3
"""K-sensitivity wrapper for the GI molecular morphology atlas pilot."""
from __future__ import annotations

import json
from pathlib import Path

import run_gi_molecular_morphology_atlas as atlas

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_OUT = PROJECT_ROOT / "Experiment/analysis/medical_morphology_atlas_k_sensitivity"
BASE_OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    summaries = []
    for k in [16, 32, 48]:
        print(f"K sensitivity run: K={k}")
        atlas.N_STATES = k
        atlas.OUT_DIR = BASE_OUT / f"K{k}"
        atlas.OUT_DIR.mkdir(parents=True, exist_ok=True)
        atlas.main()
        summary_path = atlas.OUT_DIR / "gi_molecular_morphology_atlas_summary.json"
        data = json.loads(summary_path.read_text())
        summaries.append({
            "K": k,
            "shared_msi_states": data["state_summary"]["shared_msi_states_fdr10"],
            "stad_msi_specific_states": data["state_summary"]["stad_msi_specific_states_fdr10"],
            "crc_msi_specific_states": data["state_summary"]["crc_msi_specific_states_fdr10"],
            "ebv_states": data["state_summary"]["ebv_enriched_states_fdr10_abs_cliff_ge_0p20"],
            "stad_to_crc_auroc": data["transfer"]["stad_to_crc_msi"]["auroc"],
            "crc_to_stad_auroc": data["transfer"]["crc_to_stad_msi"]["auroc"],
        })
    (BASE_OUT / "k_sensitivity_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    report = BASE_OUT / "GI_MOLECULAR_MORPHOLOGY_ATLAS_K_SENSITIVITY.md"
    report.write_text(
        "# GI Molecular Morphology Atlas K Sensitivity\n\n"
        "| K | shared MSI states | STAD-specific MSI states | CRC-specific MSI states | EBV states | STAD->CRC AUROC | CRC->STAD AUROC |\n"
        "|---:|---|---|---|---|---:|---:|\n"
        + "\n".join(
            f"| {row['K']} | {row['shared_msi_states']} | {row['stad_msi_specific_states']} | {row['crc_msi_specific_states']} | {row['ebv_states']} | {row['stad_to_crc_auroc']:.4f} | {row['crc_to_stad_auroc']:.4f} |"
            for row in summaries
        )
        + "\n",
    )
    print(json.dumps(summaries, indent=2))
    print(report)


if __name__ == "__main__":
    main()
