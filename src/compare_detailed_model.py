from __future__ import annotations

import numpy as np
import pandas as pd

from psa_surrogate_utils import DATA_DIR, DESIGN_COLS, PHYSICAL_TARGETS


def normalize_detailed_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    aliases = {
        "detailed_purity": "purity",
        "sim_purity": "purity",
        "detailed_recovery": "recovery",
        "sim_recovery": "recovery",
        "detailed_productivity_mol_kg_h": "productivity_mol_kg_h",
        "sim_productivity_mol_kg_h": "productivity_mol_kg_h",
        "detailed_energy_kWh_ton": "energy_kWh_ton",
        "sim_energy_kWh_ton": "energy_kWh_ton",
    }
    for old, new in aliases.items():
        if old in renamed.columns and new not in renamed.columns:
            renamed = renamed.rename(columns={old: new})
    return renamed


def main() -> None:
    input_path = DATA_DIR / "detailed_model_input.csv"
    results_path = DATA_DIR / "detailed_model_results.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing detailed-model input file: {input_path}")

    candidates = pd.read_csv(input_path)
    if not results_path.exists():
        print(f"{results_path} not found. Detailed-model comparison skipped.")
        return

    detailed = normalize_detailed_columns(pd.read_csv(results_path))
    required = ["candidate_id", *PHYSICAL_TARGETS]
    missing = [col for col in required if col not in detailed.columns]
    if missing:
        raise KeyError(f"Detailed-model results are missing required columns: {missing}")

    for col in PHYSICAL_TARGETS:
        detailed[col] = pd.to_numeric(detailed[col], errors="coerce")
    detailed = detailed.dropna(subset=PHYSICAL_TARGETS).copy()
    detailed_for_merge = detailed[["candidate_id", *PHYSICAL_TARGETS]].rename(
        columns={target: f"{target}_detailed" for target in PHYSICAL_TARGETS}
    )
    merged = candidates.merge(
        detailed_for_merge,
        on="candidate_id",
        how="inner",
    )
    if merged.empty:
        raise ValueError("No matching candidate_id values between input and detailed results.")

    for target in PHYSICAL_TARGETS:
        pred_col = f"pred_{target}"
        detailed_col = f"{target}_detailed"
        merged[f"abs_error_{target}"] = (merged[pred_col] - merged[detailed_col]).abs()
        denom = np.maximum(merged[detailed_col].abs(), 1e-12)
        merged[f"rel_error_{target}"] = merged[f"abs_error_{target}"] / denom

    if "energy_kWh_ton_detailed" in merged.columns:
        merged["log_energy_detailed"] = np.log10(np.clip(merged["energy_kWh_ton_detailed"], 1e-12, None))
        merged["abs_error_log_energy"] = (merged["pred_log_energy"] - merged["log_energy_detailed"]).abs()

    comparison_path = DATA_DIR / "detailed_model_comparison.csv"
    merged.to_csv(comparison_path, index=False)

    summary_rows = []
    for target in PHYSICAL_TARGETS:
        error = merged[f"abs_error_{target}"]
        rel = merged[f"rel_error_{target}"]
        summary_rows.append(
            {
                "target": target,
                "n": len(merged),
                "mae": float(error.mean()),
                "median_abs_error": float(error.median()),
                "mean_relative_error": float(rel.mean()),
            }
        )
    if "abs_error_log_energy" in merged.columns:
        summary_rows.append(
            {
                "target": "log_energy",
                "n": len(merged),
                "mae": float(merged["abs_error_log_energy"].mean()),
                "median_abs_error": float(merged["abs_error_log_energy"].median()),
                "mean_relative_error": np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_path = DATA_DIR / "detailed_model_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {comparison_path}")
    print(f"Saved {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
