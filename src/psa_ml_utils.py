from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROFILE_DIR = DATA_DIR / "profiles"
SLIDE_ML_FIG_DIR = PROJECT_ROOT / "slide" / "pic" / "psa_ml"
REPORT_ML_FIG_DIR = PROJECT_ROOT / "report" / "figures" / "psa_ml"

DESIGN_COLS = [f"x{i}" for i in range(1, 8)]
METRIC_COLS = ["purity", "recovery", "productivity_mol_kg_h", "energy_kWh_ton"]
REGRESSION_TARGETS = ["purity", "recovery", "productivity_mol_kg_h", "log_energy"]

DESIGN_LABELS = {
    "x1": "Adsorption pressure",
    "x2": "Adsorption time",
    "x3": "Light-product reflux ratio",
    "x4": "Feed velocity",
    "x5": "Blowdown pressure",
    "x6": "Pressurization time",
    "x7": "Counter-current depressurization time",
}

METRIC_LABELS = {
    "purity": "CO2 purity",
    "recovery": "CO2 recovery",
    "productivity_mol_kg_h": "Productivity",
    "energy_kWh_ton": "Specific energy",
    "log_energy": "log10(specific energy)",
}


def ensure_output_dirs() -> None:
    SLIDE_ML_FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_ML_FIG_DIR.mkdir(parents=True, exist_ok=True)


def minmax(values: pd.Series) -> pd.Series:
    span = values.max() - values.min()
    if not np.isfinite(span) or span == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.min()) / span


def pareto_mask(frame: pd.DataFrame) -> np.ndarray:
    objectives = frame[["purity", "recovery", "productivity_mol_kg_h", "energy_kWh_ton"]].to_numpy(float)
    scores = objectives.copy()
    scores[:, 3] = -np.log10(np.clip(scores[:, 3], 1e-12, None))

    n_rows = scores.shape[0]
    dominated = np.zeros(n_rows, dtype=bool)
    for i in range(n_rows):
        if dominated[i]:
            continue
        better_or_equal = np.all(scores >= scores[i], axis=1)
        strictly_better = np.any(scores > scores[i], axis=1)
        dominated[i] = bool(np.any(better_or_equal & strictly_better))
    return ~dominated


def load_manifest() -> pd.DataFrame:
    path = DATA_DIR / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest file: {path}")

    manifest = pd.read_csv(path)
    required = ["sample_id", "status", "profile_csv_path", *DESIGN_COLS, *METRIC_COLS]
    missing = [col for col in required if col not in manifest.columns]
    if missing:
        raise KeyError(f"Missing required manifest columns: {missing}")

    successful = manifest.loc[manifest["status"].eq("success")].copy()
    if len(successful) != len(manifest):
        raise ValueError("The pipeline expects all manifest rows to be successful.")
    if len(successful) != 1000:
        raise ValueError(f"Expected 1000 successful samples, found {len(successful)}.")

    for col in [*DESIGN_COLS, *METRIC_COLS, "runtime_s"]:
        if col in successful.columns:
            successful[col] = pd.to_numeric(successful[col], errors="coerce")

    successful["log_energy"] = np.log10(successful["energy_kWh_ton"].clip(lower=1e-12))
    successful["balanced_score"] = (
        minmax(successful["purity"])
        + minmax(successful["recovery"])
        + minmax(successful["productivity_mol_kg_h"])
        + (1.0 - minmax(successful["log_energy"]))
    ) / 4.0
    successful["is_pareto"] = pareto_mask(successful)
    score_threshold = successful["balanced_score"].quantile(0.80)
    successful["high_performer"] = successful["balanced_score"].ge(score_threshold).astype(int)
    return successful


def profile_path(relative_path: str) -> Path:
    return DATA_DIR / Path(str(relative_path).replace("\\", "/"))


def clean_feature_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
    )
