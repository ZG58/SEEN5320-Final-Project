from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORT_FIG_DIR = PROJECT_ROOT / "report" / "figures" / "psa_scalar"
SLIDE_FIG_DIR = PROJECT_ROOT / "slide" / "pic" / "psa_scalar"

RAW_MAT_PATH = DATA_DIR / "raw_data.mat"
SCALAR_CSV_PATH = DATA_DIR / "psa_scalar_samples.csv"

INPUT_COLS = ["FAD_m3_h", "FRP_m3_h", "FVU_m3_h", "TRP_s", "TED_s", "TPR_s"]
TARGET_COLS = ["recovery", "purity", "productivity_mol_h_kg", "energy_kJ_kgCO2"]
MODEL_TARGETS = ["recovery", "purity", "productivity_mol_h_kg", "log_energy"]

INPUT_BOUNDS = {
    "FAD_m3_h": (0.5, 1.0),
    "FRP_m3_h": (0.35, 0.85),
    "FVU_m3_h": (1.0, 2.0),
    "TRP_s": (35.0, 55.0),
    "TED_s": (5.0, 15.0),
    "TPR_s": (15.0, 25.0),
}

INPUT_LABELS = {
    "FAD_m3_h": "Feed flowrate, FAD (m3/h)",
    "FRP_m3_h": "Replacement flowrate, FRP (m3/h)",
    "FVU_m3_h": "Vacuum flowrate, FVU (m3/h)",
    "TRP_s": "Replacement time, TRP (s)",
    "TED_s": "Equalization depressurization time, TED (s)",
    "TPR_s": "Pressurization time, TPR (s)",
}

INPUT_SHORT_LABELS = {
    "FAD_m3_h": "FAD",
    "FRP_m3_h": "FRP",
    "FVU_m3_h": "FVU",
    "TRP_s": "TRP",
    "TED_s": "TED",
    "TPR_s": "TPR",
}

TARGET_LABELS = {
    "recovery": "CO2 recovery",
    "purity": "CO2 purity",
    "productivity_mol_h_kg": "Productivity",
    "energy_kJ_kgCO2": "Specific energy",
    "log_energy": "log10(specific energy)",
}

RANDOM_STATE = 42

def ensure_output_dirs() -> None:
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    SLIDE_FIG_DIR.mkdir(parents=True, exist_ok=True)


def input_bounds_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"variable": col, "lower": float(lo), "upper": float(hi)}
            for col, (lo, hi) in INPUT_BOUNDS.items()
        ]
    )


def input_bounds_arrays() -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([INPUT_BOUNDS[col][0] for col in INPUT_COLS], dtype=float)
    upper = np.array([INPUT_BOUNDS[col][1] for col in INPUT_COLS], dtype=float)
    return lower, upper


def load_scalar_samples(path: Path | str = SCALAR_CSV_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing scalar sample CSV: {path}")
    frame = pd.read_csv(path)
    required = ["sample_id", "status", *INPUT_COLS, *TARGET_COLS]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise KeyError(f"{path} is missing required columns: {missing}")
    frame = frame.loc[frame["status"].eq("success")].copy()
    numeric_cols = ["sample_id", *INPUT_COLS, *TARGET_COLS]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=[*INPUT_COLS, *TARGET_COLS]).reset_index(drop=True)
    frame["log_energy"] = np.log10(np.clip(frame["energy_kJ_kgCO2"].to_numpy(float), 1e-12, None))
    return frame


def validate_scalar_samples(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("No successful scalar PSA samples were loaded.")
    values = frame[[*INPUT_COLS, *TARGET_COLS]].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("The scalar PSA sample table contains NaN or infinite values.")
    lower, upper = input_bounds_arrays()
    x = frame[INPUT_COLS].to_numpy(float)
    outside = (x < lower - 1e-10) | (x > upper + 1e-10)
    if outside.any():
        bad_rows = np.where(outside.any(axis=1))[0][:10].tolist()
        raise ValueError(f"Some inputs lie outside the configured LHS bounds. Example row indices: {bad_rows}")


def transformed_targets(frame: pd.DataFrame) -> np.ndarray:
    return frame[MODEL_TARGETS].to_numpy(float)


def inverse_transformed_targets(values: np.ndarray) -> pd.DataFrame:
    values = np.asarray(values, dtype=float)
    recovery = np.clip(values[:, 0], 0.0, 1.0)
    purity = np.clip(values[:, 1], 0.0, 1.0)
    productivity = np.clip(values[:, 2], 0.0, None)
    log_energy = np.clip(values[:, 3], -12.0, 12.0)
    energy = np.power(10.0, log_energy)
    return pd.DataFrame(
        {
            "recovery": recovery,
            "purity": purity,
            "productivity_mol_h_kg": productivity,
            "energy_kJ_kgCO2": energy,
            "log_energy": log_energy,
        }
    )


def clip_to_observed_bounds(predictions: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    clipped = predictions.copy()
    for col in [*TARGET_COLS, "log_energy"]:
        if col not in clipped.columns or col not in observed.columns:
            continue
        clipped[col] = clipped[col].clip(float(observed[col].min()), float(observed[col].max()))
    clipped["energy_kJ_kgCO2"] = np.power(10.0, clipped["log_energy"])
    return clipped


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    denom = float(np.sum((actual - np.mean(actual)) ** 2))
    if len(actual) < 2 or denom <= 0.0:
        r2 = np.nan
    else:
        r2 = float(1.0 - np.sum(error**2) / denom)
    scale = float(np.max(actual) - np.min(actual))
    if scale <= 0.0 or not np.isfinite(scale):
        scale = 1.0
    return {"rmse": rmse, "mae": mae, "r2": r2, "normalized_rmse": rmse / scale}


def nondominated_mask(scores: np.ndarray, maximize: np.ndarray | list[bool] | None = None) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D array")
    if maximize is None:
        maximize = np.ones(scores.shape[1], dtype=bool)
    maximize = np.asarray(maximize, dtype=bool)
    oriented = scores.copy()
    oriented[:, ~maximize] *= -1.0
    n_rows = oriented.shape[0]
    mask = np.ones(n_rows, dtype=bool)
    for i in range(n_rows):
        if not mask[i]:
            continue
        dominates_i = np.all(oriented >= oriented[i], axis=1) & np.any(oriented > oriented[i], axis=1)
        if dominates_i.any():
            mask[i] = False
    return mask


def near_pareto_mask(
    scores: np.ndarray,
    front_mask: np.ndarray,
    min_count: int,
    fraction: float = 0.10,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    front_mask = np.asarray(front_mask, dtype=bool)
    if scores.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    if not front_mask.any():
        return np.zeros(scores.shape[0], dtype=bool)

    lo = np.min(scores, axis=0)
    hi = np.max(scores, axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    scaled = (scores - lo) / span
    front = scaled[front_mask]
    distances = np.sqrt(((scaled[:, None, :] - front[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    target_count = min(scores.shape[0], max(min_count, int(np.ceil(fraction * scores.shape[0]))))
    order = np.argsort(distances)
    near = np.zeros(scores.shape[0], dtype=bool)
    near[order[:target_count]] = True
    near |= front_mask
    return near


def empirical_pareto_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    pr_scores = frame[["purity", "recovery"]].to_numpy(float)
    pr_front = nondominated_mask(pr_scores, maximize=[True, True])
    pr_near = near_pareto_mask(pr_scores, pr_front, min_count=60, fraction=0.10)

    pe_scores = frame[["productivity_mol_h_kg", "energy_kJ_kgCO2"]].to_numpy(float)
    pe_front = nondominated_mask(pe_scores, maximize=[True, False])
    pe_near = near_pareto_mask(
        np.column_stack([pe_scores[:, 0], -pe_scores[:, 1]]),
        pe_front,
        min_count=60,
        fraction=0.10,
    )

    return {
        "purity_recovery_front": pr_front,
        "purity_recovery_near": pr_near,
        "productivity_energy_front": pe_front,
        "productivity_energy_near": pe_near,
        "pareto_focus": pr_near | pe_near,
    }


def summarize_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        values = frame[col].to_numpy(float)
        rows.append(
            {
                "column": col,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
            }
        )
    return pd.DataFrame(rows)
