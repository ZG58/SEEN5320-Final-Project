from __future__ import annotations

import csv
import itertools
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORT_GA_FIG_DIR = PROJECT_ROOT / "report" / "figures" / "psa_ga"
SLIDE_GA_FIG_DIR = PROJECT_ROOT / "slide" / "pic" / "psa_ga"

DESIGN_COLS = [f"x{i}" for i in range(1, 8)]
PHYSICAL_TARGETS = ["purity", "recovery", "productivity_mol_kg_h", "energy_kWh_ton"]
PREDICTION_TARGETS = [*PHYSICAL_TARGETS, "log_energy"]
MODEL_TARGETS = ["purity_log_excess", "recovery", "productivity_mol_kg_h", "log_energy"]

FEED_CO2_FRACTION = 0.15
MIN_PURITY_EXCESS = 1e-6
RANDOM_STATE = 42

DESIGN_LABELS = {
    "x1": "Adsorption pressure",
    "x2": "Adsorption time",
    "x3": "Light-product reflux ratio",
    "x4": "Feed velocity",
    "x5": "Blowdown pressure",
    "x6": "Pressurization time",
    "x7": "Counter-current depressurization time",
}

DESIGN_SHORT_LABELS = {
    "x1": "P_ads",
    "x2": "t_ads",
    "x3": "R_L",
    "x4": "v_feed",
    "x5": "P_blow",
    "x6": "t_press",
    "x7": "t_depress",
}

TARGET_LABELS = {
    "purity": "CO2 purity",
    "recovery": "CO2 recovery",
    "productivity_mol_kg_h": "Productivity",
    "energy_kWh_ton": "Specific energy",
    "log_energy": "log10(specific energy)",
}


def ensure_output_dirs() -> None:
    REPORT_GA_FIG_DIR.mkdir(parents=True, exist_ok=True)
    SLIDE_GA_FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_manifest(expect_n: int | None = 1000) -> pd.DataFrame:
    path = DATA_DIR / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest file: {path}")

    frame = pd.read_csv(path)
    required = ["sample_id", "status", *DESIGN_COLS, *PHYSICAL_TARGETS]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise KeyError(f"Missing required manifest columns: {missing}")

    success = frame.loc[frame["status"].eq("success")].copy()
    if success.empty:
        raise ValueError("No successful samples found in data/manifest.csv")
    if expect_n is not None and len(success) != expect_n:
        raise ValueError(f"Expected {expect_n} successful samples, found {len(success)}")

    numeric_cols = ["sample_id", *DESIGN_COLS, *PHYSICAL_TARGETS]
    if "runtime_s" in success.columns:
        numeric_cols.append("runtime_s")
    for col in numeric_cols:
        success[col] = pd.to_numeric(success[col], errors="coerce")
    success = success.dropna(subset=[*DESIGN_COLS, *PHYSICAL_TARGETS]).reset_index(drop=True)
    success["log_energy"] = np.log10(np.clip(success["energy_kWh_ton"].to_numpy(float), 1e-12, None))
    return success


def _read_yaml_bounds(config_path: Path) -> tuple[list[float], list[float]] | None:
    try:
        import yaml  # type: ignore
    except Exception:
        return None

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        settings = cfg.get("optimization_settings", {})
        lb = settings.get("lb")
        ub = settings.get("ub")
        if isinstance(lb, list) and isinstance(ub, list) and len(lb) >= 7 and len(ub) >= 7:
            return [float(v) for v in lb[:7]], [float(v) for v in ub[:7]]
    except Exception:
        return None
    return None


def _read_regex_bounds(config_path: Path) -> tuple[list[float], list[float]] | None:
    text = config_path.read_text(encoding="utf-8", errors="ignore")
    lb_match = re.search(r"^\s*lb\s*:\s*\[([^\]]+)\]", text, flags=re.MULTILINE)
    ub_match = re.search(r"^\s*ub\s*:\s*\[([^\]]+)\]", text, flags=re.MULTILINE)
    if not lb_match or not ub_match:
        return None

    def parse_list(raw: str) -> list[float]:
        return [float(item.strip()) for item in raw.split(",") if item.strip()]

    lb = parse_list(lb_match.group(1))
    ub = parse_list(ub_match.group(1))
    if len(lb) >= 7 and len(ub) >= 7:
        return lb[:7], ub[:7]
    return None


def variable_bounds(manifest: pd.DataFrame | None = None) -> pd.DataFrame:
    config_path = DATA_DIR / "ProcessConfig.yaml"
    bounds = None
    if config_path.exists():
        bounds = _read_yaml_bounds(config_path) or _read_regex_bounds(config_path)

    if bounds is None:
        if manifest is None:
            manifest = load_manifest()
        lb = manifest[DESIGN_COLS].min().to_list()
        ub = manifest[DESIGN_COLS].max().to_list()
        source = "manifest_minmax"
    else:
        lb, ub = bounds
        source = "ProcessConfig.yaml"

    rows = []
    for col, lo, hi in zip(DESIGN_COLS, lb, ub):
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError(f"Invalid bound for {col}: lower={lo}, upper={hi}")
        rows.append({"variable": col, "lower": float(lo), "upper": float(hi), "source": source})
    return pd.DataFrame(rows)


def bounds_arrays(bounds: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ordered = bounds.set_index("variable").loc[DESIGN_COLS]
    return ordered["lower"].to_numpy(float), ordered["upper"].to_numpy(float)


def stable_split(n_rows: int, test_size: float = 0.2, seed: int = RANDOM_STATE) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_rows)
    n_test = int(round(n_rows * test_size))
    test_idx = np.sort(indices[:n_test])
    train_idx = np.sort(indices[n_test:])
    return train_idx, test_idx


def polynomial_powers(n_features: int = 7, degree: int = 3) -> list[tuple[int, ...]]:
    powers: list[tuple[int, ...]] = [tuple([0] * n_features)]
    for total_degree in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(n_features), total_degree):
            exp = [0] * n_features
            for item in combo:
                exp[item] += 1
            powers.append(tuple(exp))
    return powers


def power_label(power: Iterable[int]) -> str:
    pieces = []
    for col, exponent in zip(DESIGN_COLS, power):
        if exponent == 1:
            pieces.append(col)
        elif exponent > 1:
            pieces.append(f"{col}^{exponent}")
    return "1" if not pieces else "*".join(pieces)


def polynomial_basis(x_scaled: np.ndarray, powers: list[tuple[int, ...]]) -> np.ndarray:
    x_scaled = np.asarray(x_scaled, dtype=float)
    basis = np.ones((x_scaled.shape[0], len(powers)), dtype=float)
    for j, power in enumerate(powers):
        for feature_idx, exponent in enumerate(power):
            if exponent:
                basis[:, j] *= x_scaled[:, feature_idx] ** exponent
    return basis


def squared_euclidean_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left_norm = np.sum(left**2, axis=1)[:, None]
    right_norm = np.sum(right**2, axis=1)[None, :]
    return np.maximum(left_norm + right_norm - 2.0 * left @ right.T, 0.0)


def rbf_kernel(left: np.ndarray, right: np.ndarray, length_scale: float) -> np.ndarray:
    if length_scale <= 0 or not np.isfinite(length_scale):
        raise ValueError(f"Invalid RBF length scale: {length_scale}")
    return np.exp(-0.5 * squared_euclidean_distances(left, right) / (length_scale**2))


def weighted_knn_predict_scaled(
    query_x_scaled: np.ndarray,
    train_x_scaled: np.ndarray,
    train_y_scaled: np.ndarray,
    k: int,
    epsilon: float = 1e-12,
) -> np.ndarray:
    query_x_scaled = np.asarray(query_x_scaled, dtype=float)
    train_x_scaled = np.asarray(train_x_scaled, dtype=float)
    train_y_scaled = np.asarray(train_y_scaled, dtype=float)
    k_eff = int(min(max(k, 1), len(train_x_scaled)))
    distances = np.sqrt(squared_euclidean_distances(query_x_scaled, train_x_scaled))
    neighbor_idx = np.argpartition(distances, kth=k_eff - 1, axis=1)[:, :k_eff]
    neighbor_dist = np.take_along_axis(distances, neighbor_idx, axis=1)
    weights = 1.0 / (neighbor_dist + epsilon)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return np.einsum("ij,ijk->ik", weights, train_y_scaled[neighbor_idx])


def transform_targets(frame: pd.DataFrame, feed_co2: float = FEED_CO2_FRACTION) -> np.ndarray:
    purity_excess = np.clip(frame["purity"].to_numpy(float) - feed_co2, MIN_PURITY_EXCESS, None)
    recovery = frame["recovery"].to_numpy(float)
    productivity = frame["productivity_mol_kg_h"].to_numpy(float)
    log_energy = np.log10(np.clip(frame["energy_kWh_ton"].to_numpy(float), 1e-12, None))
    return np.column_stack([np.log10(purity_excess), recovery, productivity, log_energy])


def inverse_targets(transformed: np.ndarray, feed_co2: float = FEED_CO2_FRACTION) -> pd.DataFrame:
    transformed = np.asarray(transformed, dtype=float)
    purity = feed_co2 + np.power(10.0, transformed[:, 0])
    recovery = transformed[:, 1]
    productivity = transformed[:, 2]
    log_energy = transformed[:, 3]
    energy = np.power(10.0, np.clip(log_energy, -12, 12))

    return pd.DataFrame(
        {
            "purity": np.clip(purity, 0.0, 1.0),
            "recovery": np.clip(recovery, 0.0, 1.0),
            "productivity_mol_kg_h": np.clip(productivity, 0.0, None),
            "energy_kWh_ton": np.clip(energy, 0.0, None),
            "log_energy": log_energy,
        }
    )


def fit_ridge(phi: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(phi.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    lhs = phi.T @ phi + alpha * penalty
    rhs = phi.T @ y
    return np.linalg.solve(lhs, rhs)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = actual - predicted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan,
    }


def save_model(model: dict, path: Path | None = None) -> Path:
    if path is None:
        path = DATA_DIR / "surrogate_model.json"
    serializable = json.loads(json.dumps(model, default=_json_default))
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return path


def load_model(path: Path | None = None) -> dict:
    if path is None:
        path = DATA_DIR / "surrogate_model.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing surrogate model file: {path}")
    model = json.loads(path.read_text(encoding="utf-8"))
    if "basis_powers" in model:
        model["basis_powers"] = [tuple(int(v) for v in power) for power in model["basis_powers"]]
    return model


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _apply_prediction_bounds(predictions: pd.DataFrame, prediction_bounds: dict | None) -> pd.DataFrame:
    if prediction_bounds:
        for target in ["purity", "recovery", "productivity_mol_kg_h", "log_energy"]:
            if target in prediction_bounds:
                lo = float(prediction_bounds[target]["min"])
                hi = float(prediction_bounds[target]["max"])
                predictions[target] = predictions[target].clip(lower=lo, upper=hi)
        predictions["energy_kWh_ton"] = np.power(10.0, predictions["log_energy"])
        if "energy_kWh_ton" in prediction_bounds:
            lo = float(prediction_bounds["energy_kWh_ton"]["min"])
            hi = float(prediction_bounds["energy_kWh_ton"]["max"])
            predictions["energy_kWh_ton"] = predictions["energy_kWh_ton"].clip(lower=lo, upper=hi)
            predictions["log_energy"] = np.log10(np.clip(predictions["energy_kWh_ton"], 1e-12, None))
    return predictions


def predict_surrogate(model: dict, x_raw: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(x_raw, pd.DataFrame):
        x_values = x_raw[DESIGN_COLS].to_numpy(float)
    else:
        x_values = np.asarray(x_raw, dtype=float)
    input_mean = np.asarray(model["input_mean"], dtype=float)
    input_scale = np.asarray(model["input_scale"], dtype=float)
    target_mean = np.asarray(model["target_mean"], dtype=float)
    target_scale = np.asarray(model["target_scale"], dtype=float)

    x_scaled = (x_values - input_mean) / input_scale
    model_type = str(model.get("model_type", "polynomial_ridge_response_surface"))

    if "rbf_kernel_ridge" in model_type:
        train_x_scaled = np.asarray(model["train_x_scaled"], dtype=float)
        dual_coefficients = np.asarray(model["dual_coefficients"], dtype=float)
        kernel = rbf_kernel(x_scaled, train_x_scaled, float(model["length_scale"]))
        transformed_scaled = kernel @ dual_coefficients
    elif "weighted_knn" in model_type:
        transformed_scaled = weighted_knn_predict_scaled(
            x_scaled,
            np.asarray(model["train_x_scaled"], dtype=float),
            np.asarray(model["train_y_scaled"], dtype=float),
            int(model["k"]),
        )
    else:
        coefficients = np.asarray(model["coefficients"], dtype=float)
        powers = [tuple(power) for power in model["basis_powers"]]
        phi = polynomial_basis(x_scaled, powers)
        transformed_scaled = phi @ coefficients

    transformed = transformed_scaled * target_scale + target_mean
    predictions = inverse_targets(transformed, feed_co2=float(model.get("feed_co2_fraction", FEED_CO2_FRACTION)))
    return _apply_prediction_bounds(predictions, model.get("prediction_bounds"))


def objective_scores(predictions: pd.DataFrame, problem: str) -> np.ndarray:
    if problem == "purity_recovery":
        return predictions[["purity", "recovery"]].to_numpy(float)
    if problem == "productivity_energy":
        return np.column_stack(
            [
                predictions["productivity_mol_kg_h"].to_numpy(float),
                -predictions["log_energy"].to_numpy(float),
            ]
        )
    raise ValueError(f"Unknown optimization problem: {problem}")


def nondominated_mask(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    n_rows = scores.shape[0]
    dominated = np.zeros(n_rows, dtype=bool)
    for i in range(n_rows):
        if dominated[i]:
            continue
        better_or_equal = np.all(scores >= scores[i], axis=1)
        strictly_better = np.any(scores > scores[i], axis=1)
        dominated[i] = bool(np.any(better_or_equal & strictly_better))
    return ~dominated


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
