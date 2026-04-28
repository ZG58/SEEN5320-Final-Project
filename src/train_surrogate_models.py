from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from psa_surrogate_utils import (
    DATA_DIR,
    DESIGN_COLS,
    FEED_CO2_FRACTION,
    MODEL_TARGETS,
    PHYSICAL_TARGETS,
    RANDOM_STATE,
    bounds_arrays,
    fit_ridge,
    load_manifest,
    polynomial_basis,
    polynomial_powers,
    power_label,
    predict_surrogate,
    regression_metrics,
    save_model,
    stable_split,
    transform_targets,
    variable_bounds,
)


ALPHA_GRID = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


def choose_alpha(
    phi: np.ndarray,
    y_transformed: np.ndarray,
    train_idx: np.ndarray,
    target_idx: int,
    seed: int,
    n_folds: int = 5,
) -> float:
    rng = np.random.default_rng(seed + 101)
    folds = np.array_split(rng.permutation(train_idx), n_folds)
    best_alpha = ALPHA_GRID[0]
    best_mse = np.inf
    for alpha in ALPHA_GRID:
        fold_mse = []
        for fold_idx in range(n_folds):
            val_idx = np.sort(folds[fold_idx])
            fit_idx = np.sort(np.concatenate([folds[i] for i in range(n_folds) if i != fold_idx]))
            y_fit = y_transformed[fit_idx, target_idx]
            mean = float(np.mean(y_fit))
            scale = float(np.std(y_fit))
            if scale == 0 or not np.isfinite(scale):
                scale = 1.0
            y_fit_scaled = (y_fit - mean) / scale
            coef = fit_ridge(phi[fit_idx], y_fit_scaled, alpha=alpha)
            pred = (phi[val_idx] @ coef) * scale + mean
            fold_mse.append(float(np.mean((y_transformed[val_idx, target_idx] - pred) ** 2)))
        mse = float(np.mean(fold_mse))
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
    return best_alpha


def fit_surrogate(degree: int = 3, seed: int = RANDOM_STATE) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    manifest = load_manifest()
    bounds = variable_bounds(manifest)
    lower, upper = bounds_arrays(bounds)
    out_of_bounds = ((manifest[DESIGN_COLS].to_numpy(float) < lower) | (manifest[DESIGN_COLS].to_numpy(float) > upper)).any()
    if out_of_bounds:
        raise ValueError("At least one manifest design point lies outside the configured bounds.")

    train_idx, test_idx = stable_split(len(manifest), test_size=0.2, seed=seed)
    x = manifest[DESIGN_COLS].to_numpy(float)
    input_mean = x[train_idx].mean(axis=0)
    input_scale = x[train_idx].std(axis=0)
    input_scale[input_scale == 0] = 1.0
    x_scaled = (x - input_mean) / input_scale

    powers = polynomial_powers(n_features=len(DESIGN_COLS), degree=degree)
    phi = polynomial_basis(x_scaled, powers)
    y_transformed = transform_targets(manifest, feed_co2=FEED_CO2_FRACTION)

    coefficients = np.zeros((phi.shape[1], len(MODEL_TARGETS)), dtype=float)
    target_mean = np.zeros(len(MODEL_TARGETS), dtype=float)
    target_scale = np.ones(len(MODEL_TARGETS), dtype=float)
    selected_alpha: list[float] = []

    for target_idx, target_name in enumerate(MODEL_TARGETS):
        alpha = choose_alpha(phi, y_transformed, train_idx, target_idx, seed)
        selected_alpha.append(float(alpha))
        y_train = y_transformed[train_idx, target_idx]
        target_mean[target_idx] = float(np.mean(y_train))
        target_scale[target_idx] = float(np.std(y_train))
        if target_scale[target_idx] == 0 or not np.isfinite(target_scale[target_idx]):
            target_scale[target_idx] = 1.0
        y_train_scaled = (y_train - target_mean[target_idx]) / target_scale[target_idx]
        coefficients[:, target_idx] = fit_ridge(phi[train_idx], y_train_scaled, alpha=alpha)
        print(f"Fitted {target_name}: alpha={alpha:g}")

    model = {
        "model_type": "degree_3_ridge_response_surface",
        "degree": degree,
        "random_state": seed,
        "design_cols": DESIGN_COLS,
        "model_targets": MODEL_TARGETS,
        "physical_targets": PHYSICAL_TARGETS,
        "basis_powers": powers,
        "basis_labels": [power_label(power) for power in powers],
        "input_mean": input_mean,
        "input_scale": input_scale,
        "target_mean": target_mean,
        "target_scale": target_scale,
        "coefficients": coefficients,
        "selected_alpha": selected_alpha,
        "feed_co2_fraction": FEED_CO2_FRACTION,
        "bounds": bounds.to_dict(orient="records"),
        "prediction_bounds": {
            target: {
                "min": float(manifest[target].min()),
                "max": float(manifest[target].max()),
            }
            for target in [*PHYSICAL_TARGETS, "log_energy"]
        },
    }

    prediction_frame = build_prediction_frame(manifest, model, train_idx, test_idx)
    metrics_frame = build_metrics_frame(prediction_frame, selected_alpha)
    return model, prediction_frame, metrics_frame


def build_prediction_frame(
    manifest: pd.DataFrame,
    model: dict,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> pd.DataFrame:
    predictions = predict_surrogate(model, manifest[DESIGN_COLS])
    split = np.full(len(manifest), "unused", dtype=object)
    split[train_idx] = "train"
    split[test_idx] = "test"

    out = manifest[["sample_id", *DESIGN_COLS]].copy()
    out["split"] = split
    for target in PHYSICAL_TARGETS:
        out[f"actual_{target}"] = manifest[target].to_numpy(float)
        out[f"pred_{target}"] = predictions[target].to_numpy(float)
    out["actual_log_energy"] = manifest["log_energy"].to_numpy(float)
    out["pred_log_energy"] = predictions["log_energy"].to_numpy(float)
    return out


def build_metrics_frame(prediction_frame: pd.DataFrame, selected_alpha: list[float]) -> pd.DataFrame:
    alpha_by_target = {
        "purity": selected_alpha[0],
        "recovery": selected_alpha[1],
        "productivity_mol_kg_h": selected_alpha[2],
        "energy_kWh_ton": selected_alpha[3],
        "log_energy": selected_alpha[3],
    }
    rows = []
    for split in ["train", "test"]:
        part = prediction_frame.loc[prediction_frame["split"].eq(split)]
        for target in [*PHYSICAL_TARGETS, "log_energy"]:
            metric = regression_metrics(part[f"actual_{target}"], part[f"pred_{target}"])
            rows.append(
                {
                    "split": split,
                    "target": target,
                    "rmse": metric["rmse"],
                    "mae": metric["mae"],
                    "r2": metric["r2"],
                    "alpha": alpha_by_target[target],
                }
            )
    return pd.DataFrame(rows)


def save_outputs(model: dict, prediction_frame: pd.DataFrame, metrics_frame: pd.DataFrame) -> None:
    save_model(model)
    prediction_frame.to_csv(DATA_DIR / "surrogate_predictions.csv", index=False)
    metrics_frame.to_csv(DATA_DIR / "surrogate_metrics.csv", index=False)

    rows = []
    for basis_idx, (label, power) in enumerate(zip(model["basis_labels"], model["basis_powers"])):
        for target_idx, target in enumerate(model["model_targets"]):
            row = {
                "basis_index": basis_idx,
                "basis_label": label,
                "target": target,
                "coefficient_standardized": float(np.asarray(model["coefficients"])[basis_idx, target_idx]),
                "alpha": float(model["selected_alpha"][target_idx]),
            }
            for col, exponent in zip(DESIGN_COLS, power):
                row[f"power_{col}"] = int(exponent)
            rows.append(row)
    pd.DataFrame(rows).to_csv(DATA_DIR / "surrogate_coefficients.csv", index=False)

    bounds = pd.DataFrame(model["bounds"])
    bounds.to_csv(DATA_DIR / "surrogate_variable_bounds.csv", index=False)

    test_metrics = metrics_frame.loc[metrics_frame["split"].eq("test")]
    summary = pd.DataFrame(
        [
            {"key": "n_samples", "value": len(prediction_frame)},
            {"key": "n_train", "value": int(prediction_frame["split"].eq("train").sum())},
            {"key": "n_test", "value": int(prediction_frame["split"].eq("test").sum())},
            {"key": "degree", "value": model["degree"]},
            {"key": "n_basis_terms", "value": len(model["basis_labels"])},
            {"key": "mean_test_r2_physical_targets", "value": float(test_metrics.loc[test_metrics["target"].isin(PHYSICAL_TARGETS), "r2"].mean())},
        ]
    )
    summary.to_csv(DATA_DIR / "surrogate_summary.csv", index=False)
    print(f"Saved {DATA_DIR / 'surrogate_model.json'}")
    print(f"Saved {DATA_DIR / 'surrogate_metrics.csv'}")
    print(test_metrics.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train manifest-only PSA surrogate models.")
    parser.add_argument("--degree", type=int, default=3, help="Polynomial response-surface degree.")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed for train/test split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, predictions, metrics = fit_surrogate(degree=args.degree, seed=args.seed)
    save_outputs(model, predictions, metrics)


if __name__ == "__main__":
    main()
