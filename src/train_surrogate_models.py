from __future__ import annotations

import argparse

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
    rbf_kernel,
    regression_metrics,
    save_model,
    squared_euclidean_distances,
    stable_split,
    transform_targets,
    variable_bounds,
    weighted_knn_predict_scaled,
)


ALPHA_GRID = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
RBF_ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
RBF_LENGTH_FACTORS = [0.5, 1.0, 2.0]
KNN_GRID = [5, 10, 20, 40, 80]
SCREENING_TARGETS = ["purity", "recovery", "productivity_mol_kg_h", "log_energy"]


MODEL_LABELS = {
    "poly1_ridge": "Linear ridge",
    "poly2_ridge": "Quadratic ridge",
    "poly3_ridge": "Cubic ridge (GA)",
    "poly4_ridge": "Quartic ridge",
    "rbf_kernel_ridge": "RBF kernel ridge",
    "weighted_knn": "Weighted kNN",
}


def make_folds(train_idx: np.ndarray, seed: int, n_folds: int = 5) -> list[np.ndarray]:
    rng = np.random.default_rng(seed + 101)
    return [np.sort(fold) for fold in np.array_split(rng.permutation(train_idx), n_folds)]


def prediction_bounds(manifest: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        target: {
            "min": float(manifest[target].min()),
            "max": float(manifest[target].max()),
        }
        for target in [*PHYSICAL_TARGETS, "log_energy"]
    }


def common_model_fields(
    model_name: str,
    model_type: str,
    model_family: str,
    seed: int,
    input_mean: np.ndarray,
    input_scale: np.ndarray,
    target_mean: np.ndarray,
    target_scale: np.ndarray,
    bounds: pd.DataFrame,
    manifest: pd.DataFrame,
) -> dict:
    return {
        "model_name": model_name,
        "model_label": MODEL_LABELS.get(model_name, model_name),
        "model_type": model_type,
        "model_family": model_family,
        "random_state": seed,
        "design_cols": DESIGN_COLS,
        "model_targets": MODEL_TARGETS,
        "physical_targets": PHYSICAL_TARGETS,
        "input_mean": input_mean,
        "input_scale": input_scale,
        "target_mean": target_mean,
        "target_scale": target_scale,
        "feed_co2_fraction": FEED_CO2_FRACTION,
        "bounds": bounds.to_dict(orient="records"),
        "prediction_bounds": prediction_bounds(manifest),
    }


def choose_alpha(
    phi: np.ndarray,
    y_transformed: np.ndarray,
    train_idx: np.ndarray,
    target_idx: int,
    seed: int,
    n_folds: int = 5,
) -> float:
    folds = make_folds(train_idx, seed, n_folds=n_folds)
    best_alpha = ALPHA_GRID[0]
    best_mse = np.inf
    for alpha in ALPHA_GRID:
        fold_mse = []
        for fold_idx in range(n_folds):
            val_idx = folds[fold_idx]
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


def fit_polynomial_model(
    degree: int,
    seed: int,
    manifest: pd.DataFrame,
    bounds: pd.DataFrame,
    x_scaled: np.ndarray,
    y_transformed: np.ndarray,
    train_idx: np.ndarray,
    input_mean: np.ndarray,
    input_scale: np.ndarray,
    target_mean: np.ndarray,
    target_scale: np.ndarray,
) -> dict:
    powers = polynomial_powers(n_features=len(DESIGN_COLS), degree=degree)
    phi = polynomial_basis(x_scaled, powers)
    coefficients = np.zeros((phi.shape[1], len(MODEL_TARGETS)), dtype=float)
    selected_alpha: list[float] = []

    for target_idx, target_name in enumerate(MODEL_TARGETS):
        alpha = choose_alpha(phi, y_transformed, train_idx, target_idx, seed + 17 * degree)
        selected_alpha.append(float(alpha))
        y_train = y_transformed[train_idx, target_idx]
        y_train_scaled = (y_train - target_mean[target_idx]) / target_scale[target_idx]
        coefficients[:, target_idx] = fit_ridge(phi[train_idx], y_train_scaled, alpha=alpha)
        print(f"Fitted poly{degree} {target_name}: alpha={alpha:g}")

    model = common_model_fields(
        model_name=f"poly{degree}_ridge",
        model_type="polynomial_ridge_response_surface",
        model_family="polynomial_ridge",
        seed=seed,
        input_mean=input_mean,
        input_scale=input_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        bounds=bounds,
        manifest=manifest,
    )
    model.update(
        {
            "degree": degree,
            "basis_powers": powers,
            "basis_labels": [power_label(power) for power in powers],
            "coefficients": coefficients,
            "selected_alpha": selected_alpha,
            "n_basis_terms": len(powers),
        }
    )
    return model


def median_pair_distance(x_train_scaled: np.ndarray) -> float:
    distances = np.sqrt(squared_euclidean_distances(x_train_scaled, x_train_scaled))
    nonzero = distances[distances > 1e-12]
    if len(nonzero) == 0:
        return 1.0
    return float(np.median(nonzero))


def choose_rbf_hyperparameters(
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    train_idx: np.ndarray,
    seed: int,
    n_folds: int = 5,
) -> tuple[float, float]:
    folds = make_folds(train_idx, seed + 601, n_folds=n_folds)
    base_length = median_pair_distance(x_scaled[train_idx])
    length_grid = [base_length * factor for factor in RBF_LENGTH_FACTORS]
    best_length = length_grid[0]
    best_alpha = RBF_ALPHA_GRID[0]
    best_mse = np.inf

    for length_scale in length_grid:
        for alpha in RBF_ALPHA_GRID:
            fold_mse = []
            for fold_idx in range(n_folds):
                val_idx = folds[fold_idx]
                fit_idx = np.sort(np.concatenate([folds[i] for i in range(n_folds) if i != fold_idx]))
                fit_x = x_scaled[fit_idx]
                kernel_fit = rbf_kernel(fit_x, fit_x, length_scale)
                lhs = kernel_fit + alpha * np.eye(len(fit_idx))
                dual = np.linalg.solve(lhs, y_scaled[fit_idx])
                pred = rbf_kernel(x_scaled[val_idx], fit_x, length_scale) @ dual
                fold_mse.append(float(np.mean((y_scaled[val_idx] - pred) ** 2)))
            mse = float(np.mean(fold_mse))
            if mse < best_mse:
                best_mse = mse
                best_length = float(length_scale)
                best_alpha = float(alpha)
    return best_length, best_alpha


def fit_rbf_kernel_ridge(
    seed: int,
    manifest: pd.DataFrame,
    bounds: pd.DataFrame,
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    train_idx: np.ndarray,
    input_mean: np.ndarray,
    input_scale: np.ndarray,
    target_mean: np.ndarray,
    target_scale: np.ndarray,
) -> dict:
    length_scale, alpha = choose_rbf_hyperparameters(x_scaled, y_scaled, train_idx, seed)
    x_train = x_scaled[train_idx]
    kernel_train = rbf_kernel(x_train, x_train, length_scale)
    dual = np.linalg.solve(kernel_train + alpha * np.eye(len(train_idx)), y_scaled[train_idx])
    print(f"Fitted RBF kernel ridge: length_scale={length_scale:.4g}, alpha={alpha:g}")

    model = common_model_fields(
        model_name="rbf_kernel_ridge",
        model_type="rbf_kernel_ridge",
        model_family="kernel_ridge",
        seed=seed,
        input_mean=input_mean,
        input_scale=input_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        bounds=bounds,
        manifest=manifest,
    )
    model.update(
        {
            "length_scale": float(length_scale),
            "alpha": float(alpha),
            "selected_alpha": [float(alpha)] * len(MODEL_TARGETS),
            "train_x_scaled": x_train,
            "dual_coefficients": dual,
        }
    )
    return model


def choose_knn_k(
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    train_idx: np.ndarray,
    seed: int,
    n_folds: int = 5,
) -> int:
    folds = make_folds(train_idx, seed + 1201, n_folds=n_folds)
    best_k = KNN_GRID[0]
    best_mse = np.inf
    for k in KNN_GRID:
        fold_mse = []
        for fold_idx in range(n_folds):
            val_idx = folds[fold_idx]
            fit_idx = np.sort(np.concatenate([folds[i] for i in range(n_folds) if i != fold_idx]))
            pred = weighted_knn_predict_scaled(
                x_scaled[val_idx],
                x_scaled[fit_idx],
                y_scaled[fit_idx],
                k=k,
            )
            fold_mse.append(float(np.mean((y_scaled[val_idx] - pred) ** 2)))
        mse = float(np.mean(fold_mse))
        if mse < best_mse:
            best_mse = mse
            best_k = int(k)
    return best_k


def fit_weighted_knn(
    seed: int,
    manifest: pd.DataFrame,
    bounds: pd.DataFrame,
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    train_idx: np.ndarray,
    input_mean: np.ndarray,
    input_scale: np.ndarray,
    target_mean: np.ndarray,
    target_scale: np.ndarray,
) -> dict:
    k = choose_knn_k(x_scaled, y_scaled, train_idx, seed)
    print(f"Fitted weighted kNN: k={k}")
    model = common_model_fields(
        model_name="weighted_knn",
        model_type="weighted_knn",
        model_family="local_interpolation",
        seed=seed,
        input_mean=input_mean,
        input_scale=input_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        bounds=bounds,
        manifest=manifest,
    )
    model.update(
        {
            "k": int(k),
            "train_x_scaled": x_scaled[train_idx],
            "train_y_scaled": y_scaled[train_idx],
        }
    )
    return model


def model_setting(model: dict) -> str:
    model_type = str(model.get("model_type", ""))
    if "polynomial" in model_type:
        return f"degree={model['degree']}, terms={model['n_basis_terms']}"
    if "rbf_kernel_ridge" in model_type:
        return f"length_scale={model['length_scale']:.4g}, alpha={model['alpha']:g}"
    if "weighted_knn" in model_type:
        return f"k={model['k']}"
    return ""


def alpha_for_target(model: dict, target: str) -> float:
    selected_alpha = model.get("selected_alpha")
    if not selected_alpha:
        return float("nan")
    target_to_idx = {
        "purity": 0,
        "recovery": 1,
        "productivity_mol_kg_h": 2,
        "energy_kWh_ton": 3,
        "log_energy": 3,
    }
    idx = target_to_idx[target]
    return float(selected_alpha[min(idx, len(selected_alpha) - 1)])


def build_prediction_frame(
    manifest: pd.DataFrame,
    model: dict,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    include_model_columns: bool = False,
) -> pd.DataFrame:
    predictions = predict_surrogate(model, manifest[DESIGN_COLS])
    split = np.full(len(manifest), "unused", dtype=object)
    split[train_idx] = "train"
    split[test_idx] = "test"

    out = manifest[["sample_id", *DESIGN_COLS]].copy()
    if include_model_columns:
        out.insert(0, "model_label", model["model_label"])
        out.insert(0, "model_name", model["model_name"])
    out["split"] = split
    for target in PHYSICAL_TARGETS:
        out[f"actual_{target}"] = manifest[target].to_numpy(float)
        out[f"pred_{target}"] = predictions[target].to_numpy(float)
    out["actual_log_energy"] = manifest["log_energy"].to_numpy(float)
    out["pred_log_energy"] = predictions["log_energy"].to_numpy(float)
    return out


def build_metrics_frame(prediction_frame: pd.DataFrame, model: dict) -> pd.DataFrame:
    rows = []
    for split in ["train", "test"]:
        part = prediction_frame.loc[prediction_frame["split"].eq(split)]
        for target in [*PHYSICAL_TARGETS, "log_energy"]:
            metric = regression_metrics(part[f"actual_{target}"], part[f"pred_{target}"])
            rows.append(
                {
                    "model_name": model["model_name"],
                    "model_label": model["model_label"],
                    "model_family": model["model_family"],
                    "model_setting": model_setting(model),
                    "split": split,
                    "target": target,
                    "rmse": metric["rmse"],
                    "mae": metric["mae"],
                    "r2": metric["r2"],
                    "alpha": alpha_for_target(model, target),
                    "degree": model.get("degree", np.nan),
                    "n_basis_terms": model.get("n_basis_terms", np.nan),
                    "length_scale": model.get("length_scale", np.nan),
                    "k": model.get("k", np.nan),
                }
            )
    return pd.DataFrame(rows)


def build_model_summary(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    test_metrics = metrics_frame.loc[metrics_frame["split"].eq("test")]
    for model_name, group in test_metrics.groupby("model_name", sort=False):
        by_target = group.set_index("target")
        row = {
            "model_name": model_name,
            "model_label": group["model_label"].iloc[0],
            "model_family": group["model_family"].iloc[0],
            "model_setting": group["model_setting"].iloc[0],
            "mean_test_r2_screening_targets": float(by_target.loc[SCREENING_TARGETS, "r2"].mean()),
            "mean_test_r2_physical_targets": float(by_target.loc[PHYSICAL_TARGETS, "r2"].mean()),
            "r2_purity": float(by_target.loc["purity", "r2"]),
            "r2_recovery": float(by_target.loc["recovery", "r2"]),
            "r2_productivity_mol_kg_h": float(by_target.loc["productivity_mol_kg_h", "r2"]),
            "r2_energy_kWh_ton": float(by_target.loc["energy_kWh_ton", "r2"]),
            "r2_log_energy": float(by_target.loc["log_energy", "r2"]),
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary = summary.sort_values("mean_test_r2_screening_targets", ascending=False).reset_index(drop=True)
    summary.insert(0, "rank_screening", np.arange(1, len(summary) + 1))
    return summary


def fit_surrogate_benchmarks(
    degrees: list[int],
    seed: int = RANDOM_STATE,
    include_rbf: bool = True,
    include_knn: bool = True,
) -> tuple[dict[str, dict], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
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

    y_transformed = transform_targets(manifest, feed_co2=FEED_CO2_FRACTION)
    target_mean = y_transformed[train_idx].mean(axis=0)
    target_scale = y_transformed[train_idx].std(axis=0)
    target_scale[(target_scale == 0) | ~np.isfinite(target_scale)] = 1.0
    y_scaled = (y_transformed - target_mean) / target_scale

    models: dict[str, dict] = {}
    for degree in degrees:
        model = fit_polynomial_model(
            degree=degree,
            seed=seed,
            manifest=manifest,
            bounds=bounds,
            x_scaled=x_scaled,
            y_transformed=y_transformed,
            train_idx=train_idx,
            input_mean=input_mean,
            input_scale=input_scale,
            target_mean=target_mean,
            target_scale=target_scale,
        )
        models[model["model_name"]] = model

    if include_rbf:
        model = fit_rbf_kernel_ridge(
            seed=seed,
            manifest=manifest,
            bounds=bounds,
            x_scaled=x_scaled,
            y_scaled=y_scaled,
            train_idx=train_idx,
            input_mean=input_mean,
            input_scale=input_scale,
            target_mean=target_mean,
            target_scale=target_scale,
        )
        models[model["model_name"]] = model

    if include_knn:
        model = fit_weighted_knn(
            seed=seed,
            manifest=manifest,
            bounds=bounds,
            x_scaled=x_scaled,
            y_scaled=y_scaled,
            train_idx=train_idx,
            input_mean=input_mean,
            input_scale=input_scale,
            target_mean=target_mean,
            target_scale=target_scale,
        )
        models[model["model_name"]] = model

    prediction_frames = []
    metric_frames = []
    for model in models.values():
        pred = build_prediction_frame(manifest, model, train_idx, test_idx, include_model_columns=True)
        prediction_frames.append(pred)
        metric_frames.append(build_metrics_frame(pred, model))

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_metrics = pd.concat(metric_frames, ignore_index=True)
    model_summary = build_model_summary(all_metrics)
    return models, all_predictions, all_metrics, model_summary, manifest, train_idx, test_idx


def select_primary_model(models: dict[str, dict], model_summary: pd.DataFrame, primary_model: str) -> dict:
    if primary_model == "auto":
        primary_name = str(model_summary.iloc[0]["model_name"])
    else:
        primary_name = primary_model
    if primary_name not in models:
        valid = ", ".join(models)
        raise ValueError(f"Unknown primary model '{primary_name}'. Valid choices: {valid}, auto")
    return models[primary_name]


def write_primary_coefficients(model: dict) -> None:
    if "basis_powers" not in model:
        return
    rows = []
    for basis_idx, (label, power) in enumerate(zip(model["basis_labels"], model["basis_powers"])):
        for target_idx, target in enumerate(model["model_targets"]):
            row = {
                "model_name": model["model_name"],
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


def save_outputs(
    primary_model: dict,
    primary_predictions: pd.DataFrame,
    primary_metrics: pd.DataFrame,
    all_predictions: pd.DataFrame,
    all_metrics: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> None:
    save_model(primary_model)
    primary_predictions.to_csv(DATA_DIR / "surrogate_predictions.csv", index=False)
    primary_metrics[["split", "target", "rmse", "mae", "r2", "alpha"]].to_csv(DATA_DIR / "surrogate_metrics.csv", index=False)
    all_predictions.to_csv(DATA_DIR / "surrogate_model_predictions.csv", index=False)
    all_metrics.to_csv(DATA_DIR / "surrogate_model_metrics.csv", index=False)
    model_summary.to_csv(DATA_DIR / "surrogate_model_summary.csv", index=False)
    write_primary_coefficients(primary_model)

    bounds = pd.DataFrame(primary_model["bounds"])
    bounds.to_csv(DATA_DIR / "surrogate_variable_bounds.csv", index=False)

    test_metrics = primary_metrics.loc[primary_metrics["split"].eq("test")]
    best_row = model_summary.iloc[0]
    summary = pd.DataFrame(
        [
            {"key": "n_samples", "value": int(len(primary_predictions))},
            {"key": "n_train", "value": int(primary_predictions["split"].eq("train").sum())},
            {"key": "n_test", "value": int(primary_predictions["split"].eq("test").sum())},
            {"key": "primary_model", "value": primary_model["model_name"]},
            {"key": "primary_model_label", "value": primary_model["model_label"]},
            {"key": "primary_model_setting", "value": model_setting(primary_model)},
            {"key": "n_benchmark_models", "value": int(model_summary.shape[0])},
            {"key": "best_screening_model", "value": best_row["model_name"]},
            {"key": "best_screening_model_label", "value": best_row["model_label"]},
            {"key": "best_mean_test_r2_screening_targets", "value": float(best_row["mean_test_r2_screening_targets"])},
            {
                "key": "mean_test_r2_physical_targets",
                "value": float(test_metrics.loc[test_metrics["target"].isin(PHYSICAL_TARGETS), "r2"].mean()),
            },
        ]
    )
    if "degree" in primary_model:
        summary.loc[len(summary)] = {"key": "degree", "value": primary_model["degree"]}
    if "n_basis_terms" in primary_model:
        summary.loc[len(summary)] = {"key": "n_basis_terms", "value": primary_model["n_basis_terms"]}
    summary.to_csv(DATA_DIR / "surrogate_summary.csv", index=False)

    print(f"Saved {DATA_DIR / 'surrogate_model.json'}")
    print(f"Saved {DATA_DIR / 'surrogate_metrics.csv'}")
    print(f"Saved {DATA_DIR / 'surrogate_model_metrics.csv'}")
    print(f"Saved {DATA_DIR / 'surrogate_model_summary.csv'}")
    print(model_summary.to_string(index=False))
    print(test_metrics[["target", "rmse", "mae", "r2"]].to_string(index=False))


def parse_degrees(raw: str, primary_degree: int) -> list[int]:
    degrees = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if primary_degree not in degrees:
        degrees.append(primary_degree)
    return sorted(degrees)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare manifest-only PSA surrogate models.")
    parser.add_argument("--degree", type=int, default=3, help="Primary polynomial degree when --primary-model is omitted.")
    parser.add_argument("--benchmark-degrees", default="1,2,3,4", help="Comma-separated polynomial degrees to benchmark.")
    parser.add_argument("--primary-model", default=None, help="Primary model saved to surrogate_model.json, or 'auto'.")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed for train/test split.")
    parser.add_argument("--no-rbf", action="store_true", help="Skip RBF kernel ridge benchmark.")
    parser.add_argument("--no-knn", action="store_true", help="Skip weighted kNN benchmark.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    degrees = parse_degrees(args.benchmark_degrees, args.degree)
    primary_name = args.primary_model or f"poly{args.degree}_ridge"
    models, all_predictions, all_metrics, model_summary, _, train_idx, test_idx = fit_surrogate_benchmarks(
        degrees=degrees,
        seed=args.seed,
        include_rbf=not args.no_rbf,
        include_knn=not args.no_knn,
    )
    primary_model = select_primary_model(models, model_summary, primary_name)
    primary_predictions = all_predictions.loc[all_predictions["model_name"].eq(primary_model["model_name"])].copy()
    primary_predictions = primary_predictions.drop(columns=["model_name", "model_label"])
    primary_metrics = all_metrics.loc[all_metrics["model_name"].eq(primary_model["model_name"])].copy()
    # Preserve the split produced during fitting even if the primary model is selected after benchmarking.
    primary_predictions["split"] = np.where(
        primary_predictions.index.isin(train_idx),
        "train",
        np.where(primary_predictions.index.isin(test_idx), "test", primary_predictions["split"]),
    )
    save_outputs(primary_model, primary_predictions, primary_metrics, all_predictions, all_metrics, model_summary)


if __name__ == "__main__":
    main()
