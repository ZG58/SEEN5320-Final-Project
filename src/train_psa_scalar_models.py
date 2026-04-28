from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

from psa_project_utils import (
    DATA_DIR,
    INPUT_COLS,
    MODEL_TARGETS,
    RANDOM_STATE,
    SCALAR_CSV_PATH,
    TARGET_COLS,
    clip_to_observed_bounds,
    empirical_pareto_masks,
    input_bounds_frame,
    inverse_transformed_targets,
    load_scalar_samples,
    regression_metrics,
    transformed_targets,
    validate_scalar_samples,
)


RANK_TARGETS = ["recovery", "purity", "productivity_mol_h_kg", "log_energy"]


def mean_normalized_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))
    scale = np.ptp(y_true, axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return float(np.mean(rmse / scale))


def build_estimator(regressor) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=regressor,
        transformer=StandardScaler(),
        check_inverse=False,
    )


def model_specs(seed: int) -> list[dict]:
    return [
        {
            "model_name": "ridge",
            "model_label": "Ridge",
            "estimator": build_estimator(
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("model", Ridge()),
                    ]
                )
            ),
            "param_grid": {"regressor__model__alpha": [1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0]},
        },
        {
            "model_name": "polynomial_ridge",
            "model_label": "Polynomial ridge",
            "estimator": build_estimator(
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("poly", PolynomialFeatures(include_bias=False)),
                        ("model", Ridge()),
                    ]
                )
            ),
            "param_grid": {
                "regressor__poly__degree": [2, 3, 4],
                "regressor__model__alpha": [1e-3, 1e-1, 1.0, 10.0, 100.0],
            },
        },
        {
            "model_name": "kernel_ridge_rbf",
            "model_label": "RBF kernel ridge",
            "estimator": build_estimator(
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("model", KernelRidge(kernel="rbf")),
                    ]
                )
            ),
            "param_grid": {
                "regressor__model__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
                "regressor__model__gamma": [0.03, 0.10, 0.30, 1.0],
            },
        },
        {
            "model_name": "svr_rbf",
            "model_label": "RBF SVR",
            "estimator": build_estimator(
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("model", MultiOutputRegressor(SVR(kernel="rbf"))),
                    ]
                )
            ),
            "param_grid": {
                "regressor__model__estimator__C": [10.0, 100.0],
                "regressor__model__estimator__epsilon": [0.001, 0.01, 0.05],
                "regressor__model__estimator__gamma": ["scale", 0.1, 0.3],
            },
        },
        {
            "model_name": "random_forest",
            "model_label": "Random forest",
            "estimator": build_estimator(
                RandomForestRegressor(
                    random_state=seed,
                    n_jobs=1,
                    bootstrap=True,
                )
            ),
            "param_grid": {
                "regressor__n_estimators": [240],
                "regressor__max_depth": [None, 10, 18],
                "regressor__min_samples_leaf": [1, 3],
                "regressor__max_features": ["sqrt", 1.0],
            },
        },
        {
            "model_name": "extra_trees",
            "model_label": "Extra trees",
            "estimator": build_estimator(
                ExtraTreesRegressor(
                    random_state=seed,
                    n_jobs=1,
                    bootstrap=False,
                )
            ),
            "param_grid": {
                "regressor__n_estimators": [300],
                "regressor__max_depth": [None, 12, 20],
                "regressor__min_samples_leaf": [1, 3],
                "regressor__max_features": ["sqrt", 1.0],
            },
        },
        {
            "model_name": "hist_gradient_boosting",
            "model_label": "Gradient boosting",
            "estimator": build_estimator(
                MultiOutputRegressor(
                    HistGradientBoostingRegressor(
                        loss="squared_error",
                        random_state=seed,
                    )
                )
            ),
            "param_grid": {
                "regressor__estimator__max_iter": [160, 320],
                "regressor__estimator__learning_rate": [0.03, 0.08],
                "regressor__estimator__max_leaf_nodes": [15, 31],
                "regressor__estimator__l2_regularization": [0.0, 1e-3],
            },
        },
        {
            "model_name": "mlp",
            "model_label": "Neural network",
            "estimator": build_estimator(
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "model",
                            MLPRegressor(
                                random_state=seed,
                                max_iter=900,
                                early_stopping=True,
                                validation_fraction=0.15,
                                n_iter_no_change=35,
                            ),
                        ),
                    ]
                )
            ),
            "param_grid": {
                "regressor__model__hidden_layer_sizes": [(64, 64), (128, 64), (96, 64, 32)],
                "regressor__model__alpha": [1e-4, 1e-3],
                "regressor__model__learning_rate_init": [5e-4, 1e-3],
            },
        },
    ]


def train_test_split_indices(n_rows: int, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_rows)
    n_test = int(round(n_rows * test_size))
    test_idx = np.sort(shuffled[:n_test])
    train_idx = np.sort(shuffled[n_test:])
    return train_idx, test_idx


def prediction_frame(frame: pd.DataFrame, model_name: str, model_label: str, estimator, split: np.ndarray) -> pd.DataFrame:
    raw_prediction = estimator.predict(frame[INPUT_COLS])
    physical = inverse_transformed_targets(raw_prediction)
    physical = clip_to_observed_bounds(physical, frame)

    out = frame[["sample_id", *INPUT_COLS]].copy()
    out.insert(1, "split", split)
    out.insert(2, "model_name", model_name)
    out.insert(3, "model_label", model_label)
    for col in [*TARGET_COLS, "log_energy"]:
        out[f"actual_{col}"] = frame[col].to_numpy(float)
        out[f"pred_{col}"] = physical[col].to_numpy(float)
    return out


def build_metrics(
    predictions: pd.DataFrame,
    model_name: str,
    model_label: str,
    masks: dict[str, np.ndarray],
) -> pd.DataFrame:
    split = predictions["split"].to_numpy()
    subset_masks = {
        "test_global": split == "test",
        "test_purity_recovery_near": (split == "test") & masks["purity_recovery_near"],
        "test_productivity_energy_near": (split == "test") & masks["productivity_energy_near"],
        "test_pareto_focus": (split == "test") & masks["pareto_focus"],
        "train_global": split == "train",
    }
    rows = []
    for subset, mask in subset_masks.items():
        for target in [*TARGET_COLS, "log_energy"]:
            if not mask.any():
                metric = {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "normalized_rmse": np.nan}
                n_rows = 0
            else:
                metric = regression_metrics(
                    predictions.loc[mask, f"actual_{target}"].to_numpy(float),
                    predictions.loc[mask, f"pred_{target}"].to_numpy(float),
                )
                n_rows = int(mask.sum())
            rows.append(
                {
                    "model_name": model_name,
                    "model_label": model_label,
                    "subset": subset,
                    "target": target,
                    "n": n_rows,
                    **metric,
                }
            )
    return pd.DataFrame(rows)


def build_model_summary(metrics: pd.DataFrame, search_rows: list[dict]) -> pd.DataFrame:
    search = pd.DataFrame(search_rows)
    rows = []
    for (model_name, model_label), group in metrics.groupby(["model_name", "model_label"], sort=False):
        global_norm = group.loc[
            group["subset"].eq("test_global") & group["target"].isin(RANK_TARGETS), "normalized_rmse"
        ].mean()
        pareto_norm = group.loc[
            group["subset"].eq("test_pareto_focus") & group["target"].isin(RANK_TARGETS), "normalized_rmse"
        ].mean()
        if not np.isfinite(pareto_norm):
            pareto_norm = global_norm
        mean_r2 = group.loc[group["subset"].eq("test_global") & group["target"].isin(RANK_TARGETS), "r2"].mean()
        selection_score = 0.6 * pareto_norm + 0.4 * global_norm
        best = search.loc[search["model_name"].eq(model_name)].iloc[0]
        rows.append(
            {
                "model_name": model_name,
                "model_label": model_label,
                "global_norm_rmse": float(global_norm),
                "pareto_norm_rmse": float(pareto_norm),
                "selection_score": float(selection_score),
                "mean_test_r2": float(mean_r2),
                "best_cv_norm_rmse": float(best["best_cv_norm_rmse"]),
                "best_params": best["best_params"],
            }
        )
    summary = pd.DataFrame(rows).sort_values("selection_score", ascending=True).reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary


def save_pareto_sets(frame: pd.DataFrame, masks: dict[str, np.ndarray]) -> None:
    pieces = []
    for problem, front_key, near_key in [
        ("purity_recovery", "purity_recovery_front", "purity_recovery_near"),
        ("productivity_energy", "productivity_energy_front", "productivity_energy_near"),
    ]:
        for set_type, key in [("front", front_key), ("near", near_key)]:
            subset = frame.loc[masks[key], ["sample_id", *INPUT_COLS, *TARGET_COLS, "log_energy"]].copy()
            subset.insert(1, "problem", problem)
            subset.insert(2, "set_type", set_type)
            pieces.append(subset)
    pd.concat(pieces, ignore_index=True).to_csv(DATA_DIR / "psa_empirical_pareto_sets.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and tune scalar PSA surrogate models.")
    parser.add_argument("--csv", default=str(SCALAR_CSV_PATH), help="Converted scalar CSV path.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Held-out test fraction.")
    parser.add_argument("--cv", type=int, default=3, help="Cross-validation folds for hyperparameter search.")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed.")
    parser.add_argument("--n-jobs", type=int, default=1, help="GridSearchCV parallel jobs.")
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    args = parse_args()
    frame = load_scalar_samples(Path(args.csv))
    validate_scalar_samples(frame)
    masks = empirical_pareto_masks(frame)
    save_pareto_sets(frame, masks)

    train_idx, test_idx = train_test_split_indices(len(frame), args.test_size, args.seed)
    split = np.full(len(frame), "unused", dtype=object)
    split[train_idx] = "train"
    split[test_idx] = "test"

    x = frame[INPUT_COLS]
    y = transformed_targets(frame)
    scorer = make_scorer(mean_normalized_rmse, greater_is_better=False)
    cv = KFold(n_splits=args.cv, shuffle=True, random_state=args.seed)

    predictions = []
    metrics = []
    search_rows = []
    cv_rows = []

    for spec in model_specs(args.seed):
        print(f"Tuning {spec['model_label']}...")
        grid = GridSearchCV(
            estimator=spec["estimator"],
            param_grid=spec["param_grid"],
            scoring=scorer,
            cv=cv,
            n_jobs=args.n_jobs,
            refit=True,
            return_train_score=False,
            error_score="raise",
        )
        grid.fit(x.iloc[train_idx], y[train_idx])
        best_norm_rmse = -float(grid.best_score_)
        best_params = json.dumps(grid.best_params_, sort_keys=True)
        search_rows.append(
            {
                "model_name": spec["model_name"],
                "model_label": spec["model_label"],
                "best_cv_norm_rmse": best_norm_rmse,
                "best_params": best_params,
            }
        )
        cv_result = pd.DataFrame(grid.cv_results_)
        for _, row in cv_result.iterrows():
            cv_rows.append(
                {
                    "model_name": spec["model_name"],
                    "model_label": spec["model_label"],
                    "rank_test_score": int(row["rank_test_score"]),
                    "mean_cv_norm_rmse": -float(row["mean_test_score"]),
                    "std_cv_norm_rmse": float(row["std_test_score"]),
                    "params": json.dumps(row["params"], sort_keys=True),
                }
            )

        pred = prediction_frame(frame, spec["model_name"], spec["model_label"], grid.best_estimator_, split)
        predictions.append(pred)
        metrics.append(build_metrics(pred, spec["model_name"], spec["model_label"], masks))
        print(f"  best CV normalized RMSE={best_norm_rmse:.4f}; params={best_params}")

    all_predictions = pd.concat(predictions, ignore_index=True)
    all_metrics = pd.concat(metrics, ignore_index=True)
    model_summary = build_model_summary(all_metrics, search_rows)
    best_name = str(model_summary.iloc[0]["model_name"])
    best_prediction = all_predictions.loc[all_predictions["model_name"].eq(best_name)].copy()
    best_metrics = all_metrics.loc[all_metrics["model_name"].eq(best_name)].copy()

    # Refit the selected model on the training split only and save the exact estimator selected by GridSearchCV.
    best_spec = next(spec for spec in model_specs(args.seed) if spec["model_name"] == best_name)
    best_params = json.loads(str(model_summary.iloc[0]["best_params"]))
    selected_estimator = best_spec["estimator"].set_params(**best_params)
    selected_estimator.fit(x.iloc[train_idx], y[train_idx])

    artifact = {
        "model_name": best_name,
        "model_label": str(model_summary.iloc[0]["model_label"]),
        "estimator": selected_estimator,
        "input_cols": INPUT_COLS,
        "target_cols": TARGET_COLS,
        "model_targets": MODEL_TARGETS,
        "input_bounds": input_bounds_frame(),
        "observed_outputs": frame[[*TARGET_COLS, "log_energy"]].agg(["min", "max"]).to_dict(),
        "selection_score": float(model_summary.iloc[0]["selection_score"]),
        "random_state": args.seed,
    }

    joblib.dump(artifact, DATA_DIR / "psa_best_surrogate.joblib")
    all_predictions.to_csv(DATA_DIR / "psa_model_predictions.csv", index=False)
    best_prediction.to_csv(DATA_DIR / "psa_best_model_predictions.csv", index=False)
    all_metrics.to_csv(DATA_DIR / "psa_model_metrics.csv", index=False)
    best_metrics.to_csv(DATA_DIR / "psa_best_model_metrics.csv", index=False)
    model_summary.to_csv(DATA_DIR / "psa_model_summary.csv", index=False)
    pd.DataFrame(search_rows).to_csv(DATA_DIR / "psa_best_hyperparameters.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(DATA_DIR / "psa_hyperparameter_search.csv", index=False)
    pd.DataFrame({"sample_id": frame["sample_id"], "split": split}).to_csv(DATA_DIR / "psa_train_test_split.csv", index=False)

    print(f"Saved {DATA_DIR / 'psa_best_surrogate.joblib'}")
    print(model_summary[["rank", "model_label", "selection_score", "global_norm_rmse", "pareto_norm_rmse", "mean_test_r2"]].to_string(index=False))
    print(best_metrics.loc[best_metrics["subset"].eq("test_global"), ["target", "rmse", "mae", "r2"]].to_string(index=False))


if __name__ == "__main__":
    main()
