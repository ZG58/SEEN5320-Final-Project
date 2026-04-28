from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from psa_ml_utils import (
    DATA_DIR,
    DESIGN_COLS,
    METRIC_COLS,
    REGRESSION_TARGETS,
    load_manifest,
)


RANDOM_STATE = 42
TEST_SIZE = 0.20


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def regression_models() -> dict[str, BaseEstimator]:
    return {
        "Ridge": TransformedTargetRegressor(
            regressor=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=1.0)),
                ]
            ),
            transformer=StandardScaler(),
        ),
        "Random forest": TransformedTargetRegressor(
            regressor=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=260,
                            min_samples_leaf=3,
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            transformer=StandardScaler(),
        ),
        "Gradient boosting": TransformedTargetRegressor(
            regressor=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        MultiOutputRegressor(
                            GradientBoostingRegressor(
                                n_estimators=350,
                                learning_rate=0.04,
                                max_depth=3,
                                subsample=0.9,
                                random_state=RANDOM_STATE,
                            )
                        ),
                    ),
                ]
            ),
            transformer=StandardScaler(),
        ),
        "SVR": TransformedTargetRegressor(
            regressor=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", MultiOutputRegressor(SVR(C=8.0, epsilon=0.03, kernel="rbf"))),
                ]
            ),
            transformer=StandardScaler(),
        ),
        "MLP": TransformedTargetRegressor(
            regressor=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPRegressor(
                            hidden_layer_sizes=(80, 40),
                            alpha=1e-4,
                            early_stopping=True,
                            max_iter=900,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            transformer=StandardScaler(),
        ),
    }


def classifier_models() -> dict[str, BaseEstimator]:
    return {
        "Logistic regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVC(C=2.0, kernel="rbf", class_weight="balanced", probability=True)),
            ]
        ),
        "Random forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=260,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def load_dataset() -> tuple[pd.DataFrame, list[str]]:
    features_path = DATA_DIR / "profile_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(
            f"Missing {features_path}. Run src/build_profile_features.py before training."
        )

    manifest = load_manifest()
    profile_features = pd.read_csv(features_path)
    dataset = manifest.merge(profile_features, on="sample_id", how="inner", validate="one_to_one")
    if len(dataset) != len(manifest):
        raise ValueError(f"Merged dataset has {len(dataset)} rows; expected {len(manifest)}.")

    profile_feature_cols = [col for col in profile_features.columns if col != "sample_id"]
    numeric_cols = [*DESIGN_COLS, *METRIC_COLS, "log_energy", "balanced_score", *profile_feature_cols]
    for col in numeric_cols:
        dataset[col] = pd.to_numeric(dataset[col], errors="coerce")
    dataset[numeric_cols] = dataset[numeric_cols].replace([np.inf, -np.inf], np.nan)
    dataset[numeric_cols] = dataset[numeric_cols].fillna(dataset[numeric_cols].median(numeric_only=True))

    out_path = DATA_DIR / "ml_dataset.csv"
    dataset.to_csv(out_path, index=False)
    print(f"Saved {out_path} with shape {dataset.shape}")
    return dataset, profile_feature_cols


def evaluate_regression(dataset: pd.DataFrame, profile_feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = dataset[REGRESSION_TARGETS].copy()
    sample_ids = dataset["sample_id"].copy()
    feature_sets = {
        "design_only": DESIGN_COLS,
        "design_plus_profile": [*DESIGN_COLS, *profile_feature_cols],
    }

    metric_rows: list[dict[str, float | str]] = []
    fitted: dict[tuple[str, str], BaseEstimator] = {}
    test_cache: dict[str, tuple[pd.Series, pd.DataFrame, pd.DataFrame]] = {}

    for feature_set_name, columns in feature_sets.items():
        X = dataset[columns].copy()
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, sample_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        test_cache[feature_set_name] = (ids_test, y_test, X_test)

        for model_name, model in regression_models().items():
            print(f"Training regression model: {feature_set_name} / {model_name}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            pred = pd.DataFrame(model.predict(X_test), columns=REGRESSION_TARGETS, index=y_test.index)
            fitted[(feature_set_name, model_name)] = model

            for target in REGRESSION_TARGETS:
                metric_rows.append(
                    {
                        "feature_set": feature_set_name,
                        "model": model_name,
                        "target": target,
                        "rmse": rmse(y_test[target].to_numpy(), pred[target].to_numpy()),
                        "mae": float(mean_absolute_error(y_test[target], pred[target])),
                        "r2": float(r2_score(y_test[target], pred[target])),
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(DATA_DIR / "regression_metrics.csv", index=False)

    summary = (
        metrics.groupby(["feature_set", "model"], as_index=False)
        .agg(mean_r2=("r2", "mean"), mean_rmse=("rmse", "mean"), mean_mae=("mae", "mean"))
        .sort_values("mean_r2", ascending=False)
    )
    summary.to_csv(DATA_DIR / "regression_metrics_summary.csv", index=False)

    best_row = summary.iloc[0]
    best_key = (str(best_row["feature_set"]), str(best_row["model"]))
    ids_test, y_test, X_test = test_cache[best_key[0]]
    best_model = fitted[best_key]
    pred = pd.DataFrame(best_model.predict(X_test), columns=REGRESSION_TARGETS, index=y_test.index)
    joblib.dump(
        {
            "model": best_model,
            "feature_set": best_key[0],
            "model_name": best_key[1],
            "feature_columns": feature_sets[best_key[0]],
            "targets": REGRESSION_TARGETS,
        },
        DATA_DIR / "best_regression_model.joblib",
    )

    pred_out = pd.DataFrame({"sample_id": ids_test.to_numpy(), "model": best_key[1], "feature_set": best_key[0]})
    for target in REGRESSION_TARGETS:
        pred_out[f"actual_{target}"] = y_test[target].to_numpy()
        pred_out[f"pred_{target}"] = pred[target].to_numpy()
    pred_out["actual_energy_kWh_ton"] = np.power(10.0, pred_out["actual_log_energy"])
    pred_out["pred_energy_kWh_ton"] = np.power(10.0, pred_out["pred_log_energy"])
    pred_out.to_csv(DATA_DIR / "regression_predictions.csv", index=False)

    rf_key = ("design_plus_profile", "Random forest")
    rf_model = fitted[rf_key]
    rf = rf_model.regressor_.named_steps["model"]
    importances = pd.DataFrame(
        {
            "feature": [*DESIGN_COLS, *profile_feature_cols],
            "importance": rf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importances.to_csv(DATA_DIR / "feature_importance.csv", index=False)
    return metrics, summary


def evaluate_classification(dataset: pd.DataFrame, profile_feature_cols: list[str]) -> pd.DataFrame:
    columns = [*DESIGN_COLS, *profile_feature_cols]
    X = dataset[columns].copy()
    y = dataset["high_performer"].astype(int)
    sample_ids = dataset["sample_id"].copy()

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, sample_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    rows: list[dict[str, float | str]] = []
    predictions: dict[str, pd.DataFrame] = {}
    fitted_models: dict[str, BaseEstimator] = {}
    for model_name, model in classifier_models().items():
        print(f"Training classifier: {model_name}")
        model.fit(X_train, y_train)
        fitted_models[model_name] = model
        pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X_test)[:, 1]
        else:
            score = model.decision_function(X_test)

        rows.append(
            {
                "model": model_name,
                "accuracy": float(accuracy_score(y_test, pred)),
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "f1": float(f1_score(y_test, pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, score)),
            }
        )
        predictions[model_name] = pd.DataFrame(
            {
                "sample_id": ids_test.to_numpy(),
                "actual_high_performer": y_test.to_numpy(),
                "pred_high_performer": pred,
                "score": score,
            }
        )

    metrics = pd.DataFrame(rows).sort_values("f1", ascending=False)
    metrics.to_csv(DATA_DIR / "classification_metrics.csv", index=False)
    best_model = str(metrics.iloc[0]["model"])
    predictions[best_model].to_csv(DATA_DIR / "classification_predictions.csv", index=False)
    joblib.dump(
        {
            "model": fitted_models[best_model],
            "model_name": best_model,
            "feature_columns": columns,
            "target": "high_performer",
        },
        DATA_DIR / "best_classifier_model.joblib",
    )

    cm = confusion_matrix(
        predictions[best_model]["actual_high_performer"],
        predictions[best_model]["pred_high_performer"],
        labels=[0, 1],
    )
    cm_out = pd.DataFrame(cm, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"])
    cm_out.to_csv(DATA_DIR / "classification_confusion_matrix.csv")
    return metrics


def run_pca_clustering(dataset: pd.DataFrame, profile_feature_cols: list[str]) -> pd.DataFrame:
    cluster_cols = [*DESIGN_COLS, *METRIC_COLS, *profile_feature_cols]
    preprocess = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    X_scaled = preprocess.fit_transform(dataset[cluster_cols])
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(X_scaled)
    clusters = KMeans(n_clusters=4, n_init=50, random_state=RANDOM_STATE).fit_predict(X_scaled)

    assignments = dataset[["sample_id", "balanced_score", "is_pareto", "high_performer", *METRIC_COLS]].copy()
    assignments["pc1"] = pcs[:, 0]
    assignments["pc2"] = pcs[:, 1]
    assignments["cluster"] = clusters
    assignments.to_csv(DATA_DIR / "cluster_assignments.csv", index=False)

    loading_rows: list[dict[str, float | str]] = []
    for component_idx, component_name in enumerate(["PC1", "PC2"]):
        for feature, loading in zip(cluster_cols, pca.components_[component_idx]):
            loading_rows.append(
                {
                    "component": component_name,
                    "feature": feature,
                    "loading": float(loading),
                    "abs_loading": float(abs(loading)),
                    "explained_variance_ratio": float(pca.explained_variance_ratio_[component_idx]),
                }
            )
    pd.DataFrame(loading_rows).sort_values(["component", "abs_loading"], ascending=[True, False]).to_csv(
        DATA_DIR / "pca_loadings.csv", index=False
    )
    return assignments


def save_summary(
    dataset: pd.DataFrame,
    profile_feature_cols: list[str],
    regression_summary: pd.DataFrame,
    classification_metrics: pd.DataFrame,
) -> None:
    best_reg = regression_summary.iloc[0]
    best_cls = classification_metrics.iloc[0]
    summary = pd.DataFrame(
        [
            ("n_samples", len(dataset)),
            ("n_profile_features", len(profile_feature_cols)),
            ("n_pareto_samples", int(dataset["is_pareto"].sum())),
            ("n_high_performers", int(dataset["high_performer"].sum())),
            ("purity_recovery_spearman", dataset[["purity", "recovery"]].corr(method="spearman").iloc[0, 1]),
            ("best_regression_feature_set", best_reg["feature_set"]),
            ("best_regression_model", best_reg["model"]),
            ("best_regression_mean_r2", best_reg["mean_r2"]),
            ("best_classifier_model", best_cls["model"]),
            ("best_classifier_f1", best_cls["f1"]),
            ("best_classifier_roc_auc", best_cls["roc_auc"]),
        ],
        columns=["key", "value"],
    )
    summary.to_csv(DATA_DIR / "ml_summary.csv", index=False)
    print(summary.to_string(index=False))


def main() -> None:
    dataset, profile_feature_cols = load_dataset()
    regression_metrics, regression_summary = evaluate_regression(dataset, profile_feature_cols)
    classification_metrics = evaluate_classification(dataset, profile_feature_cols)
    run_pca_clustering(dataset, profile_feature_cols)
    save_summary(dataset, profile_feature_cols, regression_summary, classification_metrics)


if __name__ == "__main__":
    main()
