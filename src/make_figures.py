from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from psa_ml_utils import (
    DATA_DIR,
    METRIC_LABELS,
    REGRESSION_TARGETS,
    REPORT_ML_FIG_DIR,
    SLIDE_ML_FIG_DIR,
    ensure_output_dirs,
    load_manifest,
    profile_path,
)


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)


def save_dual(fig: plt.Figure, stem: str) -> None:
    ensure_output_dirs()
    png = SLIDE_ML_FIG_DIR / f"{stem}.png"
    pdf = REPORT_ML_FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved {png}")
    print(f"Saved {pdf}")
    plt.close(fig)


def plot_regression_predictions() -> None:
    pred = pd.read_csv(DATA_DIR / "regression_predictions.csv")
    metrics = pd.read_csv(DATA_DIR / "regression_metrics.csv")
    model = pred["model"].iloc[0]
    feature_set = pred["feature_set"].iloc[0]
    metric_lookup = metrics.loc[
        metrics["model"].eq(model) & metrics["feature_set"].eq(feature_set)
    ].set_index("target")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    axes = axes.ravel()
    for ax, target in zip(axes, REGRESSION_TARGETS):
        x = pred[f"actual_{target}"]
        y = pred[f"pred_{target}"]
        ax.scatter(x, y, s=28, alpha=0.72, color="#2F6F9F", edgecolor="white", linewidth=0.3)
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        pad = 0.03 * (hi - lo if hi > lo else 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#C44E52", linewidth=1.4)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        label = METRIC_LABELS[target]
        r2 = metric_lookup.loc[target, "r2"]
        ax.set_title(f"{label} (R2={r2:.3f})")
        ax.set_xlabel("Simulation")
        ax.set_ylabel("ML prediction")
        ax.grid(alpha=0.25)
    fig.suptitle(f"Best surrogate validation: {model} using {feature_set.replace('_', ' ')}", y=1.02)
    save_dual(fig, "01_regression_predicted_vs_simulated")


def plot_model_comparison() -> None:
    summary = pd.read_csv(DATA_DIR / "regression_metrics_summary.csv")
    summary = summary.sort_values(["feature_set", "mean_r2"], ascending=[True, False])
    labels = summary["model"] + "\n" + summary["feature_set"].str.replace("_", " ")
    colors = np.where(summary["feature_set"].eq("design_plus_profile"), "#4C78A8", "#F58518")

    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    ax.bar(np.arange(len(summary)), summary["mean_r2"], color=colors, alpha=0.9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(summary)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Mean test R2 across four targets")
    ax.set_title("Surrogate model comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#4C78A8", label="Design + profile features"),
            plt.Rectangle((0, 0), 1, 1, color="#F58518", label="Design variables only"),
        ],
        loc="lower right",
    )
    save_dual(fig, "02_model_metric_comparison")


def plot_feature_importance() -> None:
    importance = pd.read_csv(DATA_DIR / "feature_importance.csv").head(20)
    importance = importance.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9.4, 7.0))
    ax.barh(importance["feature"], importance["importance"], color="#54A24B", alpha=0.9)
    ax.set_xlabel("Random forest feature importance")
    ax.set_title("Top variables controlling surrogate predictions")
    ax.grid(axis="x", alpha=0.25)
    save_dual(fig, "03_feature_importance")


def plot_pca_clusters() -> None:
    clusters = pd.read_csv(DATA_DIR / "cluster_assignments.csv")
    fig, ax = plt.subplots(figsize=(9.3, 6.7))
    scatter = ax.scatter(
        clusters["pc1"],
        clusters["pc2"],
        c=clusters["cluster"],
        s=25 + 90 * clusters["balanced_score"],
        cmap="tab10",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.25,
    )
    high = clusters.loc[clusters["high_performer"].eq(1)]
    ax.scatter(
        high["pc1"],
        high["pc2"],
        facecolors="none",
        edgecolors="black",
        linewidth=0.8,
        s=95,
        label="Top 20% balanced score",
    )
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.set_title("Operating regimes from design, KPI, and profile features")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    cbar = fig.colorbar(scatter, ax=ax, pad=0.01)
    cbar.set_label("K-means regime")
    save_dual(fig, "04_pca_operating_regimes")


def plot_classification_confusion() -> None:
    cm = pd.read_csv(DATA_DIR / "classification_confusion_matrix.csv", index_col=0)
    metrics = pd.read_csv(DATA_DIR / "classification_metrics.csv")
    best = metrics.iloc[0]

    fig, ax = plt.subplots(figsize=(5.7, 5.2))
    im = ax.imshow(cm.to_numpy(), cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm.iloc[i, j]), ha="center", va="center", color="black", fontsize=13)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted low", "Predicted high"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual low", "Actual high"])
    ax.set_title(f"High-performance classifier: {best['model']} (F1={best['f1']:.3f})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_dual(fig, "05_classification_confusion_matrix")


def _representative_samples(manifest: pd.DataFrame) -> list[tuple[str, int]]:
    representatives = [
        ("Balanced", int(manifest.loc[manifest["balanced_score"].idxmax(), "sample_id"])),
        ("Highest purity", int(manifest.loc[manifest["purity"].idxmax(), "sample_id"])),
        ("Highest recovery", int(manifest.loc[manifest["recovery"].idxmax(), "sample_id"])),
        (
            "Highest productivity",
            int(manifest.loc[manifest["productivity_mol_kg_h"].idxmax(), "sample_id"]),
        ),
    ]
    seen: set[int] = set()
    unique: list[tuple[str, int]] = []
    for label, sample_id in representatives:
        if sample_id not in seen:
            unique.append((label, sample_id))
            seen.add(sample_id)
    return unique


def plot_profile_heatmaps() -> None:
    manifest = load_manifest()
    reps = _representative_samples(manifest)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), constrained_layout=True)
    axes = axes.ravel()
    image = None

    for ax, (label, sample_id) in zip(axes, reps):
        row = manifest.loc[manifest["sample_id"].eq(sample_id)].iloc[0]
        df = pd.read_csv(profile_path(row["profile_csv_path"]))
        ads = df.loc[df["step_name"].eq("Ads")].copy()
        if ads.empty:
            ads = df.copy()
        ads["t_norm"] = (ads["t_s"] - ads["t_s"].min()) / max(ads["t_s"].max() - ads["t_s"].min(), 1e-12)
        pivot = (
            ads.pivot_table(index="z", columns="t_norm", values="y_CO2", aggfunc="mean")
            .sort_index()
            .sort_index(axis=1)
        )
        extent = [0, 1, float(pivot.index.min()), float(pivot.index.max())]
        image = ax.imshow(
            pivot.to_numpy(),
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
            vmin=0.14,
            vmax=max(0.35, float(pivot.max().max())),
        )
        ax.set_title(
            f"{label}: sample {sample_id}\n"
            f"purity={row['purity']:.3f}, recovery={row['recovery']:.3f}"
        )
        ax.set_xlabel("Normalized adsorption-step time")
        ax.set_ylabel("Bed position z")

    for ax in axes[len(reps) :]:
        ax.axis("off")
    if image is not None:
        fig.colorbar(image, ax=axes.tolist(), shrink=0.9, label="Gas-phase CO2 mole fraction")
    save_dual(fig, "06_adsorption_profile_heatmaps")


def main() -> None:
    plot_regression_predictions()
    plot_model_comparison()
    plot_feature_importance()
    plot_pca_clusters()
    plot_classification_confusion()
    plot_profile_heatmaps()


if __name__ == "__main__":
    main()
