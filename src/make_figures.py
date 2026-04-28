from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from psa_surrogate_utils import (
    DATA_DIR,
    DESIGN_COLS,
    DESIGN_LABELS,
    DESIGN_SHORT_LABELS,
    PHYSICAL_TARGETS,
    REPORT_GA_FIG_DIR,
    SLIDE_GA_FIG_DIR,
    TARGET_LABELS,
    bounds_arrays,
    ensure_output_dirs,
    load_manifest,
    variable_bounds,
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
    png_path = SLIDE_GA_FIG_DIR / f"{stem}.png"
    pdf_path = REPORT_GA_FIG_DIR / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    plt.close(fig)


def summary_value(key: str, default: str = "") -> str:
    summary_path = DATA_DIR / "surrogate_summary.csv"
    if not summary_path.exists():
        return default
    summary = pd.read_csv(summary_path)
    match = summary.loc[summary["key"].eq(key), "value"]
    if match.empty:
        return default
    return str(match.iloc[0])


def plot_sampling_coverage() -> None:
    manifest = load_manifest()
    bounds = variable_bounds(manifest).set_index("variable")
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 9.3))
    axes = axes.ravel()
    for ax, col in zip(axes, DESIGN_COLS):
        ax.hist(manifest[col], bins=30, color="#4C78A8", alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.axvline(bounds.loc[col, "lower"], color="#C44E52", linestyle="--", linewidth=1.1)
        ax.axvline(bounds.loc[col, "upper"], color="#C44E52", linestyle="--", linewidth=1.1)
        ax.set_title(DESIGN_SHORT_LABELS[col])
        ax.set_xlabel(DESIGN_LABELS[col])
        ax.set_ylabel("Count")
        ax.grid(alpha=0.2)
    for ax in axes[len(DESIGN_COLS) :]:
        ax.axis("off")
    fig.suptitle("Manifest Sampling Coverage for the Seven Design Variables", y=1.01)
    fig.tight_layout()
    save_dual(fig, "01_manifest_sampling_coverage")


def plot_surrogate_validation() -> None:
    pred = pd.read_csv(DATA_DIR / "surrogate_predictions.csv")
    metrics = pd.read_csv(DATA_DIR / "surrogate_metrics.csv")
    primary_label = summary_value("primary_model_label", "Primary surrogate")
    test = pred.loc[pred["split"].eq("test")].copy()
    metric_lookup = metrics.loc[metrics["split"].eq("test")].set_index("target")

    panels = [
        ("purity", "CO2 purity"),
        ("recovery", "CO2 recovery"),
        ("productivity_mol_kg_h", "Productivity"),
        ("log_energy", "log10 specific energy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    axes = axes.ravel()
    for ax, (target, label) in zip(axes, panels):
        actual = test[f"actual_{target}"]
        predicted = test[f"pred_{target}"]
        ax.scatter(actual, predicted, s=25, alpha=0.72, color="#2F6F9F", edgecolor="white", linewidth=0.25)
        lo = float(min(actual.min(), predicted.min()))
        hi = float(max(actual.max(), predicted.max()))
        pad = 0.04 * (hi - lo if hi > lo else 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#C44E52", linewidth=1.2)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{label} (test R2={metric_lookup.loc[target, 'r2']:.3f})")
        ax.set_xlabel("Detailed simulation in manifest")
        ax.set_ylabel("Surrogate prediction")
        ax.grid(alpha=0.25)
    fig.suptitle(f"Manifest-Only Surrogate Validation: {primary_label}", y=1.02)
    fig.tight_layout()
    save_dual(fig, "02_surrogate_validation")


def plot_surrogate_metrics() -> None:
    comparison_path = DATA_DIR / "surrogate_model_metrics.csv"
    summary_path = DATA_DIR / "surrogate_model_summary.csv"
    if comparison_path.exists() and summary_path.exists():
        metrics = pd.read_csv(comparison_path)
        summary = pd.read_csv(summary_path).sort_values("rank_screening")
        targets = ["purity", "recovery", "productivity_mol_kg_h", "log_energy"]
        labels = summary["model_label"].to_list()
        test = metrics.loc[metrics["split"].eq("test") & metrics["target"].isin(targets)].copy()
        pivot = test.pivot_table(index="target", columns="model_label", values="r2", aggfunc="first")
        values = pivot.reindex(index=targets, columns=labels).to_numpy(float)

        fig, ax = plt.subplots(figsize=(12.2, 4.8))
        im = ax.imshow(values, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=24, ha="right")
        ax.set_yticks(np.arange(len(targets)))
        ax.set_yticklabels([TARGET_LABELS.get(target, target) for target in targets])
        ax.set_title("Held-Out Test R2 Across Surrogate Candidates")
        for row_idx in range(values.shape[0]):
            for col_idx in range(values.shape[1]):
                value = values[row_idx, col_idx]
                if np.isfinite(value):
                    text_color = "white" if value < 0.55 else "black"
                    ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", color=text_color, fontsize=8.5)
        cbar = fig.colorbar(im, ax=ax, pad=0.01)
        cbar.set_label("Test R2")
        fig.tight_layout()
        save_dual(fig, "03_surrogate_metric_summary")
        return

    metrics = pd.read_csv(DATA_DIR / "surrogate_metrics.csv")
    test = metrics.loc[metrics["split"].eq("test")].copy()
    order = ["purity", "recovery", "productivity_mol_kg_h", "log_energy", "energy_kWh_ton"]
    test["order"] = test["target"].map({target: idx for idx, target in enumerate(order)})
    test = test.sort_values("order")

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    labels = [TARGET_LABELS.get(target, target) for target in test["target"]]
    colors = ["#4C78A8" if target != "energy_kWh_ton" else "#F58518" for target in test["target"]]
    ax.bar(np.arange(len(test)), test["r2"], color=colors, alpha=0.9)
    ax.set_xticks(np.arange(len(test)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Test R2")
    ax.set_title("Surrogate Accuracy by Output")
    ax.grid(axis="y", alpha=0.25)
    save_dual(fig, "03_surrogate_metric_summary")


def plot_purity_recovery_front() -> None:
    manifest = load_manifest()
    ga = pd.read_csv(DATA_DIR / "ga_optimization_nondominated.csv")
    front = ga.loc[ga["problem"].eq("purity_recovery")].copy()

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    ax.scatter(manifest["purity"], manifest["recovery"], s=16, alpha=0.28, color="#9AA0A6", label="Manifest samples")
    sc = ax.scatter(
        front["pred_purity"],
        front["pred_recovery"],
        c=front["pred_energy_kWh_ton"],
        s=48,
        cmap="viridis_r",
        alpha=0.9,
        edgecolor="white",
        linewidth=0.35,
        label="GA surrogate Pareto candidates",
    )
    ax.set_xlabel("Predicted CO2 purity [-]")
    ax.set_ylabel("Predicted CO2 recovery [-]")
    ax.set_title("GA Front for Purity-Recovery Optimization")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Predicted energy [kWh ton$^{-1}$]")
    save_dual(fig, "04_ga_purity_recovery_front")


def plot_productivity_energy_front() -> None:
    manifest = load_manifest()
    ga = pd.read_csv(DATA_DIR / "ga_optimization_nondominated.csv")
    front = ga.loc[ga["problem"].eq("productivity_energy")].copy()

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    ax.scatter(
        manifest["energy_kWh_ton"],
        manifest["productivity_mol_kg_h"],
        s=16,
        alpha=0.28,
        color="#9AA0A6",
        label="Manifest samples",
    )
    sc = ax.scatter(
        front["pred_energy_kWh_ton"],
        front["pred_productivity_mol_kg_h"],
        c=front["pred_purity"],
        s=48,
        cmap="plasma",
        alpha=0.9,
        edgecolor="white",
        linewidth=0.35,
        label="GA surrogate Pareto candidates",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Predicted specific energy [kWh ton$^{-1}$]")
    ax.set_ylabel("Predicted productivity [mol kg$^{-1}$ h$^{-1}$]")
    ax.set_title("GA Front for Productivity-Energy Optimization")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Predicted CO2 purity [-]")
    save_dual(fig, "05_ga_productivity_energy_front")


def plot_closed_loop_status() -> None:
    comparison_path = DATA_DIR / "detailed_model_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        panels = [
            ("purity", "CO2 purity"),
            ("recovery", "CO2 recovery"),
            ("productivity_mol_kg_h", "Productivity"),
            ("energy_kWh_ton", "Specific energy"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
        axes = axes.ravel()
        for ax, (target, label) in zip(axes, panels):
            x = comparison[f"{target}_detailed"]
            y = comparison[f"pred_{target}"]
            ax.scatter(x, y, s=35, alpha=0.75, color="#4C78A8", edgecolor="white", linewidth=0.3)
            lo = float(min(x.min(), y.min()))
            hi = float(max(x.max(), y.max()))
            pad = 0.04 * (hi - lo if hi > lo else 1.0)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#C44E52", linewidth=1.2)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(label)
            ax.set_xlabel("Detailed model")
            ax.set_ylabel("Surrogate")
            ax.grid(alpha=0.25)
        fig.suptitle("Closed-Loop Detailed-Model Comparison", y=1.02)
        fig.tight_layout()
        save_dual(fig, "06_closed_loop_handoff_or_comparison")
        return

    candidates = pd.read_csv(DATA_DIR / "detailed_model_input.csv")
    bounds = variable_bounds(load_manifest())
    lower, upper = bounds_arrays(bounds)
    normalized = candidates[DESIGN_COLS].copy()
    for idx, col in enumerate(DESIGN_COLS):
        normalized[col] = (normalized[col] - lower[idx]) / (upper[idx] - lower[idx])

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    x_axis = np.arange(len(DESIGN_COLS))
    colors = {"purity_recovery": "#4C78A8", "productivity_energy": "#F58518"}
    for problem, group in candidates.groupby("problem"):
        values = normalized.loc[group.index, DESIGN_COLS].to_numpy(float)
        for row in values:
            ax.plot(x_axis, row, color=colors.get(problem, "#666666"), alpha=0.28, linewidth=1.0)
        ax.plot([], [], color=colors.get(problem, "#666666"), linewidth=2.0, label=problem.replace("_", " "))
    ax.set_xticks(x_axis)
    ax.set_xticklabels([DESIGN_SHORT_LABELS[col] for col in DESIGN_COLS])
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Normalized variable value")
    ax.set_title("Detailed-Model Handoff Candidates from GA")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    save_dual(fig, "06_closed_loop_handoff_or_comparison")


def main() -> None:
    plot_sampling_coverage()
    plot_surrogate_validation()
    plot_surrogate_metrics()
    plot_purity_recovery_front()
    plot_productivity_energy_front()
    plot_closed_loop_status()


if __name__ == "__main__":
    main()
