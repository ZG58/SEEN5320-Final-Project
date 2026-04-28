from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from psa_project_utils import (
    DATA_DIR,
    INPUT_COLS,
    INPUT_LABELS,
    INPUT_SHORT_LABELS,
    REPORT_FIG_DIR,
    SLIDE_FIG_DIR,
    TARGET_LABELS,
    empirical_pareto_masks,
    ensure_output_dirs,
    input_bounds_arrays,
    input_bounds_frame,
    load_scalar_samples,
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
    png_path = SLIDE_FIG_DIR / f"{stem}.png"
    pdf_path = REPORT_FIG_DIR / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    plt.close(fig)


def plot_sampling_coverage() -> None:
    frame = load_scalar_samples()
    bounds = input_bounds_frame().set_index("variable")
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.8))
    for ax, col in zip(axes.ravel(), INPUT_COLS):
        ax.hist(frame[col], bins=32, color="#4C78A8", alpha=0.86, edgecolor="white", linewidth=0.4)
        ax.axvline(bounds.loc[col, "lower"], color="#C44E52", linestyle="--", linewidth=1.1)
        ax.axvline(bounds.loc[col, "upper"], color="#C44E52", linestyle="--", linewidth=1.1)
        ax.set_title(INPUT_SHORT_LABELS[col])
        ax.set_xlabel(INPUT_LABELS[col])
        ax.set_ylabel("Count")
        ax.grid(alpha=0.22)
    fig.suptitle("LHS Sampling Coverage for Successful gPROMS Simulations", y=1.02)
    fig.tight_layout()
    save_dual(fig, "01_sampling_coverage")


def plot_kpi_distributions() -> None:
    frame = load_scalar_samples()
    panels = [
        ("recovery", "#4C78A8"),
        ("purity", "#59A14F"),
        ("productivity_mol_h_kg", "#F58518"),
        ("energy_kJ_kgCO2", "#B279A2"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))
    for ax, (col, color) in zip(axes.ravel(), panels):
        ax.hist(frame[col], bins=34, color=color, alpha=0.86, edgecolor="white", linewidth=0.4)
        ax.axvline(frame[col].mean(), color="#222222", linewidth=1.1, linestyle="--", label="Mean")
        ax.set_title(TARGET_LABELS[col])
        ax.set_xlabel(TARGET_LABELS[col])
        ax.set_ylabel("Count")
        ax.grid(alpha=0.22)
        ax.legend(loc="best")
    fig.suptitle("Distribution of Four Scalar PSA Performance Indicators", y=1.02)
    fig.tight_layout()
    save_dual(fig, "02_kpi_distributions")


def plot_empirical_pareto_sets() -> None:
    frame = load_scalar_samples()
    masks = empirical_pareto_masks(frame)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))

    ax = axes[0]
    ax.scatter(frame["purity"], frame["recovery"], s=15, alpha=0.28, color="#9AA0A6", label="Successful samples")
    ax.scatter(
        frame.loc[masks["purity_recovery_near"], "purity"],
        frame.loc[masks["purity_recovery_near"], "recovery"],
        s=26,
        alpha=0.75,
        color="#F58518",
        label="Near front",
    )
    ax.scatter(
        frame.loc[masks["purity_recovery_front"], "purity"],
        frame.loc[masks["purity_recovery_front"], "recovery"],
        s=38,
        alpha=0.92,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.3,
        label="Empirical front",
    )
    ax.set_xlabel("CO2 purity [-]")
    ax.set_ylabel("CO2 recovery [-]")
    ax.set_title("Empirical Purity-Recovery Tradeoff")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1]
    ax.scatter(
        frame["energy_kJ_kgCO2"],
        frame["productivity_mol_h_kg"],
        s=15,
        alpha=0.28,
        color="#9AA0A6",
        label="Successful samples",
    )
    ax.scatter(
        frame.loc[masks["productivity_energy_near"], "energy_kJ_kgCO2"],
        frame.loc[masks["productivity_energy_near"], "productivity_mol_h_kg"],
        s=26,
        alpha=0.75,
        color="#F58518",
        label="Near front",
    )
    ax.scatter(
        frame.loc[masks["productivity_energy_front"], "energy_kJ_kgCO2"],
        frame.loc[masks["productivity_energy_front"], "productivity_mol_h_kg"],
        s=38,
        alpha=0.92,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.3,
        label="Empirical front",
    )
    ax.set_xlabel("Specific energy [kJ kgCO2$^{-1}$]")
    ax.set_ylabel("Productivity [mol h$^{-1}$ kg$^{-1}$]")
    ax.set_title("Empirical Productivity-Energy Tradeoff")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    save_dual(fig, "03_empirical_pareto_sets")


def plot_model_comparison() -> None:
    summary = pd.read_csv(DATA_DIR / "psa_model_summary.csv").sort_values("selection_score", ascending=True)
    labels = summary["model_label"].to_list()
    x = np.arange(len(summary))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.8, 5.6))
    ax.bar(x - width / 2, summary["global_norm_rmse"], width=width, color="#4C78A8", label="Global test")
    ax.bar(x + width / 2, summary["pareto_norm_rmse"], width=width, color="#F58518", label="Pareto-focused test")
    ax.plot(x, summary["selection_score"], color="#222222", marker="o", linewidth=1.4, label="Selection score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right")
    ax.set_ylabel("Mean normalized RMSE")
    ax.set_title("Tuned Surrogate Model Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    save_dual(fig, "04_model_comparison")


def plot_best_model_validation() -> None:
    pred = pd.read_csv(DATA_DIR / "psa_best_model_predictions.csv")
    metrics = pd.read_csv(DATA_DIR / "psa_best_model_metrics.csv")
    test = pred.loc[pred["split"].eq("test")].copy()
    metric_lookup = metrics.loc[metrics["subset"].eq("test_global")].set_index("target")
    panels = [
        ("recovery", "CO2 recovery"),
        ("purity", "CO2 purity"),
        ("productivity_mol_h_kg", "Productivity"),
        ("energy_kJ_kgCO2", "Specific energy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0))
    for ax, (target, label) in zip(axes.ravel(), panels):
        actual = test[f"actual_{target}"]
        predicted = test[f"pred_{target}"]
        ax.scatter(actual, predicted, s=24, alpha=0.74, color="#4C78A8", edgecolor="white", linewidth=0.25)
        lo = float(min(actual.min(), predicted.min()))
        hi = float(max(actual.max(), predicted.max()))
        pad = 0.04 * (hi - lo if hi > lo else 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#C44E52", linewidth=1.2)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{label} (test R2={metric_lookup.loc[target, 'r2']:.3f})")
        ax.set_xlabel("gPROMS simulation")
        ax.set_ylabel("Surrogate prediction")
        ax.grid(alpha=0.25)
    fig.suptitle("Best Tuned Surrogate: Held-Out Predictions", y=1.02)
    fig.tight_layout()
    save_dual(fig, "05_best_model_validation")


def plot_optimization_fronts() -> None:
    frame = load_scalar_samples()
    ga = pd.read_csv(DATA_DIR / "psa_optimization_nondominated.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4))

    pr = ga.loc[ga["problem"].eq("purity_recovery") & ga["is_feasible"].eq(1)].copy()
    ax = axes[0]
    ax.scatter(frame["purity"], frame["recovery"], s=15, alpha=0.25, color="#9AA0A6", label="Successful samples")
    sc = ax.scatter(
        pr["pred_purity"],
        pr["pred_recovery"],
        c=pr["pred_energy_kJ_kgCO2"],
        s=42,
        cmap="viridis_r",
        alpha=0.9,
        edgecolor="white",
        linewidth=0.3,
        label="NSGA-II candidates",
    )
    ax.set_xlabel("Predicted CO2 purity [-]")
    ax.set_ylabel("Predicted CO2 recovery [-]")
    ax.set_title("Surrogate Front: Purity-Recovery")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Energy [kJ kgCO2$^{-1}$]")

    pe = ga.loc[ga["problem"].eq("productivity_energy") & ga["is_feasible"].eq(1)].copy()
    ax = axes[1]
    ax.scatter(
        frame["energy_kJ_kgCO2"],
        frame["productivity_mol_h_kg"],
        s=18,
        alpha=0.25,
        color="#9AA0A6",
        label="Successful samples",
    )
    sc = ax.scatter(
        pe["pred_energy_kJ_kgCO2"],
        pe["pred_productivity_mol_h_kg"],
        c=pe["pred_purity"],
        s=42,
        cmap="plasma",
        alpha=0.9,
        edgecolor="white",
        linewidth=0.3,
        label="NSGA-II candidates",
    )
    ax.set_xlabel("Predicted specific energy [kJ kgCO2$^{-1}$]")
    ax.set_ylabel("Predicted productivity [mol h$^{-1}$ kg$^{-1}$]")
    ax.set_title("Surrogate Front: Productivity-Energy")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("CO2 purity [-]")

    fig.tight_layout()
    save_dual(fig, "06_optimization_fronts")


def plot_optimization_input_distributions() -> None:
    samples = load_scalar_samples()
    candidates = pd.read_csv(DATA_DIR / "psa_optimization_nondominated.csv")
    candidates = candidates.loc[candidates["is_feasible"].eq(1)].copy()
    bounds = input_bounds_frame().set_index("variable")

    colors = {"purity_recovery": "#4C78A8", "productivity_energy": "#F58518"}
    labels = {"purity_recovery": "Purity-recovery", "productivity_energy": "Productivity-energy"}
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.8))
    for ax, col in zip(axes.ravel(), INPUT_COLS):
        x_min = float(bounds.loc[col, "lower"])
        x_max = float(bounds.loc[col, "upper"])
        span = x_max - x_min if x_max > x_min else 1.0
        bins = np.linspace(x_min, x_max, 26)
        ax.hist(
            samples[col],
            bins=bins,
            density=True,
            color="#BDBDBD",
            alpha=0.40,
            edgecolor="white",
            linewidth=0.35,
            label="Successful samples",
        )
        for problem, group in candidates.groupby("problem", sort=False):
            ax.hist(
                group[col],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                color=colors.get(problem, "#666666"),
                label=labels.get(problem, problem.replace("_", " ")),
            )
            jitter = np.linspace(0.0, 0.025, len(group), endpoint=False)
            ax.scatter(
                group[col],
                np.full(len(group), -0.015) - jitter,
                s=8,
                marker="|",
                color=colors.get(problem, "#666666"),
                alpha=0.35,
                clip_on=False,
            )
        ax.set_xlim(x_min - 0.015 * span, x_max + 0.015 * span)
        ax.set_ylim(bottom=-0.055)
        ax.set_title(INPUT_SHORT_LABELS[col])
        ax.set_xlabel(INPUT_LABELS[col])
        ax.set_ylabel("Density")
        ax.grid(alpha=0.22)
    handles, legend_labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Optimized Input Distributions within Original Sampling Ranges", y=1.08)
    fig.tight_layout()
    save_dual(fig, "08_optimization_input_distributions")


def plot_representative_conditions() -> None:
    reps = pd.read_csv(DATA_DIR / "psa_optimization_representative_cases.csv")
    lower, upper = input_bounds_arrays()
    normalized = reps[INPUT_COLS].copy()
    for idx, col in enumerate(INPUT_COLS):
        normalized[col] = (normalized[col] - lower[idx]) / (upper[idx] - lower[idx])
    x_axis = np.arange(len(INPUT_COLS))
    fig, ax = plt.subplots(figsize=(9.6, 5.5))
    colors = {"purity_recovery": "#4C78A8", "productivity_energy": "#F58518"}
    for problem, group in reps.groupby("problem", sort=False):
        values = normalized.loc[group.index, INPUT_COLS].to_numpy(float)
        for row in values:
            ax.plot(x_axis, row, color=colors.get(problem, "#666666"), alpha=0.36, linewidth=1.2)
        ax.plot([], [], color=colors.get(problem, "#666666"), linewidth=2.0, label=problem.replace("_", " "))
    ax.set_xticks(x_axis)
    ax.set_xticklabels([INPUT_SHORT_LABELS[col] for col in INPUT_COLS])
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Normalized input value")
    ax.set_title("Representative Optimized Operating Conditions")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    save_dual(fig, "07_representative_conditions")


def main() -> None:
    plot_sampling_coverage()
    plot_kpi_distributions()
    plot_empirical_pareto_sets()
    plot_model_comparison()
    plot_best_model_validation()
    plot_optimization_fronts()
    plot_optimization_input_distributions()
    plot_representative_conditions()


if __name__ == "__main__":
    main()
