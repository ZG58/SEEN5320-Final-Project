from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from psa_ml_utils import DATA_DIR, load_manifest, profile_path


STEP_ORDER = ["CoCPres", "Ads", "HR1", "CoCDepres", "HR2", "CnCDepres", "LR"]
FIELD_COLS = ["P_Pa", "y_CO2", "q_CO2_mol_kg", "q_N2_mol_kg", "T_K", "T_wall_K"]


def _numeric_summary(prefix: str, values: pd.Series, features: dict[str, float]) -> None:
    arr = pd.to_numeric(values, errors="coerce")
    features[f"{prefix}_mean"] = float(arr.mean())
    features[f"{prefix}_std"] = float(arr.std(ddof=0))
    features[f"{prefix}_min"] = float(arr.min())
    features[f"{prefix}_max"] = float(arr.max())
    features[f"{prefix}_range"] = float(arr.max() - arr.min())


def _final_spatial_features(prefix: str, step: pd.DataFrame, col: str, features: dict[str, float]) -> None:
    final_time = step["t_s"].max()
    final = step.loc[step["t_s"].eq(final_time)].sort_values("z")
    if final.empty:
        return
    inlet = float(final[col].iloc[0])
    outlet = float(final[col].iloc[-1])
    features[f"{prefix}_{col}_final_mean"] = float(final[col].mean())
    features[f"{prefix}_{col}_final_inlet"] = inlet
    features[f"{prefix}_{col}_final_outlet"] = outlet
    features[f"{prefix}_{col}_final_gradient"] = outlet - inlet


def extract_profile_features(sample_id: int, csv_path: str) -> dict[str, float]:
    path = profile_path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing profile for sample {sample_id}: {path}")

    df = pd.read_csv(path)
    required = ["sample_id", "step_name", "t_s", "node", "z", *FIELD_COLS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Profile {path} is missing columns: {missing}")

    for col in ["t_s", "node", "z", *FIELD_COLS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    features: dict[str, float] = {
        "sample_id": int(sample_id),
        "profile_rows": float(len(df)),
        "profile_steps": float(df["step_name"].nunique()),
        "profile_nodes": float(df["node"].nunique()),
        "profile_time_points": float(df["t_s"].nunique()),
        "profile_t_min": float(df["t_s"].min()),
        "profile_t_max": float(df["t_s"].max()),
        "profile_t_span": float(df["t_s"].max() - df["t_s"].min()),
    }

    for col in FIELD_COLS:
        _numeric_summary(f"profile_{col}", df[col], features)

    features["profile_loading_selectivity_proxy"] = float(
        df["q_CO2_mol_kg"].mean() / max(df["q_N2_mol_kg"].mean(), 1e-12)
    )
    features["profile_temperature_wall_lag_mean"] = float((df["T_K"] - df["T_wall_K"]).mean())
    features["profile_temperature_wall_lag_abs_mean"] = float((df["T_K"] - df["T_wall_K"]).abs().mean())

    for step_name in STEP_ORDER:
        step = df.loc[df["step_name"].eq(step_name)].copy()
        prefix = f"step_{step_name}"
        features[f"{prefix}_present"] = float(not step.empty)
        features[f"{prefix}_rows"] = float(len(step))
        if step.empty:
            continue

        features[f"{prefix}_time_points"] = float(step["t_s"].nunique())
        features[f"{prefix}_duration"] = float(step["t_s"].max() - step["t_s"].min())
        features[f"{prefix}_z_span"] = float(step["z"].max() - step["z"].min())
        for col in FIELD_COLS:
            _numeric_summary(f"{prefix}_{col}", step[col], features)
            _final_spatial_features(prefix, step, col, features)

        final = step.loc[step["t_s"].eq(step["t_s"].max())].sort_values("z")
        if not final.empty:
            features[f"{prefix}_final_loading_selectivity_proxy"] = float(
                final["q_CO2_mol_kg"].mean() / max(final["q_N2_mol_kg"].mean(), 1e-12)
            )
            features[f"{prefix}_final_temperature_wall_lag_mean"] = float(
                (final["T_K"] - final["T_wall_K"]).mean()
            )

    ads = df.loc[df["step_name"].eq("Ads")]
    if not ads.empty:
        final_ads = ads.loc[ads["t_s"].eq(ads["t_s"].max())].sort_values("z")
        features["ads_final_outlet_y_CO2_minus_feed"] = float(final_ads["y_CO2"].iloc[-1] - 0.15)
        features["ads_final_mean_q_CO2"] = float(final_ads["q_CO2_mol_kg"].mean())
        features["ads_final_mean_q_N2"] = float(final_ads["q_N2_mol_kg"].mean())

    return features


def build_features(limit: int | None = None) -> pd.DataFrame:
    manifest = load_manifest()
    rows = manifest if limit is None else manifest.head(limit)

    start = time.time()
    feature_rows: list[dict[str, float]] = []
    for idx, row in enumerate(rows.itertuples(index=False), start=1):
        feature_rows.append(extract_profile_features(int(row.sample_id), row.profile_csv_path))
        if idx % 100 == 0 or idx == len(rows):
            elapsed = time.time() - start
            print(f"Extracted {idx}/{len(rows)} profiles in {elapsed:.1f} s")

    features = pd.DataFrame(feature_rows).sort_values("sample_id").reset_index(drop=True)
    numeric_cols = [col for col in features.columns if col != "sample_id"]
    features[numeric_cols] = features[numeric_cols].replace([np.inf, -np.inf], np.nan)
    features[numeric_cols] = features[numeric_cols].fillna(features[numeric_cols].median(numeric_only=True))

    out_path = DATA_DIR / "profile_features.csv"
    features.to_csv(out_path, index=False)
    print(f"Saved {out_path} with shape {features.shape}")
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PSA final-cycle profile features.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of profiles for a dry run.")
    args = parser.parse_args()
    build_features(limit=args.limit)


if __name__ == "__main__":
    main()
