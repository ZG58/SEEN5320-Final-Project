from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from psa_project_utils import (
    DATA_DIR,
    INPUT_COLS,
    RAW_MAT_PATH,
    SCALAR_CSV_PATH,
    TARGET_COLS,
    input_bounds_frame,
    summarize_numeric,
    validate_scalar_samples,
)


def load_raw_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    mat = loadmat(path)
    if "invar" not in mat:
        raise KeyError("MAT file does not contain variable 'invar'.")
    if "result_matrix" not in mat:
        raise KeyError("MAT file does not contain variable 'result_matrix'.")

    invar = np.asarray(mat["invar"], dtype=float)
    result_matrix = np.asarray(mat["result_matrix"], dtype=float)
    if invar.ndim != 2 or invar.shape[1] != len(INPUT_COLS):
        raise ValueError(f"Expected invar to have shape N x {len(INPUT_COLS)}, found {invar.shape}.")
    if result_matrix.ndim != 2 or result_matrix.shape[1] < len(TARGET_COLS):
        raise ValueError(
            f"Expected result_matrix to have at least {len(TARGET_COLS)} columns, found {result_matrix.shape}."
        )
    if invar.shape[0] != result_matrix.shape[0]:
        raise ValueError(f"Row mismatch: invar has {invar.shape[0]} rows, result_matrix has {result_matrix.shape[0]}.")
    return invar, result_matrix


def convert_raw_mat_to_frame(path: str | Path = RAW_MAT_PATH) -> pd.DataFrame:
    invar, result_matrix = load_raw_arrays(path)
    frame = pd.DataFrame(invar, columns=INPUT_COLS)
    for idx, col in enumerate(TARGET_COLS):
        frame[col] = result_matrix[:, idx]
    frame.insert(0, "status", "success")
    frame.insert(0, "sample_id", np.arange(1, len(frame) + 1, dtype=int))
    validate_scalar_samples(frame)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw gPROMS PSA MAT data to a scalar CSV table.")
    parser.add_argument("--input", default=str(RAW_MAT_PATH), help="Path to raw_data.mat.")
    parser.add_argument("--output", default=str(SCALAR_CSV_PATH), help="Path for the converted CSV table.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = convert_raw_mat_to_frame(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    input_bounds_frame().to_csv(DATA_DIR / "psa_input_bounds.csv", index=False)
    summarize_numeric(frame, [*INPUT_COLS, *TARGET_COLS]).to_csv(DATA_DIR / "psa_scalar_summary.csv", index=False)
    print(f"Saved {output_path}")
    print(f"Rows: {len(frame)}")
    print(f"Columns: {len(frame.columns)}")
    print(frame[[*INPUT_COLS, *TARGET_COLS]].describe().to_string())


if __name__ == "__main__":
    main()
