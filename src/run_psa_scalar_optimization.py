from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from psa_project_utils import (
    DATA_DIR,
    INPUT_COLS,
    PURITY_CONSTRAINT,
    RANDOM_STATE,
    RECOVERY_CONSTRAINT,
    TARGET_COLS,
    clip_to_observed_bounds,
    inverse_transformed_targets,
    load_scalar_samples,
)


PROBLEMS = ["purity_recovery", "productivity_energy"]


def load_artifact(path=None) -> dict:
    if path is None:
        path = DATA_DIR / "psa_best_surrogate.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing trained surrogate artifact: {path}")
    return joblib.load(path)


def bounds_from_artifact(artifact: dict) -> tuple[np.ndarray, np.ndarray]:
    bounds = artifact.get("input_bounds")
    if bounds is None:
        raise KeyError("Surrogate artifact does not contain input_bounds.")
    if not isinstance(bounds, pd.DataFrame):
        bounds = pd.DataFrame(bounds)
    ordered = bounds.set_index("variable").loc[INPUT_COLS]
    return ordered["lower"].to_numpy(float), ordered["upper"].to_numpy(float)


def observed_bounds_frame(artifact: dict) -> pd.DataFrame:
    observed = artifact.get("observed_outputs")
    if not observed:
        raise KeyError("Surrogate artifact does not contain observed output bounds.")
    return pd.DataFrame(
        {
            col: [float(values["min"]), float(values["max"])]
            for col, values in observed.items()
            if col in [*TARGET_COLS, "log_energy"]
        }
    )


def predict_artifact(artifact: dict, x_raw: np.ndarray) -> pd.DataFrame:
    x_frame = pd.DataFrame(x_raw, columns=INPUT_COLS)
    transformed = artifact["estimator"].predict(x_frame)
    predictions = inverse_transformed_targets(transformed)
    return clip_to_observed_bounds(predictions, observed_bounds_frame(artifact))


def normalized_to_raw(population: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return lower + population * (upper - lower)


def objective_scores(predictions: pd.DataFrame, problem: str) -> tuple[np.ndarray, np.ndarray]:
    if problem == "purity_recovery":
        scores = predictions[["purity", "recovery"]].to_numpy(float)
        violations = np.zeros(len(predictions), dtype=float)
        return scores, violations
    if problem == "productivity_energy":
        purity_gap = np.maximum(0.0, PURITY_CONSTRAINT - predictions["purity"].to_numpy(float)) / PURITY_CONSTRAINT
        recovery_gap = np.maximum(0.0, RECOVERY_CONSTRAINT - predictions["recovery"].to_numpy(float)) / RECOVERY_CONSTRAINT
        violations = purity_gap + recovery_gap
        scores = np.column_stack(
            [
                predictions["productivity_mol_h_kg"].to_numpy(float),
                -predictions["log_energy"].to_numpy(float),
            ]
        )
        return scores, violations
    raise ValueError(f"Unknown optimization problem: {problem}")


def evaluate_population(
    population: np.ndarray,
    artifact: dict,
    lower: np.ndarray,
    upper: np.ndarray,
    problem: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    x_raw = normalized_to_raw(population, lower, upper)
    predictions = predict_artifact(artifact, x_raw)
    scores, violations = objective_scores(predictions, problem)
    return predictions, scores, violations


def dominates(scores: np.ndarray, violations: np.ndarray, i: int, j: int) -> bool:
    tol = 1e-12
    i_feasible = violations[i] <= tol
    j_feasible = violations[j] <= tol
    if i_feasible and not j_feasible:
        return True
    if j_feasible and not i_feasible:
        return False
    if not i_feasible and not j_feasible and abs(violations[i] - violations[j]) > tol:
        return bool(violations[i] < violations[j])
    return bool(np.all(scores[i] >= scores[j]) and np.any(scores[i] > scores[j]))


def fast_nondominated_sort(scores: np.ndarray, violations: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
    n_rows = scores.shape[0]
    dominated_sets: list[list[int]] = [[] for _ in range(n_rows)]
    domination_counts = np.zeros(n_rows, dtype=int)
    ranks = np.full(n_rows, -1, dtype=int)
    fronts: list[list[int]] = [[]]

    for p in range(n_rows):
        for q in range(n_rows):
            if p == q:
                continue
            if dominates(scores, violations, p, q):
                dominated_sets[p].append(q)
            elif dominates(scores, violations, q, p):
                domination_counts[p] += 1
        if domination_counts[p] == 0:
            ranks[p] = 0
            fronts[0].append(p)

    current = 0
    while current < len(fronts) and fronts[current]:
        next_front: list[int] = []
        for p in fronts[current]:
            for q in dominated_sets[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    ranks[q] = current + 1
                    next_front.append(q)
        current += 1
        if next_front:
            fronts.append(next_front)
    return fronts, ranks


def crowding_distance(scores: np.ndarray, fronts: list[list[int]]) -> np.ndarray:
    distances = np.zeros(scores.shape[0], dtype=float)
    for front in fronts:
        if not front:
            continue
        if len(front) <= 2:
            distances[front] = np.inf
            continue
        front_scores = scores[front]
        front_dist = np.zeros(len(front), dtype=float)
        for obj_idx in range(scores.shape[1]):
            order = np.argsort(front_scores[:, obj_idx])
            front_dist[order[0]] = np.inf
            front_dist[order[-1]] = np.inf
            lo = front_scores[order[0], obj_idx]
            hi = front_scores[order[-1], obj_idx]
            if hi <= lo:
                continue
            for k in range(1, len(front) - 1):
                front_dist[order[k]] += (front_scores[order[k + 1], obj_idx] - front_scores[order[k - 1], obj_idx]) / (
                    hi - lo
                )
        for local_idx, global_idx in enumerate(front):
            distances[global_idx] = front_dist[local_idx]
    return distances


def rank_and_crowding(scores: np.ndarray, violations: np.ndarray) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    fronts, ranks = fast_nondominated_sort(scores, violations)
    crowding = crowding_distance(scores, fronts)
    return fronts, ranks, crowding


def tournament_select(ranks: np.ndarray, crowding: np.ndarray, rng: np.random.Generator) -> int:
    i, j = rng.integers(0, len(ranks), size=2)
    if ranks[i] < ranks[j]:
        return int(i)
    if ranks[j] < ranks[i]:
        return int(j)
    if crowding[i] > crowding[j]:
        return int(i)
    if crowding[j] > crowding[i]:
        return int(j)
    return int(i if rng.random() < 0.5 else j)


def sbx_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    rng: np.random.Generator,
    probability: float,
    eta: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    child1 = parent1.copy()
    child2 = parent2.copy()
    if rng.random() > probability:
        return child1, child2
    for idx in range(parent1.size):
        if rng.random() > 0.5 or abs(parent1[idx] - parent2[idx]) < 1e-14:
            continue
        x1 = min(parent1[idx], parent2[idx])
        x2 = max(parent1[idx], parent2[idx])
        rand = rng.random()

        beta = 1.0 + 2.0 * x1 / (x2 - x1)
        alpha = 2.0 - beta ** (-(eta + 1.0))
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
        c1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

        beta = 1.0 + 2.0 * (1.0 - x2) / (x2 - x1)
        alpha = 2.0 - beta ** (-(eta + 1.0))
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
        c2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

        c1 = float(np.clip(c1, 0.0, 1.0))
        c2 = float(np.clip(c2, 0.0, 1.0))
        if rng.random() <= 0.5:
            child1[idx] = c2
            child2[idx] = c1
        else:
            child1[idx] = c1
            child2[idx] = c2
    return child1, child2


def polynomial_mutation(
    individual: np.ndarray,
    rng: np.random.Generator,
    probability: float,
    eta: float = 20.0,
) -> np.ndarray:
    mutant = individual.copy()
    mut_pow = 1.0 / (eta + 1.0)
    for idx in range(mutant.size):
        if rng.random() > probability:
            continue
        x = mutant[idx]
        delta1 = x
        delta2 = 1.0 - x
        rand = rng.random()
        if rand < 0.5:
            xy = 1.0 - delta1
            val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1.0)
            deltaq = val**mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1.0)
            deltaq = 1.0 - val**mut_pow
        mutant[idx] = np.clip(x + deltaq, 0.0, 1.0)
    return mutant


def make_offspring(
    population: np.ndarray,
    ranks: np.ndarray,
    crowding: np.ndarray,
    rng: np.random.Generator,
    crossover_probability: float,
    mutation_probability: float,
) -> np.ndarray:
    children: list[np.ndarray] = []
    while len(children) < len(population):
        parent1 = population[tournament_select(ranks, crowding, rng)]
        parent2 = population[tournament_select(ranks, crowding, rng)]
        child1, child2 = sbx_crossover(parent1, parent2, rng, probability=crossover_probability)
        children.append(polynomial_mutation(child1, rng, probability=mutation_probability))
        if len(children) < len(population):
            children.append(polynomial_mutation(child2, rng, probability=mutation_probability))
    return np.vstack(children)


def select_next_generation(
    population: np.ndarray,
    scores: np.ndarray,
    violations: np.ndarray,
    popsize: int,
) -> np.ndarray:
    fronts, _, crowding = rank_and_crowding(scores, violations)
    selected: list[int] = []
    for front in fronts:
        if len(selected) + len(front) <= popsize:
            selected.extend(front)
        else:
            remaining = popsize - len(selected)
            order = sorted(front, key=lambda idx: crowding[idx], reverse=True)
            selected.extend(order[:remaining])
            break
    return population[np.asarray(selected, dtype=int)]


def initial_population(
    samples: pd.DataFrame,
    lower: np.ndarray,
    upper: np.ndarray,
    popsize: int,
    rng: np.random.Generator,
) -> np.ndarray:
    observed = (samples[INPUT_COLS].to_numpy(float) - lower) / (upper - lower)
    observed = np.clip(observed, 0.0, 1.0)
    n_seed = min(popsize // 2, len(observed))
    seed_idx = rng.choice(len(observed), size=n_seed, replace=False)
    random_part = rng.random((popsize - n_seed, len(INPUT_COLS)))
    return np.vstack([observed[seed_idx], random_part])


def run_problem(
    problem: str,
    artifact: dict,
    samples: pd.DataFrame,
    lower: np.ndarray,
    upper: np.ndarray,
    popsize: int,
    generations: int,
    seed: int,
    crossover_probability: float,
    mutation_probability: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    population = initial_population(samples, lower, upper, popsize, rng)
    for generation in range(generations):
        _, scores, violations = evaluate_population(population, artifact, lower, upper, problem)
        _, ranks, crowding = rank_and_crowding(scores, violations)
        offspring = make_offspring(
            population,
            ranks,
            crowding,
            rng,
            crossover_probability=crossover_probability,
            mutation_probability=mutation_probability,
        )
        combined = np.vstack([population, offspring])
        _, combined_scores, combined_violations = evaluate_population(combined, artifact, lower, upper, problem)
        population = select_next_generation(combined, combined_scores, combined_violations, popsize)
        if generation == 0 or (generation + 1) % 50 == 0 or generation + 1 == generations:
            _, current_scores, current_violations = evaluate_population(population, artifact, lower, upper, problem)
            _, current_ranks, _ = rank_and_crowding(current_scores, current_violations)
            rank0 = int(np.sum(current_ranks == 0))
            feasible = int(np.sum(current_violations <= 1e-12))
            print(f"{problem}: generation {generation + 1}/{generations}, rank-0={rank0}, feasible={feasible}")

    predictions, scores, violations = evaluate_population(population, artifact, lower, upper, problem)
    _, ranks, crowding = rank_and_crowding(scores, violations)
    x_raw = normalized_to_raw(population, lower, upper)
    frame = pd.DataFrame(x_raw, columns=INPUT_COLS)
    frame.insert(0, "individual_id", np.arange(1, len(frame) + 1))
    frame.insert(0, "problem", problem)
    frame["rank"] = ranks
    frame["crowding_distance"] = crowding
    frame["constraint_violation"] = violations
    frame["is_feasible"] = (violations <= 1e-12).astype(int)
    for col in [*TARGET_COLS, "log_energy"]:
        frame[f"pred_{col}"] = predictions[col].to_numpy(float)
    frame["objective_1_score"] = scores[:, 0]
    frame["objective_2_score"] = scores[:, 1]
    frame["is_nondominated"] = (ranks == 0).astype(int)
    return frame


def select_representative_cases(nondominated: pd.DataFrame, max_per_problem: int) -> pd.DataFrame:
    pieces = []
    prefixes = {"purity_recovery": "PR", "productivity_energy": "PE"}
    sort_cols = {"purity_recovery": "pred_purity", "productivity_energy": "pred_productivity_mol_h_kg"}
    for problem, group in nondominated.groupby("problem", sort=False):
        feasible = group.loc[group["is_feasible"].eq(1)].copy()
        if feasible.empty:
            feasible = group.copy()
        ordered = feasible.sort_values(sort_cols[problem]).reset_index(drop=True)
        n_take = min(max_per_problem, len(ordered))
        if n_take == 0:
            continue
        indices = np.unique(np.rint(np.linspace(0, len(ordered) - 1, n_take)).astype(int))
        selected = ordered.iloc[indices].copy().reset_index(drop=True)
        selected.insert(0, "candidate_id", [f"{prefixes[problem]}_{idx + 1:03d}" for idx in range(len(selected))])
        pieces.append(selected)
    if not pieces:
        raise ValueError("No representative optimization candidates could be selected.")
    return pd.concat(pieces, ignore_index=True)


def build_summary(all_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for problem, group in all_results.groupby("problem", sort=False):
        front = group.loc[group["is_nondominated"].eq(1)]
        feasible_front = front.loc[front["is_feasible"].eq(1)]
        if feasible_front.empty:
            feasible_front = front
        rows.append(
            {
                "problem": problem,
                "n_population": int(len(group)),
                "n_nondominated": int(len(front)),
                "n_feasible_nondominated": int(front["is_feasible"].sum()),
                "purity_min": float(feasible_front["pred_purity"].min()),
                "purity_max": float(feasible_front["pred_purity"].max()),
                "recovery_min": float(feasible_front["pred_recovery"].min()),
                "recovery_max": float(feasible_front["pred_recovery"].max()),
                "productivity_min": float(feasible_front["pred_productivity_mol_h_kg"].min()),
                "productivity_max": float(feasible_front["pred_productivity_mol_h_kg"].max()),
                "energy_min": float(feasible_front["pred_energy_kJ_kgCO2"].min()),
                "energy_max": float(feasible_front["pred_energy_kJ_kgCO2"].max()),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NSGA-II optimization on the best scalar PSA surrogate.")
    parser.add_argument("--popsize", type=int, default=160)
    parser.add_argument("--generations", type=int, default=250)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--crossover-probability", type=float, default=0.9)
    parser.add_argument("--mutation-probability", type=float, default=1.0 / len(INPUT_COLS))
    parser.add_argument("--max-representative-cases", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.popsize < 4:
        raise ValueError("--popsize must be at least 4")
    artifact = load_artifact()
    samples = load_scalar_samples()
    lower, upper = bounds_from_artifact(artifact)

    frames = []
    for offset, problem in enumerate(PROBLEMS):
        frames.append(
            run_problem(
                problem=problem,
                artifact=artifact,
                samples=samples,
                lower=lower,
                upper=upper,
                popsize=args.popsize,
                generations=args.generations,
                seed=args.seed + 1000 * offset,
                crossover_probability=args.crossover_probability,
                mutation_probability=args.mutation_probability,
            )
        )

    all_results = pd.concat(frames, ignore_index=True)
    nondominated = all_results.loc[all_results["is_nondominated"].eq(1)].copy().reset_index(drop=True)
    representatives = select_representative_cases(nondominated, max_per_problem=args.max_representative_cases)
    summary = build_summary(all_results)

    all_path = DATA_DIR / "psa_optimization_all.csv"
    nondom_path = DATA_DIR / "psa_optimization_nondominated.csv"
    representative_path = DATA_DIR / "psa_optimization_representative_cases.csv"
    summary_path = DATA_DIR / "psa_optimization_summary.csv"
    all_results.to_csv(all_path, index=False)
    nondominated.to_csv(nondom_path, index=False)
    representatives.to_csv(representative_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Saved {all_path}")
    print(f"Saved {nondom_path}")
    print(f"Saved {representative_path}")
    print(f"Saved {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
