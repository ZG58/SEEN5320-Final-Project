from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from psa_surrogate_utils import (
    DATA_DIR,
    DESIGN_COLS,
    PREDICTION_TARGETS,
    RANDOM_STATE,
    bounds_arrays,
    load_manifest,
    load_model,
    objective_scores,
    predict_surrogate,
    variable_bounds,
)


PROBLEMS = ["purity_recovery", "productivity_energy"]


def dominates(scores: np.ndarray, i: int, j: int) -> bool:
    return bool(np.all(scores[i] >= scores[j]) and np.any(scores[i] > scores[j]))


def fast_nondominated_sort(scores: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
    n_rows = scores.shape[0]
    dominated_sets: list[list[int]] = [[] for _ in range(n_rows)]
    domination_counts = np.zeros(n_rows, dtype=int)
    ranks = np.full(n_rows, -1, dtype=int)
    fronts: list[list[int]] = [[]]

    for p in range(n_rows):
        for q in range(n_rows):
            if p == q:
                continue
            if dominates(scores, p, q):
                dominated_sets[p].append(q)
            elif dominates(scores, q, p):
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
    n_obj = scores.shape[1]
    for front in fronts:
        if not front:
            continue
        if len(front) <= 2:
            distances[front] = np.inf
            continue
        front_scores = scores[front]
        front_dist = np.zeros(len(front), dtype=float)
        for obj_idx in range(n_obj):
            order = np.argsort(front_scores[:, obj_idx])
            front_dist[order[0]] = np.inf
            front_dist[order[-1]] = np.inf
            lo = front_scores[order[0], obj_idx]
            hi = front_scores[order[-1], obj_idx]
            if hi == lo:
                continue
            for k in range(1, len(front) - 1):
                front_dist[order[k]] += (front_scores[order[k + 1], obj_idx] - front_scores[order[k - 1], obj_idx]) / (hi - lo)
        for local_idx, global_idx in enumerate(front):
            distances[global_idx] = front_dist[local_idx]
    return distances


def rank_and_crowding(scores: np.ndarray) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    fronts, ranks = fast_nondominated_sort(scores)
    crowding = crowding_distance(scores, fronts)
    return fronts, ranks, crowding


def normalized_to_raw(population: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return lower + population * (upper - lower)


def evaluate_population(
    population: np.ndarray,
    model: dict,
    lower: np.ndarray,
    upper: np.ndarray,
    problem: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    x_raw = normalized_to_raw(population, lower, upper)
    predictions = predict_surrogate(model, x_raw)
    scores = objective_scores(predictions, problem)
    return predictions, scores


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
    probability: float = 0.9,
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

        beta = 1.0 + 2.0 * (x1 - 0.0) / (x2 - x1)
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
            val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
            deltaq = val**mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
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
        p1 = population[tournament_select(ranks, crowding, rng)]
        p2 = population[tournament_select(ranks, crowding, rng)]
        c1, c2 = sbx_crossover(p1, p2, rng, probability=crossover_probability)
        children.append(polynomial_mutation(c1, rng, probability=mutation_probability))
        if len(children) < len(population):
            children.append(polynomial_mutation(c2, rng, probability=mutation_probability))
    return np.vstack(children)


def select_next_generation(population: np.ndarray, scores: np.ndarray, popsize: int) -> np.ndarray:
    fronts, _, crowding = rank_and_crowding(scores)
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
    manifest: pd.DataFrame,
    lower: np.ndarray,
    upper: np.ndarray,
    popsize: int,
    rng: np.random.Generator,
) -> np.ndarray:
    manifest_norm = (manifest[DESIGN_COLS].to_numpy(float) - lower) / (upper - lower)
    manifest_norm = np.clip(manifest_norm, 0.0, 1.0)
    n_seed = min(popsize // 2, len(manifest_norm))
    seed_idx = rng.choice(len(manifest_norm), size=n_seed, replace=False)
    random_part = rng.random((popsize - n_seed, len(DESIGN_COLS)))
    return np.vstack([manifest_norm[seed_idx], random_part])


def run_problem(
    problem: str,
    model: dict,
    manifest: pd.DataFrame,
    lower: np.ndarray,
    upper: np.ndarray,
    popsize: int,
    generations: int,
    seed: int,
    crossover_probability: float,
    mutation_probability: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    population = initial_population(manifest, lower, upper, popsize, rng)

    for generation in range(generations):
        _, scores = evaluate_population(population, model, lower, upper, problem)
        _, ranks, crowding = rank_and_crowding(scores)
        offspring = make_offspring(
            population,
            ranks,
            crowding,
            rng,
            crossover_probability=crossover_probability,
            mutation_probability=mutation_probability,
        )
        combined = np.vstack([population, offspring])
        _, combined_scores = evaluate_population(combined, model, lower, upper, problem)
        population = select_next_generation(combined, combined_scores, popsize)
        if generation == 0 or (generation + 1) % 50 == 0 or generation + 1 == generations:
            current_front = int((rank_and_crowding(evaluate_population(population, model, lower, upper, problem)[1])[1] == 0).sum())
            print(f"{problem}: generation {generation + 1}/{generations}, rank-0 points={current_front}")

    predictions, scores = evaluate_population(population, model, lower, upper, problem)
    fronts, ranks, crowding = rank_and_crowding(scores)
    x_raw = normalized_to_raw(population, lower, upper)
    frame = pd.DataFrame(x_raw, columns=DESIGN_COLS)
    frame.insert(0, "individual_id", np.arange(1, len(frame) + 1))
    frame.insert(0, "problem", problem)
    frame["rank"] = ranks
    frame["crowding_distance"] = crowding
    for target in PREDICTION_TARGETS:
        frame[f"pred_{target}"] = predictions[target].to_numpy(float)
    frame["objective_1_score"] = scores[:, 0]
    frame["objective_2_score"] = scores[:, 1]
    frame["is_nondominated"] = frame["rank"].eq(0).astype(int)
    return frame


def select_detailed_candidates(nondominated: pd.DataFrame, max_per_problem: int) -> pd.DataFrame:
    selected_frames = []
    prefix = {"purity_recovery": "PR", "productivity_energy": "PE"}
    sort_key = {
        "purity_recovery": "pred_purity",
        "productivity_energy": "pred_productivity_mol_kg_h",
    }
    for problem, group in nondominated.groupby("problem", sort=False):
        ordered = group.sort_values(sort_key[problem]).reset_index(drop=True)
        n_take = min(max_per_problem, len(ordered))
        if n_take == 0:
            continue
        indices = np.unique(np.rint(np.linspace(0, len(ordered) - 1, n_take)).astype(int))
        selected = ordered.iloc[indices].copy().reset_index(drop=True)
        selected.insert(0, "candidate_id", [f"{prefix[problem]}_{idx + 1:03d}" for idx in range(len(selected))])
        selected_frames.append(selected)
    if not selected_frames:
        raise ValueError("No nondominated candidates available for detailed-model export.")
    return pd.concat(selected_frames, ignore_index=True)


def write_detailed_template(candidates: pd.DataFrame) -> None:
    input_cols = [
        "candidate_id",
        "problem",
        *DESIGN_COLS,
        "pred_purity",
        "pred_recovery",
        "pred_productivity_mol_kg_h",
        "pred_energy_kWh_ton",
        "pred_log_energy",
    ]
    candidates[input_cols].to_csv(DATA_DIR / "detailed_model_input.csv", index=False)

    template = candidates[["candidate_id", "problem", *DESIGN_COLS]].copy()
    template["status"] = ""
    template["purity"] = ""
    template["recovery"] = ""
    template["productivity_mol_kg_h"] = ""
    template["energy_kWh_ton"] = ""
    template["runtime_s"] = ""
    template["notes"] = ""
    template.to_csv(DATA_DIR / "detailed_model_results_template.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NSGA-II surrogate-assisted PSA optimization.")
    parser.add_argument("--popsize", type=int, default=120)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--crossover-probability", type=float, default=0.9)
    parser.add_argument("--mutation-probability", type=float, default=1.0 / len(DESIGN_COLS))
    parser.add_argument("--max-detailed-candidates", type=int, default=20)
    parser.add_argument("--output-prefix", default="ga_optimization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.popsize < 4:
        raise ValueError("--popsize must be at least 4")
    model = load_model()
    manifest = load_manifest()
    bounds = variable_bounds(manifest)
    lower, upper = bounds_arrays(bounds)

    frames = []
    for offset, problem in enumerate(PROBLEMS):
        frames.append(
            run_problem(
                problem=problem,
                model=model,
                manifest=manifest,
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

    all_path = DATA_DIR / f"{args.output_prefix}_all.csv"
    nondom_path = DATA_DIR / f"{args.output_prefix}_nondominated.csv"
    all_results.to_csv(all_path, index=False)
    nondominated.to_csv(nondom_path, index=False)
    print(f"Saved {all_path}")
    print(f"Saved {nondom_path}")

    if args.output_prefix == "ga_optimization":
        candidates = select_detailed_candidates(nondominated, max_per_problem=args.max_detailed_candidates)
        write_detailed_template(candidates)
        print(f"Saved {DATA_DIR / 'detailed_model_input.csv'}")
        print(f"Saved {DATA_DIR / 'detailed_model_results_template.csv'}")


if __name__ == "__main__":
    main()
