"""Stochastic peregrine falcon population viability simulation.

This module is a cleaned, scriptable version of the AMATH 422/522 final
project notebooks. It models stage-structured peregrine falcon populations
across urban, Southern Illinois cliff, and optional bridge/lower-Midwest cliff
patches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class HabitatParameters:
    juvenile_survival: float
    subadult_survival: float
    adult_survival: float
    subadult_fecundity: float
    adult_fecundity: float
    juvenile_survival_var: float
    subadult_survival_var: float
    adult_survival_var: float
    subadult_fecundity_var: float
    adult_fecundity_var: float
    r_max: float
    carrying_capacity: float


@dataclass(frozen=True)
class Patch:
    name: str
    habitat: str
    initial_adults: float = 0.0


@dataclass(frozen=True)
class Scenario:
    name: str
    patches: tuple[Patch, ...]
    distances: dict[tuple[str, str], float]
    release_patch: str = "southern_illinois_cliff"


DEFAULT_HABITATS: dict[str, HabitatParameters] = {
    "cliff": HabitatParameters(
        juvenile_survival=0.20,
        subadult_survival=0.84,
        adult_survival=0.85,
        subadult_fecundity=0.76,
        adult_fecundity=0.77,
        juvenile_survival_var=0.03,
        subadult_survival_var=0.13,
        adult_survival_var=0.04,
        subadult_fecundity_var=0.15,
        adult_fecundity_var=0.09,
        r_max=1.002,
        carrying_capacity=16,
    ),
    "urban": HabitatParameters(
        juvenile_survival=0.24,
        subadult_survival=0.85,
        adult_survival=0.85,
        subadult_fecundity=1.11,
        adult_fecundity=1.11,
        juvenile_survival_var=0.02,
        subadult_survival_var=0.07,
        adult_survival_var=0.02,
        subadult_fecundity_var=0.11,
        adult_fecundity_var=0.06,
        r_max=1.094,
        carrying_capacity=171,
    ),
    "lower_midwest_cliff": HabitatParameters(
        juvenile_survival=0.20,
        subadult_survival=0.84,
        adult_survival=0.85,
        subadult_fecundity=0.76,
        adult_fecundity=0.77,
        juvenile_survival_var=0.03,
        subadult_survival_var=0.13,
        adult_survival_var=0.04,
        subadult_fecundity_var=0.15,
        adult_fecundity_var=0.09,
        r_max=1.002,
        carrying_capacity=4,
    ),
}


def _sample_survival(mean: float, variance: float, rng: np.random.Generator) -> float:
    return float(np.clip(rng.normal(mean, np.sqrt(variance)), 0.0, 1.0))


def _sample_fecundity(mean: float, variance: float, rng: np.random.Generator) -> float:
    return float(max(rng.normal(mean, np.sqrt(variance)), 0.0))


def density_dependence(adults: float, r_max: float, carrying_capacity: float) -> float:
    if adults <= 0:
        return 1.0
    realized_growth = (r_max * carrying_capacity) / (
        (r_max * adults) - adults + carrying_capacity
    )
    return max(realized_growth / r_max, 0.0)


def projection_matrix(
    adults: float, params: HabitatParameters, rng: np.random.Generator
) -> np.ndarray:
    factor = density_dependence(adults, params.r_max, params.carrying_capacity)
    juvenile_survival = _sample_survival(
        params.juvenile_survival, params.juvenile_survival_var, rng
    )
    subadult_survival = _sample_survival(
        params.subadult_survival, params.subadult_survival_var, rng
    )
    adult_survival = _sample_survival(
        params.adult_survival, params.adult_survival_var, rng
    )
    subadult_fecundity = _sample_fecundity(
        params.subadult_fecundity, params.subadult_fecundity_var, rng
    )
    adult_fecundity = _sample_fecundity(
        params.adult_fecundity, params.adult_fecundity_var, rng
    )
    return factor * np.array(
        [
            [0.0, subadult_fecundity, adult_fecundity],
            [juvenile_survival, 0.0, 0.0],
            [0.0, subadult_survival, adult_survival],
        ]
    )


def stable_stage_population(adults: float, params: HabitatParameters) -> np.ndarray:
    matrix = np.array(
        [
            [0.0, params.subadult_fecundity, params.adult_fecundity],
            [params.juvenile_survival, 0.0, 0.0],
            [0.0, params.subadult_survival, params.adult_survival],
        ]
    )
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    index = np.argmax(eigenvalues.real)
    stable = eigenvectors[:, index].real
    stable = stable / stable.sum()
    return np.round(adults * stable / stable[2]).astype(int)


def demographic_step(
    population: np.ndarray, matrix: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    juveniles, subadults, adults = population.astype(int)
    next_subadults = rng.binomial(juveniles, np.clip(matrix[1, 0], 0, 1))
    next_adults = rng.binomial(subadults, np.clip(matrix[2, 1], 0, 1))
    next_adults += rng.binomial(adults, np.clip(matrix[2, 2], 0, 1))
    expected_juveniles = matrix[0, 1] * subadults + matrix[0, 2] * adults
    next_juveniles = rng.poisson(max(expected_juveniles, 0.0))
    return np.array([next_juveniles, next_subadults, next_adults], dtype=int)


def dispersal_probability(
    distance: float, source_habitat: str, target_habitat: str
) -> float:
    habitat_penalty = 1.0
    cliff_habitats = {"cliff", "lower_midwest_cliff"}
    if source_habitat in cliff_habitats and target_habitat == "urban":
        habitat_penalty = 1.0 - 0.67
    elif source_habitat == "urban" and target_habitat in cliff_habitats:
        habitat_penalty = 1.0 - 0.90
    return habitat_penalty * (1.08 * np.exp(-0.01 * distance))


def disperse_juveniles(
    populations: dict[str, np.ndarray],
    scenario: Scenario,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    next_populations = {name: pop.copy() for name, pop in populations.items()}

    for source in scenario.patches:
        juveniles = int(populations[source.name][0])
        if juveniles <= 0:
            continue
        targets = [patch for patch in scenario.patches if patch.name != source.name]
        weights = []
        for target in targets:
            distance = scenario.distances.get((source.name, target.name))
            if distance is None:
                distance = scenario.distances.get((target.name, source.name))
            if distance is None:
                weights.append(0.0)
                continue
            weights.append(
                dispersal_probability(distance, source.habitat, target.habitat)
            )

        total_out = sum(weights)
        if total_out > 1.0:
            weights = [weight / total_out for weight in weights]
            total_out = 1.0
        draws = rng.multinomial(juveniles, [*weights, 1.0 - total_out])
        next_populations[source.name][0] -= int(draws[:-1].sum())
        for target, moved in zip(targets, draws[:-1]):
            next_populations[target.name][0] += int(moved)

    return next_populations


def release_schedule(
    years: int, juveniles_per_release: int = 0, release_years: Iterable[int] = ()
) -> np.ndarray:
    releases = np.zeros(years, dtype=int)
    for year in release_years:
        if 0 <= year < years:
            releases[year] = juveniles_per_release
    return releases


def simulate(
    scenario: Scenario,
    years: int = 50,
    replicates: int = 1000,
    releases: np.ndarray | None = None,
    seed: int | None = None,
    habitats: dict[str, HabitatParameters] = DEFAULT_HABITATS,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    releases = np.zeros(years, dtype=int) if releases is None else releases
    patch_names = [patch.name for patch in scenario.patches]
    history = {
        name: np.zeros((replicates, years + 1, 3), dtype=float) for name in patch_names
    }
    colonization_years: list[int] = []

    for replicate in range(replicates):
        populations = {}
        for patch in scenario.patches:
            if patch.initial_adults > 0:
                populations[patch.name] = stable_stage_population(
                    patch.initial_adults, habitats[patch.habitat]
                )
            else:
                populations[patch.name] = np.zeros(3, dtype=int)
            history[patch.name][replicate, 0, :] = populations[patch.name]

        colonized_at = None
        for year in range(years):
            stepped = {}
            for patch in scenario.patches:
                params = habitats[patch.habitat]
                matrix = projection_matrix(populations[patch.name][2], params, rng)
                stepped[patch.name] = demographic_step(
                    populations[patch.name], matrix, rng
                )

            populations = disperse_juveniles(stepped, scenario, rng)
            populations[scenario.release_patch][0] += int(releases[year])

            for patch_name in patch_names:
                history[patch_name][replicate, year + 1, :] = populations[patch_name]

            release_patch_adults = populations[scenario.release_patch][2]
            if colonized_at is None and release_patch_adults > 0:
                colonized_at = year + 1

        if colonized_at is not None:
            colonization_years.append(colonized_at)

    return {
        "scenario": scenario.name,
        "history": history,
        "colonization_years": np.array(colonization_years, dtype=float),
    }


def summarize(result: dict[str, object], focal_patch: str) -> dict[str, float]:
    history = result["history"]
    if not isinstance(history, dict):
        raise TypeError("result['history'] must be a patch history dictionary")
    adult_histories = [values[:, :, 2] for values in history.values()]
    total_adults = np.sum(adult_histories, axis=0)
    min_expected_adults = float(total_adults.min(axis=1).mean())
    focal_terminal_adults = float(history[focal_patch][:, -1, 2].mean())
    colonization_years = result["colonization_years"]
    if not isinstance(colonization_years, np.ndarray):
        raise TypeError("result['colonization_years'] must be an ndarray")
    median_colonization = (
        float(np.median(colonization_years)) if colonization_years.size else float("nan")
    )
    return {
        "expected_minimum_adults": min_expected_adults,
        "median_time_to_colonization": median_colonization,
        "terminal_focal_adults": focal_terminal_adults,
    }


BASELINE_SCENARIO = Scenario(
    name="urban-to-southern-illinois-cliff",
    patches=(
        Patch("southern_illinois_cliff", "cliff"),
        Patch("urban", "urban", initial_adults=31),
    ),
    distances={("southern_illinois_cliff", "urban"): 250},
)


BRIDGE_CLIFF_SCENARIO = Scenario(
    name="bridge-cliff-between-urban-and-southern-illinois",
    patches=(
        Patch("southern_illinois_cliff", "cliff"),
        Patch("urban", "urban", initial_adults=31),
        Patch("bridge_cliff", "lower_midwest_cliff"),
    ),
    distances={
        ("southern_illinois_cliff", "urban"): 300,
        ("southern_illinois_cliff", "bridge_cliff"): 260,
        ("bridge_cliff", "urban"): 70,
    },
)


LOWER_MIDWEST_MULTI_CLIFF_SCENARIO = Scenario(
    name="lower-midwest-multiple-cliffs",
    patches=(
        Patch("southern_illinois_cliff_1", "cliff"),
        Patch("southern_illinois_cliff_2", "cliff"),
        Patch("urban", "urban", initial_adults=31),
        Patch("lower_midwest_cliff_1", "lower_midwest_cliff"),
        Patch("lower_midwest_cliff_2", "lower_midwest_cliff"),
        Patch("lower_midwest_cliff_3", "lower_midwest_cliff"),
    ),
    distances={
        ("southern_illinois_cliff_1", "urban"): 370,
        ("southern_illinois_cliff_2", "urban"): 370,
        ("southern_illinois_cliff_1", "southern_illinois_cliff_2"): 70,
        ("southern_illinois_cliff_1", "lower_midwest_cliff_1"): 330,
        ("southern_illinois_cliff_1", "lower_midwest_cliff_2"): 320,
        ("southern_illinois_cliff_1", "lower_midwest_cliff_3"): 360,
        ("southern_illinois_cliff_2", "lower_midwest_cliff_1"): 330,
        ("southern_illinois_cliff_2", "lower_midwest_cliff_2"): 330,
        ("southern_illinois_cliff_2", "lower_midwest_cliff_3"): 380,
        ("lower_midwest_cliff_1", "urban"): 70,
        ("lower_midwest_cliff_2", "urban"): 70,
        ("lower_midwest_cliff_3", "urban"): 70,
        ("lower_midwest_cliff_1", "lower_midwest_cliff_2"): 150,
        ("lower_midwest_cliff_1", "lower_midwest_cliff_3"): 150,
        ("lower_midwest_cliff_2", "lower_midwest_cliff_3"): 70,
    },
    release_patch="southern_illinois_cliff_1",
)
