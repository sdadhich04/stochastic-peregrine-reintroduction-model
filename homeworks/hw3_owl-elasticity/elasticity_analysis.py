"""Northern spotted owl Leslie-matrix elasticity analysis.

This script refactors the recovered HW3 notebook into a small, runnable module.
It computes the dominant growth rate, stable age distribution, reproductive
value, and elasticity matrix for an age-structured owl population model.
"""

from __future__ import annotations

import argparse

import numpy as np


def build_leslie_matrix(
    max_age: int = 50,
    fecundity: float = 0.24,
    adult_survival: float = 0.942,
    juvenile_to_age3: float = 0.0722,
) -> np.ndarray:
    """Build the age-structured matrix used in the recovered HW3 notebook."""
    if max_age < 4:
        raise ValueError("max_age must include juvenile and adult age classes")

    matrix = np.zeros((max_age, max_age), dtype=float)
    matrix[0, 2:] = fecundity
    matrix[1, 0] = juvenile_to_age3
    for age in range(1, max_age - 1):
        matrix[age + 1, age] = adult_survival
    return matrix


def dominant_eigenvalue_and_vectors(matrix: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return dominant lambda, stable age distribution, and reproductive value."""
    eigenvalues, right_vectors = np.linalg.eig(matrix)
    dominant_index = int(np.argmax(eigenvalues.real))
    lambda_value = float(eigenvalues[dominant_index].real)

    stable = right_vectors[:, dominant_index].real
    stable = stable / stable.sum()

    left_values, left_vectors = np.linalg.eig(matrix.T)
    left_index = int(np.argmin(np.abs(left_values.real - lambda_value)))
    reproductive_value = left_vectors[:, left_index].real
    reproductive_value = reproductive_value / reproductive_value[0]

    return lambda_value, stable, reproductive_value


def elasticity_matrix(matrix: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Compute matrix elasticity using right and left dominant eigenvectors."""
    lambda_value, stable, reproductive_value = dominant_eigenvalue_and_vectors(matrix)
    denominator = float(np.dot(reproductive_value, stable))
    sensitivity = np.outer(reproductive_value, stable) / denominator
    elasticity = matrix / lambda_value * sensitivity
    return elasticity, lambda_value, stable, reproductive_value


def summarize_elasticity(matrix: np.ndarray) -> dict[str, float]:
    elasticity, lambda_value, stable, reproductive_value = elasticity_matrix(matrix)
    fecundity_elasticity = float(elasticity[0, 2:].sum())
    survival_elasticity = float(elasticity[1:, :-1].sum())
    return {
        "lambda": lambda_value,
        "juvenile_fraction": float(stable[0]),
        "adult_fraction": float(stable[2:].sum()),
        "age_3_reproductive_value": float(reproductive_value[2]),
        "fecundity_elasticity": fecundity_elasticity,
        "survival_elasticity": survival_elasticity,
        "elasticity_sum": float(elasticity.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-age", type=int, default=50)
    args = parser.parse_args()

    metrics = summarize_elasticity(build_leslie_matrix(max_age=args.max_age))
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
