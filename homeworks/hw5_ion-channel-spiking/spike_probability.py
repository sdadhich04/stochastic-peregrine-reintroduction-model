"""Ion-channel stationary distribution and spike-probability calculations."""

from __future__ import annotations

import argparse
from math import comb

import numpy as np


INWARD_CHANNEL = np.array(
    [
        [0.95, 0.20],
        [0.05, 0.80],
    ],
    dtype=float,
)

OUTWARD_CHANNEL = np.array(
    [
        [0.90, 0.10],
        [0.10, 0.90],
    ],
    dtype=float,
)


def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    """Return the stationary distribution for a column-stochastic chain."""
    if not np.allclose(transition.sum(axis=0), 1.0):
        raise ValueError("transition matrix must be column-stochastic")
    eigenvalues, eigenvectors = np.linalg.eig(transition)
    index = int(np.argmin(np.abs(eigenvalues - 1.0)))
    stationary = eigenvectors[:, index].real
    stationary = stationary / stationary.sum()
    return stationary


def exact_spike_probability(open_probability: float, channels: int, threshold: int) -> float:
    """Probability that at least threshold channels are open."""
    if not 0 <= threshold <= channels:
        raise ValueError("threshold must be between 0 and channels")
    return float(
        sum(
            comb(channels, open_count)
            * open_probability**open_count
            * (1 - open_probability) ** (channels - open_count)
            for open_count in range(threshold, channels + 1)
        )
    )


def monte_carlo_spike_probability(
    open_probability: float,
    channels: int,
    threshold: int,
    trials: int = 50_000,
    seed: int = 422,
) -> float:
    rng = np.random.default_rng(seed)
    open_counts = rng.binomial(channels, open_probability, size=trials)
    return float(np.mean(open_counts >= threshold))


def scan_channel_counts(
    open_probability: float,
    min_channels: int = 5,
    max_channels: int = 100,
    threshold_fraction: float = 0.5,
) -> list[tuple[int, int, float]]:
    results = []
    for channels in range(min_channels, max_channels + 1, 5):
        threshold = int(np.ceil(threshold_fraction * channels))
        probability = exact_spike_probability(open_probability, channels, threshold)
        results.append((channels, threshold, probability))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, default=40)
    parser.add_argument("--threshold-fraction", type=float, default=0.5)
    parser.add_argument("--trials", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=422)
    args = parser.parse_args()

    inward_stationary = stationary_distribution(INWARD_CHANNEL)
    outward_stationary = stationary_distribution(OUTWARD_CHANNEL)
    open_probability = float(inward_stationary[1])
    threshold = int(np.ceil(args.threshold_fraction * args.channels))
    exact = exact_spike_probability(open_probability, args.channels, threshold)
    simulated = monte_carlo_spike_probability(
        open_probability, args.channels, threshold, args.trials, args.seed
    )

    print(f"inward stationary distribution: {inward_stationary}")
    print(f"outward stationary distribution: {outward_stationary}")
    print(f"open probability used for spike scan: {open_probability:.6f}")
    print(f"channels: {args.channels}, threshold: {threshold}")
    print(f"exact spike probability: {exact:.6f}")
    print(f"monte carlo spike probability: {simulated:.6f}")
    print("selected channel-count scan:")
    for channels, count_threshold, probability in scan_channel_counts(open_probability)[::4]:
        print(f"  n={channels:3d}, threshold={count_threshold:3d}, p={probability:.6f}")


if __name__ == "__main__":
    main()
