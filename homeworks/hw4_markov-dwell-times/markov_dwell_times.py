"""Ion-channel Markov-chain dwell-time analysis from HW4."""

from __future__ import annotations

import argparse

import numpy as np


TRANSITION_MATRIX = np.array(
    [
        [0.98, 0.10, 0.00],
        [0.02, 0.70, 0.05],
        [0.00, 0.20, 0.95],
    ],
    dtype=float,
)

STATE_NAMES = ("closed_1", "closed_2", "open")
OPEN_STATE = 2


def simulate_chain(
    transition: np.ndarray = TRANSITION_MATRIX,
    steps: int = 100_000,
    seed: int = 422,
    initial_state: int = 0,
) -> np.ndarray:
    """Simulate a column-stochastic Markov chain."""
    rng = np.random.default_rng(seed)
    column_sums = transition.sum(axis=0)
    if not np.allclose(column_sums, 1.0):
        raise ValueError("transition matrix must be column-stochastic")

    states = np.empty(steps, dtype=int)
    states[0] = initial_state
    for index in range(1, steps):
        probabilities = transition[:, states[index - 1]]
        states[index] = int(rng.choice(len(probabilities), p=probabilities))
    return states


def dwell_times(states: np.ndarray, observed_state: int = OPEN_STATE) -> np.ndarray:
    """Return consecutive dwell lengths for one observed state."""
    runs: list[int] = []
    current = 0
    for state in states:
        if state == observed_state:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return np.array(runs, dtype=float)


def empirical_survival(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert dwell samples into an empirical survival curve."""
    if times.size == 0:
        raise ValueError("no dwell times were observed")
    durations = np.arange(1, int(times.max()) + 1, dtype=float)
    survival = np.array([(times >= duration).mean() for duration in durations])
    return durations, survival


def exponential_survival(t: np.ndarray, rate: float) -> np.ndarray:
    return np.exp(-rate * t)


def fit_exponential_rate(times: np.ndarray) -> float:
    durations, survival = empirical_survival(times)
    usable = survival > 0
    slope, _intercept = np.polyfit(durations[usable], np.log(survival[usable]), 1)
    return float(-slope)


def collapse_closed_states(states: np.ndarray) -> np.ndarray:
    """Map the two closed states to 0 and the open state to 1."""
    return (states == OPEN_STATE).astype(int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=422)
    args = parser.parse_args()

    states = simulate_chain(steps=args.steps, seed=args.seed)
    open_dwell = dwell_times(states, OPEN_STATE)
    collapsed = collapse_closed_states(states)
    closed_dwell = dwell_times(collapsed, observed_state=0)
    open_rate = fit_exponential_rate(open_dwell)

    print(f"states: {', '.join(STATE_NAMES)}")
    print(f"open dwell count: {open_dwell.size}")
    print(f"mean open dwell time: {open_dwell.mean():.3f}")
    print(f"single-exponential open survival rate: {open_rate:.5f}")
    print(f"mean collapsed closed dwell time: {closed_dwell.mean():.3f}")


if __name__ == "__main__":
    main()
