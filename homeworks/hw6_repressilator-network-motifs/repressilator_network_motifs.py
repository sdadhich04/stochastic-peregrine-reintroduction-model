"""Repressilator and two-gene network motif simulations from HW6."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OdeSolution:
    t: np.ndarray
    y: np.ndarray
    success: bool = True


def repressilator_odefun(
    _time: float,
    state: np.ndarray,
    alpha: float = 50.0,
    alpha0: float = 0.0,
    beta: float = 5.0,
    hill: float = 2.0,
    coupling: float = 0.0,
) -> np.ndarray:
    """Two coupled repressilators with diffusive protein coupling."""
    m1, m2, m3, p1, p2, p3, m4, m5, m6, p4, p5, p6 = state

    dm1 = -m1 + alpha / (1 + p3**hill) + alpha0
    dm2 = -m2 + alpha / (1 + p1**hill) + alpha0
    dm3 = -m3 + alpha / (1 + p2**hill) + alpha0
    dp1 = -beta * (p1 - m1) + coupling * (p4 - p1)
    dp2 = -beta * (p2 - m2) + coupling * (p5 - p2)
    dp3 = -beta * (p3 - m3) + coupling * (p6 - p3)

    dm4 = -m4 + alpha / (1 + p6**hill) + alpha0
    dm5 = -m5 + alpha / (1 + p4**hill) + alpha0
    dm6 = -m6 + alpha / (1 + p5**hill) + alpha0
    dp4 = -beta * (p4 - m4) + coupling * (p1 - p4)
    dp5 = -beta * (p5 - m5) + coupling * (p2 - p5)
    dp6 = -beta * (p6 - m6) + coupling * (p3 - p6)

    return np.array([dm1, dm2, dm3, dp1, dp2, dp3, dm4, dm5, dm6, dp4, dp5, dp6])


def simulate_repressilator(
    coupling: float = 0.0,
    t_end: float = 60.0,
    points: int = 600,
) -> OdeSolution:
    initial_state = np.array([1, 2, 3, 0, 0, 0, 2, 1, 3, 0, 0, 0], dtype=float)
    return integrate_ode(
        lambda time, state: repressilator_odefun(time, state, coupling=coupling),
        initial_state,
        t_end,
        points,
    )


def integrate_ode(
    function,
    initial_state: np.ndarray,
    t_end: float,
    points: int,
) -> OdeSolution:
    """Integrate an ODE with a fixed-step fourth-order Runge-Kutta method."""
    if points < 2:
        raise ValueError("points must be at least 2")

    t_values = np.linspace(0.0, t_end, points)
    y_values = np.empty((initial_state.size, points), dtype=float)
    y_values[:, 0] = initial_state

    for index in range(1, points):
        t = t_values[index - 1]
        h = t_values[index] - t
        y = y_values[:, index - 1]
        k1 = function(t, y)
        k2 = function(t + h / 2, y + h * k1 / 2)
        k3 = function(t + h / 2, y + h * k2 / 2)
        k4 = function(t + h, y + h * k3)
        y_values[:, index] = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return OdeSolution(t=t_values, y=y_values)


def terminal_sync_error(solution: OdeSolution) -> float:
    proteins_a = solution.y[3:6, -1]
    proteins_b = solution.y[9:12, -1]
    return float(np.linalg.norm(proteins_a - proteins_b))


def double_positive_motif(_time: float, state: np.ndarray) -> np.ndarray:
    x, y = state
    dx = -x + y**2 / (1 + y**2)
    dy = -y + x**2 / (1 + x**2)
    return np.array([dx, dy])


def double_negative_motif(_time: float, state: np.ndarray) -> np.ndarray:
    x, y = state
    dx = -x + 1 / (1 + y**2)
    dy = -y + 1 / (1 + x**2)
    return np.array([dx, dy])


def simulate_motif(
    motif: str,
    initial_state: tuple[float, float] = (0.2, 0.8),
    t_end: float = 40.0,
) -> np.ndarray:
    functions = {
        "double_positive": double_positive_motif,
        "double_negative": double_negative_motif,
    }
    if motif not in functions:
        raise ValueError(f"unknown motif: {motif}")
    solution = integrate_ode(
        functions[motif],
        np.array(initial_state, dtype=float),
        t_end,
        points=400,
    )
    return solution.y[:, -1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-end", type=float, default=60.0)
    args = parser.parse_args()

    test_state = np.ones(12)
    derivative = repressilator_odefun(0.0, test_state)
    print(f"test derivative first three entries: {derivative[:3]}")

    for coupling in (0.0, 0.1, -0.1):
        solution = simulate_repressilator(coupling=coupling, t_end=args.t_end)
        print(
            f"coupling={coupling:+.1f}, success={solution.success}, "
            f"terminal sync error={terminal_sync_error(solution):.6f}"
        )

    for motif in ("double_positive", "double_negative"):
        terminal = simulate_motif(motif)
        print(f"{motif} terminal state: {terminal}")


if __name__ == "__main__":
    main()
