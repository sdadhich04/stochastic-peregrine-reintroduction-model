"""Run a compact set of peregrine falcon PVA scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from falcon_pva import (  # noqa: E402
    BASELINE_SCENARIO,
    BRIDGE_CLIFF_SCENARIO,
    LOWER_MIDWEST_MULTI_CLIFF_SCENARIO,
    release_schedule,
    simulate,
    summarize,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=50)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=422)
    args = parser.parse_args()

    scenarios = [
        (BASELINE_SCENARIO, "southern_illinois_cliff"),
        (BRIDGE_CLIFF_SCENARIO, "southern_illinois_cliff"),
        (LOWER_MIDWEST_MULTI_CLIFF_SCENARIO, "southern_illinois_cliff_1"),
    ]
    release_plans = {
        "no_release": release_schedule(args.years),
        "release_8_every_3_years": release_schedule(
            args.years, juveniles_per_release=8, release_years=range(0, 10, 3)
        ),
        "release_16_every_3_years": release_schedule(
            args.years, juveniles_per_release=16, release_years=range(0, 10, 3)
        ),
        "release_24_every_3_years": release_schedule(
            args.years, juveniles_per_release=24, release_years=range(0, 10, 3)
        ),
    }

    for scenario, focal_patch in scenarios:
        print(f"\n{scenario.name}")
        for label, releases in release_plans.items():
            result = simulate(
                scenario,
                years=args.years,
                replicates=args.replicates,
                releases=releases,
                seed=args.seed,
            )
            metrics = summarize(result, focal_patch)
            print(
                f"  {label}: "
                f"EMA={metrics['expected_minimum_adults']:.2f}, "
                f"median_colonization={metrics['median_time_to_colonization']:.1f}, "
                f"terminal_focal_adults={metrics['terminal_focal_adults']:.2f}"
            )


if __name__ == "__main__":
    main()
