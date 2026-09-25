#!/usr/bin/env python3
"""Decision-support Monte Carlo for BENCH-001B execution topology.

Engineering decision support only. Scores and weights are explicit priors,
not scientific probabilities or provider reliability forecasts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


OPTIONS = [
    "A github-t4-larger-runner",
    "B persistent-public-self-hosted-gpu",
    "C isolated-approved-gpu-plus-actions-validator",
    "D defer-all-work-until-gpu-exists",
]

CRITERIA = [
    "source_fidelity",
    "public_repo_security",
    "paid_resource_control",
    "current_feasibility",
    "provenance_quality",
    "reproducibility",
    "automation",
    "research_progress",
]

BASE_SCORES = np.array(
    [
        [3.0, 8.0, 4.0, 5.0, 8.0, 8.0, 10.0, 6.0],
        [9.0, 3.0, 8.0, 8.0, 7.0, 7.0, 9.0, 8.0],
        [10.0, 9.0, 8.0, 7.0, 10.0, 10.0, 7.0, 9.0],
        [10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 1.0, 1.0],
    ],
    dtype=float,
)

SCENARIOS = {
    "balanced": [0.18, 0.15, 0.12, 0.12, 0.13, 0.12, 0.08, 0.10],
    "source_fidelity_heavy": [0.32, 0.12, 0.08, 0.09, 0.12, 0.10, 0.06, 0.11],
    "public_repo_security_heavy": [0.12, 0.31, 0.12, 0.10, 0.12, 0.09, 0.05, 0.09],
    "paid_resource_control_heavy": [0.10, 0.14, 0.30, 0.16, 0.08, 0.07, 0.05, 0.10],
    "automation_heavy": [0.12, 0.10, 0.07, 0.09, 0.10, 0.10, 0.32, 0.10],
}


def simulate_one(
    *,
    samples: int,
    seed: int,
    weights0: list[float],
    score_sigma: float,
    concentration: float,
) -> dict:
    rng = np.random.default_rng(seed)
    base_weights = np.asarray(weights0, dtype=float)
    base_weights /= base_weights.sum()

    weights = rng.dirichlet(base_weights * concentration, size=samples)
    noise = rng.normal(
        0.0,
        score_sigma,
        size=(samples, len(OPTIONS), len(CRITERIA)),
    )
    scores = np.clip(BASE_SCORES[None, :, :] + noise, 0.0, 10.0)
    utility = (scores * weights[:, None, :]).sum(axis=2)
    winners = np.argmax(utility, axis=1)

    rates = {
        name: float(np.mean(winners == index))
        for index, name in enumerate(OPTIONS)
    }
    return {
        "weights": dict(zip(CRITERIA, base_weights.tolist())),
        "win_rates": rates,
        "winner": max(rates, key=rates.get),
    }


def simulate(
    *,
    samples: int,
    seed: int,
    score_sigma: float,
    concentration: float,
) -> dict:
    scenarios = {}
    for offset, (name, weights) in enumerate(SCENARIOS.items()):
        scenarios[name] = simulate_one(
            samples=samples,
            seed=seed + offset,
            weights0=weights,
            score_sigma=score_sigma,
            concentration=concentration,
        )
    return {
        "kind": "engineering_decision_support",
        "scientific_probability": False,
        "samples_per_scenario": samples,
        "base_seed": seed,
        "score_sigma": score_sigma,
        "dirichlet_concentration": concentration,
        "criteria": CRITERIA,
        "options": OPTIONS,
        "base_scores": BASE_SCORES.tolist(),
        "scenarios": scenarios,
    }


def markdown(result: dict) -> str:
    lines = [
        "# BENCH-001B execution-topology decision support",
        "",
        "> Engineering decision support only. These are not scientific probabilities.",
        "",
        f"- samples per scenario: {result['samples_per_scenario']}",
        f"- base seed: {result['base_seed']}",
        "",
        "| Scenario | Winner | A | B | C | D |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, scenario in result["scenarios"].items():
        rates = scenario["win_rates"]
        lines.append(
            f"| {name} | {scenario['winner']} | "
            + " | ".join(f"{rates[option]:.2%}" for option in OPTIONS)
            + " |"
        )
    lines.extend(
        [
            "",
            "Default choice:",
            "",
            "~~~text",
            "C isolated-approved-gpu-plus-actions-validator",
            "~~~",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--score-sigma", type=float, default=1.0)
    p.add_argument("--concentration", type=float, default=50.0)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/decision-support/bench001b-execution"),
    )
    args = p.parse_args()

    result = simulate(
        samples=args.samples,
        seed=args.seed,
        score_sigma=args.score_sigma,
        concentration=args.concentration,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "result.md").write_text(markdown(result) + "\n")
    print(markdown(result))

    required = [
        "balanced",
        "source_fidelity_heavy",
        "public_repo_security_heavy",
        "paid_resource_control_heavy",
        "automation_heavy",
    ]
    if any(
        result["scenarios"][name]["winner"]
        != "C isolated-approved-gpu-plus-actions-validator"
        for name in required
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
