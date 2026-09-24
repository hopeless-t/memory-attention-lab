#!/usr/bin/env python3
"""Decision-support Monte Carlo for Stage 0 research sequencing.

This is engineering decision support, not a scientific probability model.
Scores and weights are explicit priors used to stress-test whether the
preferred sequence is stable under reasonable perturbations.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np


SEQUENCES = {
    "S1 lineage-first controls": [
        "VAL-001",
        "VAL-002",
        "BENCH-001",
        "BENCH-002",
        "EXP-001",
    ],
    "S2 systems-first": [
        "VAL-001",
        "BENCH-001",
        "BENCH-002",
        "VAL-002",
        "EXP-001",
    ],
    "S3 recall-first after controls": [
        "VAL-001",
        "VAL-002",
        "BENCH-002",
        "BENCH-001",
        "EXP-001",
    ],
    "S4 mixed": [
        "VAL-001",
        "BENCH-001",
        "VAL-002",
        "BENCH-002",
        "EXP-001",
    ],
}

CRITERIA = [
    "causal_identifiability",
    "prerequisite_coherence",
    "early_low_cost_evidence",
    "systems_relevance",
    "novelty_pacing",
    "reproducibility",
]

# Scores are 0..10 engineering priors, not measured scientific quantities.
BASE_SCORES = {
    "S1 lineage-first controls": [10.0, 10.0, 9.0, 8.5, 8.5, 10.0],
    "S2 systems-first": [7.0, 7.5, 8.5, 10.0, 9.0, 8.0],
    "S3 recall-first after controls": [9.0, 8.5, 8.0, 9.0, 10.0, 8.5],
    "S4 mixed": [8.5, 9.0, 9.0, 9.5, 9.0, 9.0],
}

SCENARIOS = {
    "balanced": [0.22, 0.19, 0.17, 0.15, 0.13, 0.14],
    "causal_identifiability_heavy": [0.36, 0.20, 0.12, 0.10, 0.08, 0.14],
    "systems_relevance_heavy": [0.16, 0.13, 0.13, 0.30, 0.14, 0.14],
    "reproducibility_heavy": [0.18, 0.18, 0.15, 0.12, 0.10, 0.27],
    "novelty_heavy": [0.15, 0.14, 0.13, 0.13, 0.31, 0.14],
}


def simulate_scenario(
    *,
    samples: int,
    seed: int,
    score_sigma: float,
    concentration: float,
    base_weights: list[float],
) -> dict:
    rng = np.random.default_rng(seed)
    names = list(SEQUENCES)
    scores = np.array([BASE_SCORES[name] for name in names], dtype=float)
    weights0 = np.array(base_weights, dtype=float)
    weights0 = weights0 / weights0.sum()

    weights = rng.dirichlet(weights0 * concentration, size=samples)
    noise = rng.normal(0.0, score_sigma, size=(samples, len(names), len(CRITERIA)))
    sampled_scores = np.clip(scores[None, :, :] + noise, 0.0, 10.0)
    utility = (sampled_scores * weights[:, None, :]).sum(axis=2)

    winners = np.argmax(utility, axis=1)
    win_rates = {
        name: float(np.mean(winners == i))
        for i, name in enumerate(names)
    }
    mean_utility = {
        name: float(utility[:, i].mean())
        for i, name in enumerate(names)
    }

    return {
        "base_weights": dict(zip(CRITERIA, weights0.tolist())),
        "win_rates": win_rates,
        "mean_utility": mean_utility,
        "winner": max(win_rates, key=win_rates.get),
    }


def simulate(samples: int, seed: int, score_sigma: float, concentration: float) -> dict:
    scenario_results = {}
    for index, (name, weights) in enumerate(SCENARIOS.items()):
        scenario_results[name] = simulate_scenario(
            samples=samples,
            seed=seed + index,
            score_sigma=score_sigma,
            concentration=concentration,
            base_weights=weights,
        )

    return {
        "kind": "engineering_decision_support",
        "scientific_probability": False,
        "samples_per_scenario": samples,
        "seed": seed,
        "score_sigma": score_sigma,
        "dirichlet_concentration": concentration,
        "criteria": CRITERIA,
        "base_scores": BASE_SCORES,
        "sequences": SEQUENCES,
        "scenarios": scenario_results,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }


def markdown(result: dict) -> str:
    lines = [
        "# Stage 0 sequence decision-support result",
        "",
        "> Engineering decision support only. These are not scientific probabilities.",
        "",
        f"- samples per scenario: {result['samples_per_scenario']}",
        f"- base seed: {result['seed']}",
        f"- Python: {result['environment']['python']}",
        f"- NumPy: {result['environment']['numpy']}",
        "",
        "| Scenario | Winner | S1 | S2 | S3 | S4 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    short = {
        "S1 lineage-first controls": "S1",
        "S2 systems-first": "S2",
        "S3 recall-first after controls": "S3",
        "S4 mixed": "S4",
    }
    for scenario_name, scenario in result["scenarios"].items():
        rates = scenario["win_rates"]
        lines.append(
            "| "
            + scenario_name
            + " | "
            + scenario["winner"]
            + " | "
            + " | ".join(
                f"{rates[name]:.2%}" for name in SEQUENCES
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "Preferred execution order:",
            "",
            "~~~text",
            "VAL-001",
            "  -> VAL-002",
            "  -> BENCH-001",
            "  -> BENCH-002",
            "  -> EXP-001",
            "~~~",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--score-sigma", type=float, default=0.6)
    parser.add_argument("--concentration", type=float, default=70.0)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/decision-support"))
    parser.add_argument("--expected-winner", default=None)
    args = parser.parse_args()

    result = simulate(
        samples=args.samples,
        seed=args.seed,
        score_sigma=args.score_sigma,
        concentration=args.concentration,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "stage0-sequence.json"
    md_path = args.out_dir / "stage0-sequence.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    md_path.write_text(markdown(result) + "\n")

    print(markdown(result))

    if args.expected_winner is not None:
        failures = [
            name
            for name, scenario in result["scenarios"].items()
            if scenario["winner"] != args.expected_winner
        ]
        if failures:
            print(
                f"Expected winner {args.expected_winner!r}; "
                f"different winner in scenarios: {failures}"
            )
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
