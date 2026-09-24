#!/usr/bin/env python3
"""Decision-support Monte Carlo for VAL-002 control-family design.

Engineering decision support only. Scores and weights are explicit priors,
not scientific probabilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


OPTIONS = [
    "A historical-only",
    "B factorial-only",
    "C dual-lane",
    "D broad-zoo",
]

CRITERIA = [
    "causal_identifiability",
    "historical_fidelity",
    "scope_discipline",
    "implementation_cost",
    "future_training_reuse",
    "interpretability",
    "confound_control",
]

BASE_SCORES = np.array(
    [
        [5.5, 10.0, 9.0, 9.0, 6.0, 7.0, 5.0],
        [9.5, 4.5, 10.0, 9.5, 9.0, 9.5, 9.0],
        [10.0, 10.0, 8.0, 7.5, 10.0, 10.0, 10.0],
        [8.5, 9.0, 4.0, 4.0, 9.0, 6.0, 8.5],
    ],
    dtype=float,
)

SCENARIOS = {
    "balanced": [0.22, 0.13, 0.13, 0.10, 0.13, 0.14, 0.15],
    "causal_heavy": [0.34, 0.07, 0.10, 0.07, 0.10, 0.14, 0.18],
    "lineage_heavy": [0.13, 0.33, 0.13, 0.10, 0.09, 0.10, 0.12],
    "scope_cost_heavy": [0.12, 0.07, 0.29, 0.30, 0.07, 0.08, 0.07],
    "future_training_heavy": [0.18, 0.08, 0.08, 0.06, 0.29, 0.14, 0.17],
}


def simulate_one(
    *,
    samples: int,
    seed: int,
    base_weights: list[float],
    score_sigma: float,
    concentration: float,
) -> dict:
    rng = np.random.default_rng(seed)
    weights0 = np.asarray(base_weights, dtype=float)
    weights0 = weights0 / weights0.sum()

    sampled_weights = rng.dirichlet(weights0 * concentration, size=samples)
    score_noise = rng.normal(
        0.0,
        score_sigma,
        size=(samples, len(OPTIONS), len(CRITERIA)),
    )
    sampled_scores = np.clip(BASE_SCORES[None, :, :] + score_noise, 0.0, 10.0)
    utility = (sampled_scores * sampled_weights[:, None, :]).sum(axis=2)

    winners = np.argmax(utility, axis=1)
    win_rates = {
        name: float(np.mean(winners == index))
        for index, name in enumerate(OPTIONS)
    }

    return {
        "base_weights": dict(zip(CRITERIA, weights0.tolist())),
        "win_rates": win_rates,
        "mean_utility": {
            name: float(utility[:, index].mean())
            for index, name in enumerate(OPTIONS)
        },
        "winner": max(win_rates, key=win_rates.get),
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
            base_weights=weights,
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
        "# VAL-002 control-family decision support",
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
            "Default decision:",
            "",
            "~~~text",
            "C dual-lane",
            "",
            "causal lane:",
            "    2 x 2 memory/source factorial",
            "",
            "historical lane:",
            "    2024-12-04 learned-lambda Value Embedding reproduction",
            "~~~",
            "",
            "Important counterexample:",
            "",
            "Under the scope/cost-heavy scenario, factorial-only is expected to win.",
            "Therefore the historical lane is kept separate and should run only when",
            "lineage/source-fidelity evidence is required.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--score-sigma", type=float, default=0.75)
    parser.add_argument("--concentration", type=float, default=70.0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/decision-support/val002-controls"),
    )
    args = parser.parse_args()

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

    expected = {
        "balanced": "C dual-lane",
        "causal_heavy": "C dual-lane",
        "lineage_heavy": "C dual-lane",
        "scope_cost_heavy": "B factorial-only",
        "future_training_heavy": "C dual-lane",
    }
    failures = [
        name
        for name, winner in expected.items()
        if result["scenarios"][name]["winner"] != winner
    ]
    if failures:
        print(f"Unexpected winner in scenarios: {failures}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
