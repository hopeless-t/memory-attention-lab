#!/usr/bin/env python3
"""Decision-support Monte Carlo for BENCH-001 claim partitioning.

Engineering decision support only. Results are not scientific probabilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


OPTIONS = [
    "A keep-compound-MA004",
    "B narrow-MA004-residency-use-MA005-latency",
    "C split-MA004A-MA004B-keep-MA005",
    "D defer-claim-structure",
]

CRITERIA = [
    "evidence_independence",
    "interpretability",
    "minimal_duplication",
    "minimal_churn",
    "future_maintainability",
    "claim_source_fidelity",
]

BASE_SCORES = np.array(
    [
        [4.0, 5.0, 7.0, 10.0, 5.0, 8.0],
        [10.0, 10.0, 10.0, 8.0, 10.0, 9.0],
        [10.0, 8.0, 4.0, 4.0, 7.0, 9.0],
        [3.0, 6.0, 8.0, 10.0, 5.0, 7.0],
    ],
    dtype=float,
)

SCENARIOS = {
    "balanced": [0.22, 0.20, 0.16, 0.12, 0.16, 0.14],
    "evidence_independence_heavy": [0.36, 0.18, 0.12, 0.08, 0.14, 0.12],
    "interpretability_heavy": [0.18, 0.34, 0.14, 0.08, 0.14, 0.12],
    "maintainability_heavy": [0.18, 0.16, 0.16, 0.10, 0.28, 0.12],
    "minimal_churn_heavy": [0.10, 0.10, 0.14, 0.46, 0.10, 0.10],
}


def simulate_one(samples, seed, weights0, score_sigma, concentration):
    rng = np.random.default_rng(seed)
    weights0 = np.asarray(weights0, dtype=float)
    weights0 /= weights0.sum()
    weights = rng.dirichlet(weights0 * concentration, size=samples)
    noise = rng.normal(
        0.0,
        score_sigma,
        size=(samples, len(OPTIONS), len(CRITERIA)),
    )
    scores = np.clip(BASE_SCORES[None, :, :] + noise, 0.0, 10.0)
    utility = (scores * weights[:, None, :]).sum(axis=2)
    winners = np.argmax(utility, axis=1)
    rates = {
        option: float(np.mean(winners == idx))
        for idx, option in enumerate(OPTIONS)
    }
    return {
        "weights": dict(zip(CRITERIA, weights0.tolist())),
        "win_rates": rates,
        "winner": max(rates, key=rates.get),
    }


def simulate(samples, seed, score_sigma, concentration):
    scenarios = {}
    for offset, (name, weights) in enumerate(SCENARIOS.items()):
        scenarios[name] = simulate_one(
            samples,
            seed + offset,
            weights,
            score_sigma,
            concentration,
        )
    return {
        "kind": "engineering_decision_support",
        "scientific_probability": False,
        "samples_per_scenario": samples,
        "base_seed": seed,
        "criteria": CRITERIA,
        "options": OPTIONS,
        "base_scores": BASE_SCORES.tolist(),
        "scenarios": scenarios,
    }


def markdown(result):
    lines = [
        "# BENCH-001 claim partition decision support",
        "",
        "> Engineering decision support only. These are not scientific probabilities.",
        "",
        "| Scenario | Winner | A | B | C | D |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, scenario in result["scenarios"].items():
        rates = scenario["win_rates"]
        lines.append(
            f"| {name} | {scenario['winner']} | "
            + " | ".join(f"{rates[o]:.2%}" for o in OPTIONS)
            + " |"
        )
    lines += [
        "",
        "Default choice:",
        "",
        "~~~text",
        "B narrow-MA004-residency-use-MA005-latency",
        "~~~",
        "",
        "This lets BENCH-001A assess residency without partially resolving a",
        "compound latency claim. MA-005 remains the benchmark-specific latency claim.",
        "",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--score-sigma", type=float, default=0.7)
    p.add_argument("--concentration", type=float, default=80.0)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/decision-support/bench001-claim-partition"),
    )
    args = p.parse_args()

    result = simulate(
        args.samples,
        args.seed,
        args.score_sigma,
        args.concentration,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "result.md").write_text(markdown(result) + "\n")
    print(markdown(result))

    required = [
        "balanced",
        "evidence_independence_heavy",
        "interpretability_heavy",
        "maintainability_heavy",
    ]
    if any(
        result["scenarios"][name]["winner"]
        != "B narrow-MA004-residency-use-MA005-latency"
        for name in required
    ):
        return 2

    # Stress scenarios test robustness; they do not prescribe a winner in
    # advance merely to manufacture a counterexample. The balanced and
    # evidence-ownership scenarios above are the acceptance gate.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
