#!/usr/bin/env python3
"""Decision-support Monte Carlo for the VAL-001 oracle architecture.

Engineering decision support only. The scores and weights are explicit priors,
not scientific probabilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


OPTIONS = [
    "A NumPy oracle only",
    "B PyTorch oracle only",
    "C Upstream FLA as oracle",
    "D NumPy oracle + later PyTorch adapter",
]

CRITERIA = [
    "implementation_independence",
    "source_fidelity",
    "ci_cost",
    "consumer_hardware_feasibility",
    "future_integration_reuse",
    "shared_bug_resistance",
    "scope_discipline",
]

BASE_SCORES = np.array(
    [
        [10.0, 8.0, 10.0, 10.0, 6.5, 10.0, 9.0],
        [7.0, 9.0, 8.0, 8.0, 9.0, 7.0, 8.0],
        [2.0, 10.0, 3.0, 2.0, 8.0, 2.0, 5.0],
        [10.0, 9.5, 8.5, 8.5, 10.0, 10.0, 8.5],
    ],
    dtype=float,
)

SCENARIOS = {
    "balanced": [0.20, 0.15, 0.13, 0.12, 0.14, 0.15, 0.11],
    "independence_heavy": [0.30, 0.13, 0.09, 0.09, 0.10, 0.20, 0.09],
    "ci_simplicity_heavy": [0.14, 0.11, 0.24, 0.22, 0.08, 0.12, 0.09],
    "future_adapter_heavy": [0.16, 0.18, 0.08, 0.08, 0.25, 0.16, 0.09],
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
    base_weights = base_weights / base_weights.sum()

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
        "# VAL-001 oracle decision support",
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
            "D NumPy oracle + later PyTorch adapter",
            "~~~",
            "",
            "Interpretation:",
            "",
            "- The independent NumPy oracle protects against shared implementation bugs.",
            "- A later PyTorch adapter preserves a path to actual model/system integration.",
            "- Under an extreme CI-simplicity weighting, NumPy-only may be preferred.",
            "- Therefore the adapter is deferred until an experiment actually requires it.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--score-sigma", type=float, default=0.85)
    parser.add_argument("--concentration", type=float, default=70.0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/decision-support/val001-oracle"),
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

    if result["scenarios"]["balanced"]["winner"] != OPTIONS[3]:
        return 2
    if result["scenarios"]["independence_heavy"]["winner"] != OPTIONS[3]:
        return 2
    if result["scenarios"]["future_adapter_heavy"]["winner"] != OPTIONS[3]:
        return 2
    if result["scenarios"]["ci_simplicity_heavy"]["winner"] != OPTIONS[0]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
