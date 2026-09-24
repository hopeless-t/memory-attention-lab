#!/usr/bin/env python3
"""Decision-support Monte Carlo for VAL-002 control-family selection.

Engineering decision support only. Scores and weights are explicit priors used
to stress-test the research design. They are not scientific probabilities.
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
    "source_fidelity",
    "causal_identifiability",
    "scope_discipline",
    "implementation_cost",
    "future_training_reuse",
    "parameter_comparability",
    "interpretability",
]

BASE_SCORES = np.array(
    [
        [10.0, 5.0, 10.0, 10.0, 6.5, 5.0, 7.0],
        [6.0, 10.0, 9.0, 9.0, 9.0, 10.0, 9.0],
        [10.0, 10.0, 8.0, 7.5, 10.0, 10.0, 10.0],
        [9.0, 8.0, 2.0, 2.0, 8.0, 6.0, 5.0],
    ],
    dtype=float,
)

SCENARIOS = {
    "balanced": [0.17, 0.21, 0.15, 0.11, 0.13, 0.12, 0.11],
    "source_heavy": [0.32, 0.15, 0.15, 0.10, 0.08, 0.10, 0.10],
    "causal_heavy": [0.12, 0.34, 0.13, 0.08, 0.12, 0.12, 0.09],
    "scope_cost_heavy": [0.12, 0.13, 0.28, 0.24, 0.07, 0.08, 0.08],
    "future_training_heavy": [0.12, 0.17, 0.10, 0.08, 0.29, 0.14, 0.10],
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
            "Default choice:",
            "",
            "~~~text",
            "C dual-lane",
            "~~~",
            "",
            "Historical lane:",
            "",
            "~~~text",
            "VE_lambda = (1-lambda) * Vproj + lambda * E[token]",
            "~~~",
            "",
            "Causal factorial lane:",
            "",
            "~~~text",
            "C00 = Vproj",
            "C01 = Vproj + M",
            "C10 = Kcontent",
            "C11 = Kcontent + M    # Memory Attention construction",
            "~~~",
            "",
            "The historical lane preserves source fidelity.",
            "The factorial lane exists for causal decomposition and must not be",
            "misrepresented as the historical Value Embeddings method.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--score-sigma", type=float, default=0.7)
    parser.add_argument("--concentration", type=float, default=80.0)
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

    # The dual-lane design should win the scenarios that reflect the current
    # research roadmap. A scope/cost-heavy scenario is intentionally allowed
    # to prefer the smaller factorial-only design.
    required_dual = [
        "balanced",
        "source_heavy",
        "causal_heavy",
        "future_training_heavy",
    ]
    if any(
        result["scenarios"][name]["winner"] != "C dual-lane"
        for name in required_dual
    ):
        return 2

    if result["scenarios"]["scope_cost_heavy"]["winner"] != "B factorial-only":
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
