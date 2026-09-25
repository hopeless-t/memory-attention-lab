#!/usr/bin/env python3
"""Decision-support Monte Carlo for BENCH-001 evidence-lane selection.

This is engineering decision support, not a scientific probability model.
Scores and weights are explicit priors used to stress-test the research design.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


OPTIONS = [
    "A upstream-GPU-only",
    "B accounting-only",
    "C dual-lane",
    "D simulation-substitute",
]

CRITERIA = [
    "scientific_validity",
    "source_fidelity",
    "executable_now",
    "hardware_realism",
    "claim_separation",
    "reproducibility",
    "scope_cost",
]

# 0..10 engineering priors. These are not measured scientific quantities.
BASE_SCORES = np.array(
    [
        [7.0, 10.0, 2.0, 10.0, 6.0, 5.0, 8.0],
        [8.0, 7.0, 10.0, 3.0, 9.0, 10.0, 10.0],
        [10.0, 10.0, 8.5, 10.0, 10.0, 9.0, 7.0],
        [5.0, 5.0, 10.0, 4.0, 6.0, 10.0, 7.0],
    ],
    dtype=float,
)

SCENARIOS = {
    "balanced": [0.19, 0.15, 0.14, 0.14, 0.16, 0.13, 0.09],
    "actions_heavy": [0.14, 0.10, 0.26, 0.08, 0.14, 0.18, 0.10],
    "hardware_fidelity": [0.18, 0.13, 0.08, 0.28, 0.15, 0.10, 0.08],
    "scope_cost_heavy": [0.10, 0.07, 0.20, 0.06, 0.10, 0.12, 0.35],
    "source_reproduction": [0.15, 0.30, 0.08, 0.18, 0.11, 0.10, 0.08],
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
        "# BENCH-001 evidence-lane decision support",
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
            "Lane A:",
            "",
            "~~~text",
            "deterministic residency / staging accounting",
            "runs on standard public Actions",
            "hardware-independent evidence",
            "~~~",
            "",
            "Lane B:",
            "",
            "~~~text",
            "source-faithful CUDA latency measurement",
            "requires an identified CUDA system",
            "never substituted by CPU-only timing",
            "~~~",
            "",
            "A bandwidth model may be used only as decision support or experiment",
            "planning. It is not measured GPU performance.",
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
        default=Path("runs/decision-support/bench001-design"),
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

    dual_required = [
        "balanced",
        "actions_heavy",
        "hardware_fidelity",
        "source_reproduction",
    ]
    if any(
        result["scenarios"][name]["winner"] != "C dual-lane"
        for name in dual_required
    ):
        return 2

    if result["scenarios"]["scope_cost_heavy"]["winner"] != "B accounting-only":
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
