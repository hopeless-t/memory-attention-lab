#!/usr/bin/env python3
"""Decision-support Monte Carlo for BENCH-001B GPU candidate priority.

Engineering decision support only. Scores and weights are explicit priors,
not scientific probabilities and not a purchase authorization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


OPTIONS = [
    "A free-notebook-opportunistic",
    "B runpod-secure-a5000-24gb",
    "C runpod-secure-a40-48gb",
    "D runpod-secure-a100-80gb",
    "E lambda-a100-40gb",
]

CRITERIA = [
    "protocol_qualification",
    "vram_headroom",
    "security_isolation",
    "reproducibility",
    "cost_efficiency",
    "setup_friction",
    "source_proximity",
    "availability",
]

BASE_SCORES = np.array(
    [
        [3.0, 3.0, 4.0, 2.0, 10.0, 7.0, 5.0, 4.0],
        [10.0, 6.0, 9.0, 8.0, 10.0, 8.0, 7.0, 8.0],
        [10.0, 9.0, 9.0, 8.0, 9.0, 8.0, 8.0, 8.0],
        [10.0, 10.0, 9.0, 8.0, 6.0, 8.0, 9.0, 7.0],
        [10.0, 8.0, 9.0, 8.0, 5.0, 8.0, 9.0, 6.0],
    ],
    dtype=float,
)

SCENARIOS = {
    "balanced": [0.18, 0.18, 0.13, 0.12, 0.13, 0.08, 0.10, 0.08],
    "oom_risk_heavy": [0.15, 0.35, 0.10, 0.10, 0.08, 0.05, 0.10, 0.07],
    "budget_heavy": [0.14, 0.12, 0.10, 0.08, 0.32, 0.08, 0.08, 0.08],
    "reproducibility_heavy": [0.16, 0.15, 0.15, 0.23, 0.08, 0.07, 0.09, 0.07],
    "source_similarity_heavy": [0.16, 0.17, 0.10, 0.10, 0.07, 0.06, 0.27, 0.07],
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
        option: float(np.mean(winners == index))
        for index, option in enumerate(OPTIONS)
    }
    return {
        "weights": dict(zip(CRITERIA, weights0.tolist())),
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
        "purchase_authority": False,
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
        "# BENCH-001B GPU candidate decision support",
        "",
        "> Engineering decision support only. Not a purchase authorization.",
        "",
        f"- samples per scenario: {result['samples_per_scenario']}",
        f"- base seed: {result['base_seed']}",
        "",
        "| Scenario | Winner | A | B | C | D | E |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
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
            "Default candidate:",
            "",
            "~~~text",
            "C runpod-secure-a40-48gb",
            "~~~",
            "",
            "Fallback logic:",
            "",
            "~~~text",
            "If 48 GB headroom is still insufficient:",
            "    D runpod-secure-a100-80gb",
            "",
            "If budget dominates and probe/headroom checks support it:",
            "    B runpod-secure-a5000-24gb",
            "~~~",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--score-sigma", type=float, default=0.8)
    parser.add_argument("--concentration", type=float, default=60.0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/decision-support/bench001b-gpu-candidates"),
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

    if result["scenarios"]["balanced"]["winner"] != OPTIONS[2]:
        return 2
    if result["scenarios"]["budget_heavy"]["winner"] != OPTIONS[2]:
        return 2
    if result["scenarios"]["reproducibility_heavy"]["winner"] != OPTIONS[2]:
        return 2
    if result["scenarios"]["oom_risk_heavy"]["winner"] != OPTIONS[3]:
        return 2
    if result["scenarios"]["source_similarity_heavy"]["winner"] != OPTIONS[3]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
