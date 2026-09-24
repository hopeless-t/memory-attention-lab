#!/usr/bin/env python3
"""Decision-support Monte Carlo for Stage 0 research sequencing.

This is engineering decision support, not a scientific probability model.
The scores and weights are explicit priors used to stress-test whether the
preferred sequence is stable under reasonable perturbations.
"""

from __future__ import annotations

import argparse
import json
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

BASE_WEIGHTS = np.array([0.22, 0.19, 0.17, 0.15, 0.13, 0.14], dtype=float)


def simulate(samples: int, seed: int, score_sigma: float, concentration: float) -> dict:
    rng = np.random.default_rng(seed)
    names = list(SEQUENCES)
    scores = np.array([BASE_SCORES[name] for name in names], dtype=float)

    weights = rng.dirichlet(BASE_WEIGHTS * concentration, size=samples)
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

    winner = max(win_rates, key=win_rates.get)

    return {
        "kind": "engineering_decision_support",
        "scientific_probability": False,
        "samples": samples,
        "seed": seed,
        "score_sigma": score_sigma,
        "dirichlet_concentration": concentration,
        "criteria": CRITERIA,
        "base_weights": dict(zip(CRITERIA, BASE_WEIGHTS.tolist())),
        "base_scores": BASE_SCORES,
        "sequences": SEQUENCES,
        "win_rates": win_rates,
        "mean_utility": mean_utility,
        "winner": winner,
    }


def markdown(result: dict) -> str:
    lines = [
        "# Stage 0 sequence decision-support result",
        "",
        "> Engineering decision support only. These are not scientific probabilities.",
        "",
        f"- samples: {result['samples']}",
        f"- seed: {result['seed']}",
        f"- winner: **{result['winner']}**",
        "",
        "| Sequence | Win rate | Mean utility |",
        "| --- | ---: | ---: |",
    ]
    for name in SEQUENCES:
        lines.append(
            f"| {name} | {result['win_rates'][name]:.4%} | "
            f"{result['mean_utility'][name]:.4f} |"
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

    if args.expected_winner is not None and result["winner"] != args.expected_winner:
        print(
            f"Expected winner {args.expected_winner!r}, "
            f"observed {result['winner']!r}"
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
