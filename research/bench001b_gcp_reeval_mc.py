#!/usr/bin/env python3
"""Decision-support Monte Carlo for BENCH-001B Google Cloud reevaluation.

Canonical candidates are hard-gated for the frozen FlashAttention-2 lane before
Monte Carlo scoring. Google Compute Engine G4 / RTX PRO 6000 Blackwell is
recorded separately as exploratory because current official FlashAttention-2
support is Ampere/Ada/Hopper, not Blackwell SM120.

Engineering decision support only. Not scientific probability and not spend
authorization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


OPTIONS = [
    "A runpod-secure-a40-48gb",
    "B runpod-secure-a100-80gb",
    "C gcp-a2-a100-40gb",
    "D gcp-a2-ultra-a100-80gb",
]

CRITERIA = [
    "source_fidelity",
    "vram_headroom",
    "isolation_provenance",
    "region_proximity",
    "cost_efficiency",
    "setup_reproducibility",
    "availability",
    "orchestration_control",
]

BASE_SCORES = np.array(
    [
        [10.0, 8.0, 8.5, 7.0, 10.0, 8.0, 8.0, 7.5],
        [10.0, 10.0, 8.5, 7.0, 7.0, 8.0, 7.0, 7.5],
        [10.0, 7.5, 10.0, 10.0, 4.0, 9.5, 8.0, 10.0],
        [10.0, 10.0, 10.0, 9.0, 3.0, 9.5, 6.5, 10.0],
    ],
    dtype=float,
)

SCENARIOS = {
    "balanced": [0.18, 0.16, 0.14, 0.08, 0.16, 0.12, 0.08, 0.08],
    "budget_heavy": [0.14, 0.12, 0.10, 0.06, 0.34, 0.10, 0.07, 0.07],
    "provenance_heavy": [0.16, 0.12, 0.28, 0.10, 0.08, 0.14, 0.05, 0.07],
    "japan_proximity_heavy": [0.14, 0.12, 0.12, 0.28, 0.10, 0.10, 0.07, 0.07],
    "headroom_heavy": [0.14, 0.32, 0.12, 0.06, 0.10, 0.10, 0.08, 0.08],
    "cloud_credit_heavy": [0.16, 0.15, 0.18, 0.12, 0.05, 0.14, 0.07, 0.13],
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
        "canonical_hard_gate": {
            "gcp_g4_rtx_pro_6000_blackwell": "EXCLUDED_FROM_FA2_CANONICAL_LANE",
            "reason": (
                "official FlashAttention-2 CUDA support is currently "
                "Ampere/Ada/Hopper; RTX PRO 6000 Blackwell is SM120"
            ),
        },
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
        "# BENCH-001B Google Cloud reevaluation",
        "",
        "> Engineering decision support only. Not a purchase authorization.",
        "",
        "Hard gate:",
        "",
        "~~~text",
        "GCP G4 / RTX PRO 6000 Blackwell:",
        "    exploratory only for the current frozen FA2 lane",
        "~~~",
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
            "Interpretation:",
            "",
            "~~~text",
            "Cost-first default:",
            "    Runpod Secure A40 48 GB",
            "",
            "Google / Japan-proximity canonical option:",
            "    Compute Engine A2 / A100 40 GB",
            "",
            "Google provenance/headroom option:",
            "    Compute Engine A2 Ultra / A100 80 GB",
            "",
            "Blackwell exploratory lane:",
            "    Compute Engine G4 / RTX PRO 6000 96 GB",
            "    requires a separate protocol / attention-kernel decision",
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
        default=Path("runs/decision-support/bench001b-gcp-reeval"),
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
        "balanced": OPTIONS[0],
        "budget_heavy": OPTIONS[0],
        "provenance_heavy": OPTIONS[3],
        "japan_proximity_heavy": OPTIONS[2],
        "headroom_heavy": OPTIONS[3],
        "cloud_credit_heavy": OPTIONS[3],
    }
    for scenario, winner in expected.items():
        if result["scenarios"][scenario]["winner"] != winner:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
