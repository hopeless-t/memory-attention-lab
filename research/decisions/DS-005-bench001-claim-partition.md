# DS-005 — BENCH-001 claim partition

> **Status:** ACCEPTED DECISION SUPPORT  
> **Scientific authority:** NONE  
> **Claim-map authority:** BENCH-001 claim structure only  
> **Decision date:** 2026-09-25

## Problem

MA-004 currently combines two separable questions:

~~~text
A. can CPU placement reduce accelerator-resident parameter bytes?

B. can transfer / prefetch overhead be hidden or tolerated?
~~~

BENCH-001A can answer A without a GPU.

BENCH-001B is required for B.

If MA-004 remains compound, a valid BENCH-001A result cannot move the claim to
CONSISTENT without implying latency evidence that does not exist.

## Existing adjacent claim

MA-005 already records the benchmark-specific latency question:

> Can MA-Offload latency approach resident execution in the reported profile?

Creating both MA-004B and MA-005 would therefore duplicate the performance
claim.

## Options

~~~text
A keep MA-004 compound

B narrow MA-004 to residency / placement
  keep MA-005 as the latency claim

C split MA-004A / MA-004B
  and keep MA-005

D defer claim restructuring until GPU evidence exists
~~~

## Council conclusion

Option B has the cleanest evidence ownership.

~~~text
MA-004
    CPU offload can reduce accelerator parameter residency
    -> BENCH-001A

MA-005
    MA-Offload latency can approach resident execution
    -> BENCH-001B
~~~

The prefetch mechanism remains relevant context for MA-004, but its runtime
benefit is not part of MA-004's assessment.

## Monte Carlo stress test

The committed calculation perturbs six engineering criteria across five
scenarios:

~~~text
evidence independence
interpretability
minimal duplication
minimal churn
future maintainability
claim/source fidelity
~~~

Option B is required to win the balanced, evidence-independence,
interpretability, and maintainability scenarios.

The frozen 300,000-sample-per-scenario run produced:

| Scenario | A compound | B narrow MA-004 | C split MA-004 | D defer |
| --- | ---: | ---: | ---: | ---: |
| balanced | 0.00% | 100.00% | 0.00% | 0.00% |
| evidence independence heavy | 0.00% | 100.00% | 0.00% | 0.00% |
| interpretability heavy | 0.00% | 100.00% | 0.00% | 0.00% |
| maintainability heavy | 0.00% | 100.00% | 0.00% | 0.00% |
| minimal churn heavy | 1.03% | 97.53% | 0.00% | 1.44% |

The minimal-churn scenario was intentionally included as an adversarial stress
case. It still selected option B in the frozen run.

The implementation does not force a counterexample merely to make the decision
look less robust.

This is decision support, not scientific probability.

## Decision

Adopt option B before BENCH-001A evidence is published.

This avoids a future partial-status ambiguity.

## Evidence boundary

Changing claim structure does not change claim truth.

MA-004 remains NOT_TESTED until BENCH-001A canonical evidence exists.

MA-005 remains NOT_TESTED until valid CUDA BENCH-001B evidence exists.
