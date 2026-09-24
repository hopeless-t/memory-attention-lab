# DS-002 — VAL-001 oracle architecture

> **Status:** ACCEPTED DECISION SUPPORT  
> **Scientific authority:** NONE  
> **Implementation authority:** VAL-001 harness only  
> **Decision date:** 2026-09-25

## Question

What should serve as the first independent validation oracle for Memory Attention algebra?

## Options

~~~text
A. NumPy oracle only

B. PyTorch oracle only

C. Upstream FLA implementation as the oracle

D. Independent NumPy oracle first, with a PyTorch adapter added later
   only when an executable comparison requires it
~~~

## Council concerns

The council separated seven criteria:

~~~text
implementation independence
source fidelity
CI cost
consumer-hardware feasibility
future integration reuse
shared-bug resistance
scope discipline
~~~

The hardest conflict is:

~~~text
maximum immediate simplicity
        vs
independent oracle + realistic future integration
~~~

Using upstream FLA directly was rejected as the primary oracle because the code under study must not also be its own sole validator.

PyTorch-only was not selected because a framework-independent numerical oracle is cheaper to inspect and less likely to share implementation assumptions with later model code.

## Monte Carlo stress test

The committed decision-support calculation runs four weighting regimes with 300,000 samples each:

~~~text
balanced
independence-heavy
CI-simplicity-heavy
future-adapter-heavy
~~~

The frozen design calculation produced approximately:

| Scenario | Preferred option | Approx. win rate |
| --- | --- | ---: |
| balanced | D | 77.76% |
| independence-heavy | D | 73.73% |
| CI-simplicity-heavy | A | 66.26% |
| future-adapter-heavy | D | 96.76% |

These are engineering decision-support frequencies, not scientific probabilities.

## Interpretation

The counterexample matters.

If this repository intended to remain a tiny algebra-only project forever, NumPy-only would be a reasonable endpoint and wins the CI-simplicity stress case.

But the frozen roadmap already contains:

~~~text
VAL-002
BENCH-001
BENCH-002
EXP-001
~~~

which will eventually require framework and systems integration.

Therefore the decision is:

~~~text
NOW:
    independent NumPy oracle

LATER:
    PyTorch adapter checked against the NumPy oracle

NOT NOW:
    FLA / Triton / FlashAttention as validation authority
~~~

## Decision

Adopt option D.

The PyTorch adapter is **deferred**, not prebuilt.

It earns its way into the repository only when VAL-002 or a later experiment requires it.

## Revisit conditions

Reopen DS-002 if:

~~~text
the independent oracle cannot represent a required Memory Attention contract;

a later source revision materially changes the algebra;

the PyTorch adapter exposes a precision or layout behavior the NumPy contract
cannot faithfully model;

the project scope is explicitly reduced to algebra-only validation.
~~~

## Boundary

This decision says nothing about whether Memory Attention is better, faster, or more accurate.

It only selects how this repository should validate its first numerical contract.
