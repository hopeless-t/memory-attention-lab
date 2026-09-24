# DS-003 — VAL-002 control-family design

> **Status:** ACCEPTED DECISION SUPPORT  
> **Scientific authority:** NONE  
> **Implementation authority:** VAL-002 control family only  
> **Decision date:** 2026-09-25

## Question

How should VAL-002 separate source fidelity from causal identifiability when
comparing Standard Attention, historical Value Embeddings, and Memory Attention?

## New source finding

The previously unresolved Value Embeddings lineage is now sufficiently resolved
for an implementation-level control.

The Memory Attention paper cites KoszarskyB (2024), *Layerwise token value
embeddings*, as nearby prior work.

The modded-nanogpt project records **Value Embeddings** on 2024-12-04 and credits
@KoszarskyB. Its archived implementation uses:

~~~text
v = Vproj(x)
v = (1 - lambda) * v + lambda * token_value_embedding
~~~

with a learnable lambda initialized to 0.5 in that record.

The project README also describes extra embeddings mixed into attention values
as inspired by Zhou et al. (2024), Value Residual Learning.

Frozen implementation source for lineage work:

~~~text
repository:
KellerJordan/modded-nanogpt

repository ref:
bc3a0c2d640d0d73dedaef87eae26148d2e32afb

2024-12-04 ValueEmbed training script blob:
c3e21231926be6904e79720ffb19895c5493ed1c

README blob:
7df95e21a027b5f6db4f3b6c20b6b7e7805a56b6
~~~

This resolves the implementation lineage needed for VAL-002.

It does **not** assert that a standalone archival paper exists for the original
Value Embeddings record.

## Why the previous three-way shorthand was insufficient

An earlier local shorthand described the control as:

~~~text
V = Vproj + E[token]
~~~

That is useful as a synthetic additive control, but it is **not** the
source-faithful 2024-12-04 Value Embeddings formula.

Treating those as the same method would mix two different goals:

~~~text
historical/source fidelity
vs
causal decomposition
~~~

VAL-002 must preserve both without conflating them.

## Candidate designs

### A — Historical-only

~~~text
Standard:
    V = Vproj

Historical VE:
    V = (1-lambda) * Vproj + lambda * E[token]

Memory Attention:
    V = Kcontent + M
~~~

Strength: source fidelity.

Weakness: the changed scaling/mixing rule makes it difficult to isolate the
effect of memory addition from the effect of replacing Vproj with Kcontent.

### B — Factorial-only

~~~text
C00 = Vproj
C01 = Vproj + M
C10 = Kcontent
C11 = Kcontent + M
~~~

Strength: clean causal decomposition.

Weakness: C01 is a lab-created control, not the historical Value Embeddings
method.

### C — Dual-lane

Keep both:

~~~text
SOURCE-FIDELITY LANE

Historical VE(lambda):
    (1-lambda) * Vproj + lambda * E[token]

CAUSAL-FACTORIAL LANE

C00 = Vproj
C01 = Vproj + M
C10 = Kcontent
C11 = Kcontent + M
~~~

The two lanes have different authority.

Historical VE answers:

> Are we representing the nearby prior mechanism faithfully?

The factorial lane answers:

> Which algebraic ingredient changes when memory is added or Vproj is replaced
> by Kcontent under the same memory contribution?

### D — Broad zoo

Add MoVE, PLE, DeepEmbed, Engram, and other nearby mechanisms to VAL-002.

Rejected for the initial stage because it destroys scope discipline before the
first causal control is validated.

## Atomic decomposition

The design factors are:

~~~text
CONTEXTUAL SOURCE
    Vproj
    Kcontent

TOKEN MEMORY
    absent
    present

FUSION RULE
    additive
    learned convex-style blend form
    other gated/mixed rules

MEMORY TRANSFORM
    raw embedding
    normalized Memory Attention contribution

AUTHORITY
    historical reproduction
    local experimental control
~~~

The last axis is essential.

A local control is not historical prior art merely because it resembles it.

## Monte Carlo stress test

The committed calculation evaluates five weighting regimes with 300,000
samples each:

~~~text
balanced
source-heavy
causal-identifiability-heavy
scope/cost-heavy
future-training-heavy
~~~

Frozen design-run result:

| Scenario | A historical | B factorial | C dual-lane | D broad zoo |
| --- | ---: | ---: | ---: | ---: |
| balanced | 0.00% | 7.69% | 92.30% | 0.00% |
| source-heavy | 0.18% | 0.51% | 99.31% | 0.00% |
| causal-heavy | 0.00% | 14.36% | 85.64% | 0.00% |
| scope/cost-heavy | 6.61% | 52.56% | 40.83% | 0.00% |
| future-training-heavy | 0.00% | 6.92% | 93.08% | 0.00% |

These are engineering decision-support frequencies, not scientific
probabilities.

The counterexample is intentional: when scope and immediate implementation cost
dominate strongly, the smaller factorial-only design wins.

## Council conclusion

The dual-lane design is selected because the current repository already has a
broad literature/claim mandate and a later training experiment.

However, scope is constrained as follows:

~~~text
IMPLEMENT NOW
    source-faithful Value Embeddings fusion
    four-cell causal factorial construction
    exact algebraic identities
    linkage from C11 to the frozen VAL-001 Memory Attention construction

DEFER
    MoVE
    PLE
    DeepEmbed
    Engram
    training
    quality claims
~~~

## Factorial identities

For a fixed projected value P, contextual key K, and Memory Attention memory
contribution M:

~~~text
C00 = P
C01 = P + M
C10 = K
C11 = K + M
~~~

VAL-002 should verify:

~~~text
C01 - C00 = M
C11 - C10 = M

C10 - C00 = K - P
C11 - C01 = K - P

C11 - C10 - C01 + C00 = 0
~~~

The zero interaction here is an **algebraic property of this additive
construction at fixed inputs**.

It is not a claim that training dynamics have zero interaction.

## Decision

Adopt **C — dual-lane**.

Keep the historical and causal lanes visibly separate in code, spec, evidence,
and prose.

## Revisit conditions

Reopen DS-003 if:

~~~text
the source-faithful Value Embeddings lineage changes materially;

VAL-002 shows the proposed factorial controls fail to isolate the intended
construction factors;

a later training design requires a different matched scaling convention;

a new source demonstrates that the local additive control duplicates a named
historical method exactly.
~~~

## Boundary

DS-003 chooses the validation/control structure only.

It does not establish that Memory Attention, Value Embeddings, or any later
model is better.
