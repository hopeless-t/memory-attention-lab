# DS-003 — VAL-002 control-family design

> **Status:** ACCEPTED DECISION SUPPORT  
> **Scientific authority:** NONE  
> **Implementation authority:** VAL-002 design only  
> **Decision date:** 2026-09-25

## Question

How should VAL-002 separate historical fidelity from causal identification?

The original three-way sketch was:

~~~text
Standard:
    V = X Wv

Value-Embedding control:
    V = X Wv + E[token]

Memory Attention:
    V = X Wk + E[token]
~~~

Source recovery showed that this is not an exact reproduction of the historical
Value Embedding implementation cited by Memory Attention.

The inspected modded-nanogpt 2024-12-04 record uses a learned scalar mixture:

~~~text
V_hist = (1 - lambda) * (X Wv) + lambda * E_layer[token]
~~~

with a per-attention-layer learnable lambda initialized to 0.5 in that record.

Therefore one control cannot simultaneously serve as both:

~~~text
historical reproduction
and
minimal causal intervention
~~~

without introducing unnecessary ambiguity.

## Options

### A — historical-only

~~~text
Standard
Historical Value Embedding
Memory Attention
~~~

Strong source fidelity, weak causal decomposition.

### B — factorial-only

Use a 2 x 2 causal matrix:

~~~text
                     memory OFF           memory ON

Wv contextual source    X Wv               X Wv + M

Wk contextual source    X Wk               X Wk + M
~~~

where:

~~~text
M = Norm(E_layer[token])
~~~

The fourth cell is Memory Attention.

This cleanly tests the main effects of:

~~~text
token-indexed memory
contextual value source (Wv vs Wk)
their interaction
~~~

but does not reproduce the historical learned-lambda Value Embedding mechanism.

### C — dual-lane

Keep the factorial causal matrix from B and add a separate historical lane:

~~~text
Historical VE:
    V_hist = (1 - lambda) * X Wv + lambda * E_layer[token]
~~~

The historical lane answers a lineage/source-fidelity question.

The factorial lane answers a causal-identification question.

They are not pooled as if they were the same intervention.

### D — broad-zoo

Immediately add historical VE, factorial controls, MoVE-like mixing,
shared-Q controls, and additional memory variants.

This may become useful later, but it adds too many simultaneous degrees of
freedom for the first VAL-002 contract.

## Important boundary

The 2 x 2 factorial separates:

~~~text
memory absent vs present
Wv-derived contextual value vs Wk-derived contextual value
~~~

It does **not** fully separate:

~~~text
removing Wv
from
reusing K specifically
~~~

because exact Memory Attention couples those two changes.

A later shared-projection sentinel, such as a carefully designed shared-Q or
tied-projection control, may be added only if the primary experiment leaves that
question unresolved.

## Source recovery

Memory Attention cites:

~~~text
KoszarskyB.
Layerwise token value embeddings.
X post, 2024.
Announcement accompanying the modded-nanogpt implementation.
~~~

The implementation lineage can be anchored to:

~~~text
repository:
KellerJordan/modded-nanogpt

record:
records/track_1_short/2024-12-04_ValueEmbed/train_gpt2.py

inspected repository commit:
bc3a0c2d640d0d73dedaef87eae26148d2e32afb

file blob:
c3e21231926be6904e79720ffb19895c5493ed1c
~~~

The record code contains:

~~~text
vte = nn.Embedding(vocab_size, n_embd * 12)
vi = vte(token_ids).chunk(12, dim=-1)

v = c_v(x)
v = (1 - lambda) * v + lambda * vi
~~~

and attributes token value embeddings to @KoszarskyB, inspired by the
Value Residual implementation.

The modded-nanogpt project history also lists its 2024-12-04 record as
"Value Embeddings" with @KoszarskyB as contributor.

## Monte Carlo stress test

The committed decision-support script evaluates four design options under five
weighting regimes, 300,000 samples each:

~~~text
balanced
causal-heavy
lineage-heavy
scope/cost-heavy
future-training-heavy
~~~

Fresh-run result on GitHub Actions:

| Scenario | A historical-only | B factorial-only | C dual-lane | D broad-zoo | Winner |
| --- | ---: | ---: | ---: | ---: | --- |
| balanced | 0.00% | 7.16% | 92.84% | 0.00% | C |
| causal-heavy | 0.00% | 14.57% | 85.43% | 0.00% | C |
| lineage-heavy | 0.07% | 0.05% | 99.88% | 0.00% | C |
| scope/cost-heavy | 0.80% | 86.30% | 12.89% | 0.00% | B |
| future-training-heavy | 0.00% | 6.41% | 93.59% | 0.00% | C |

Run:

~~~text
GitHub Actions run:
36075177415

samples per scenario:
300,000

total simulated decision states:
1,500,000
~~~

The scope/cost counterexample is intentional.

It means the historical reproduction lane should remain optional and isolated,
not become a mandatory dependency of every later run.

## Decision

Adopt C:

~~~text
PRIMARY CAUSAL LANE

C00 Standard
    V = X Wv

C01 Add-memory
    V = X Wv + M

C10 Key-reuse-only
    V = X Wk

C11 Memory Attention
    V = X Wk + M

M = Norm(E_layer[token])
~~~

plus a separate historical reproduction lane:

~~~text
HIST-VE

V = (1 - lambda) * X Wv + lambda * E_layer[token]
~~~

## Execution order inside VAL-002

~~~text
1. Freeze algebra and tensor contracts for the four causal cells.
2. Add deterministic known-answer fixtures.
3. Add pairwise difference / interaction checks.
4. Add historical VE lane with its own source-pinned contract.
5. Only then add a framework adapter if training/system integration requires it.
~~~

## Non-claims

DS-003 does not claim:

- that the historical Value Embedding method is equivalent to Memory Attention;
- that a 2 x 2 algebraic test predicts training quality;
- that Wk is a better value source than Wv;
- that token memory improves quality;
- that removing Wv is independently beneficial;
- that a historical speedrun result transfers to another model or hardware.

It only fixes the first control-family design.
