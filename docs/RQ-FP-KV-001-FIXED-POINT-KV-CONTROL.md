# RQ-FP-KV-001 — Fixed-point-conditioned KV state collapse as a BENCH-002 control

> Status: LITERATURE / CONTROL INTAKE
> Primary subject remains Memory Attention
> Implementation target: NOT YET EARNED

## Why this intake exists

Memory Attention Lab already separates:

~~~text
capacity
active computation
accelerator residency
persistent inference state
reconstructability
~~~

Huang et al. (2026), *Towards Looped Models Done Right — Part II: Rethinking at
Fixed Points*, adds a nearby but distinct mechanism:

> recurrent state can become sufficiently settled that multiple loop-specific KV
> states can be replaced by the terminal KV representation.

Sources:

- https://www.alphaxiv.org/abs/2610.looped-models-fixed-points
- https://github.com/ifm-ai/xllm-loop
- official paper PDF:
  https://github.com/ifm-ai/xllm-loop/blob/main/papers/part2.pdf

This is not Memory Attention and must not be described as MA-Recall.

## Atomic mechanism separation

### Memory Attention

Core intervention:

~~~text
standard:
    V = X Wv

Memory Attention:
    V = K0 + token_memory
~~~

Question:

~~~text
how should value content be constructed,
and where should token-indexed memory reside?
~~~

### MA-Recall / BENCH-002

Working research question:

~~~text
can value-side state be reconstructed
instead of persistently cached?
~~~

This trades persistent storage against reconstruction / retrieval work.

### Looped-model terminal KV sharing

Different question:

~~~text
if repeated recurrent states converge,
can multiple loop-specific KV banks collapse
to the terminal bank without material prediction loss?
~~~

This is **state equivalence by convergence**, not value reconstruction.

### Offload

Another distinct mechanism:

~~~text
keep the state,
but move its residency tier
accelerator -> host -> storage
~~~

### Required vocabulary

Do not collapse these into "KV compression."

~~~text
RECONSTRUCT
    derive omitted state again from retained information

SHARE
    reuse one representation across consumers / recurrence depths

OFFLOAD
    retain state but change physical residency tier

DISCARD
    delete state because it is no longer semantically required

QUANTIZE / COMPRESS
    encode retained state with fewer physical bits
~~~

A valid benchmark must identify which operation produced the byte reduction.

## What the fixed-point paper contributes

### FP-1 — terminal sharing is conditional

The upstream paper reports that fixed-depth training can break under terminal KV
sharing because recurrent states keep drifting, while sampled / learned depth
training can shape states that settle enough for sharing.

Control implication:

> a smaller cache is not evidence of semantic equivalence.

BENCH-002 should always pair byte accounting with a semantic-equivalence metric.

### FP-2 — convergence is heterogeneous

The upstream analysis reports different convergence depths across tokens and
states; the paper says 95% of tokens are stable by loop 8 in the studied
configuration.

Control implication:

A per-sequence "done" flag can hide object-level differences. Persistent-state
accounting should be able to express:

~~~text
token i / state group j
    convergence depth
    retained banks
    endpoint error
~~~

### FP-3 — full trajectory can become redundant

For the reported R=5 setup, terminal sharing retains four KV banks instead of
twelve.

This supplies an important control category for this lab:

~~~text
persistent-state reduction
without host offload
without value reconstruction
without quantization
~~~

The causal mechanism is recurrence convergence.

### FP-4 — the endpoint can support compute shortcuts

The source also reports distilled prefill and fixed-point reuse in RL.

For this lab, the relevant lesson is narrower:

> reconstruction cost and storage cost are coupled to the representation
> contract; a representation that is endpoint-sufficient can change both.

Do not import the upstream speedups as Memory Attention results.

## Proposed BENCH-002 control taxonomy

BENCH-002 should distinguish at least the following state-management families
when interpreting nearby work:

| Family | What is retained? | Extra work | Semantic precondition |
|---|---|---|---|
| Full KV cache | all declared K/V state | minimal reconstruction | none beyond baseline |
| Host/storage offload | same logical state elsewhere | transfer / prefetch | state still required |
| MA-Recall-like reconstruction | compact reconstruction inputs | rebuild V or related state | reconstruction equivalence |
| Terminal KV sharing | terminal recurrent K/V only | recurrent consumer reuses endpoint | convergence / endpoint equivalence |
| Quantized/compressed cache | encoded K/V | decode / lower precision math | error tolerance |

This table is a research taxonomy, not a claim that all mechanisms are
interchangeable or compatible.

## New measurements for BENCH-002

### 1. Persistent-state bank count

~~~text
B_persist =
    number of logically distinct state banks
    required after the current step
~~~

This is different from physical bytes.

### 2. Persistent bytes

~~~text
M_persist = sum encoded_bytes(bank_i)
~~~

### 3. Reconstruction / reuse work

~~~text
C_restore =
    FLOPs + lookup + transfer + synchronization
    required because the full baseline state was not retained
~~~

### 4. Endpoint-equivalence gap

For task-local output Q:

~~~text
G_endpoint =
    distance(
        Q(full declared cache),
        Q(reduced endpoint representation)
    )
~~~

The distance may be loss, accuracy delta, distribution divergence, or another
frozen metric.

### 5. Convergence residual

For recurrent state x:

~~~text
R_t = ||F(x_t) - x_t||
~~~

This metric is relevant only to recurrent / looped controls. It should not be
forced onto Memory Attention configurations that do not expose the same
dynamical object.

### 6. Residency split

Continue to report separately:

~~~text
accelerator bytes
host RAM bytes
storage bytes
transfer bytes
~~~

A smaller accelerator footprint can still mean a larger total persistent state.

## Proposed control: FP-KV-CONTROL-001

Purpose:

> demonstrate that BENCH-002 can distinguish reconstruction-based state
> reduction from convergence-based state sharing.

Minimum arms:

| Arm | Mechanism | Expected interpretation |
|---|---|---|
| A | full cache | persistence baseline |
| B | offload full logical state | residency-only intervention |
| C | reconstruct omitted value-side state | reconstruction intervention |
| D | terminal-share a known convergent looped control | convergence-sharing intervention |
| E | terminal-share a drifting looped negative control | required failure control |

The looped control can be synthetic or the smallest reproducible upstream
configuration. It does not need to become a permanent dependency.

## Acceptance conditions

The control is useful only if the harness can report, separately:

~~~text
logical persistent banks
physical resident bytes by tier
restore / reuse work
semantic-equivalence gap
convergence residual when applicable
~~~

and if the negative control can show:

~~~text
same "terminal sharing" operation
+ non-converged state
-> measurable semantic failure
~~~

If the harness cannot distinguish these mechanisms, BENCH-002 is not yet ready
to interpret cache-state reduction claims.

## Relationship to current MA claims

This intake does **not** modify the existing meaning of:

~~~text
MA-006 — value state may be reconstructable instead of persistently cached
~~~

Instead it adds a nearby control:

~~~text
FP-KV — recurrent state may become shareable because multiple loop-specific
        representations converge toward an endpoint
~~~

These are mechanistically different hypotheses.

## Why this matters for Finite RAM Lab

Finite RAM Lab asks whether application-local lifecycle information can improve
memory decisions.

This lab is narrower: it asks **which logical state exists and why** before OS
policy is considered.

The order should remain:

~~~text
representation semantics
    -> logical persistent-state requirement
    -> physical residency choice
    -> OS / finite-RAM behavior
~~~

That prevents a representation-level byte reduction from being misreported as
an operating-system memory-management improvement.

## Evidence boundary

The upstream alphaXiv summary reports:

- models from 100M to 1.6B;
- terminal sharing with four rather than twelve KV banks in the described R=5
  setup;
- learned-depth / orthogonal-injection improvements in the tested looped
  models;
- one training seed per configuration;
- larger scales, other configurations, and real-world feasibility as untested.

Therefore this repository may cite the work as a control / prior-art mechanism,
but not generalize it into a universal KV-cache law.

## Claim ceiling

This intake can establish:

~~~text
FIXED_POINT_KV_SHARING_REGISTERED_AS_DISTINCT_CONTROL
BENCH002_STATE_MANAGEMENT_TAXONOMY_EXTENDED
~~~

It cannot establish:

~~~text
MEMORY_ATTENTION_GAINS_FROM_FIXED_POINTS
MA_RECALL_EQUALS_TERMINAL_KV_SHARING
TERMINAL_KV_SHARING_IS_UNIVERSALLY_SAFE
~~~
