# DS-004 — BENCH-001 evidence lanes

> **Status:** ACCEPTED DECISION SUPPORT  
> **Scientific authority:** NONE  
> **Implementation authority:** BENCH-001 design only  
> **Decision date:** 2026-09-25

## Question

How should BENCH-001 test Memory Attention residency and offload behavior when
the repository has abundant standard GitHub Actions CPU capacity but no standard
hosted CUDA runner?

## Source observations

The current public Memory Attention benchmark is pinned to:

~~~text
repository:
Joluck/memory-attention

main commit:
8176f1feaff6724af670e70d75cd9767f8e38223

profile/bmk.py blob:
9ce6143fe7df1422b2f617996395d238edd2cc1e

profile/ma_profile.py blob:
186c386e4d631197e40251c1fc14ad0e46c80318

profile/sweep_bmk.py blob:
932cf4dcac6fb51135cce441810061ab245a1d51
~~~

The upstream benchmark requires CUDA, BF16 support, and FlashAttention >= 2.1.

Its source-fidelity timing scope includes the model forward and active offload
work, while token IDs are already present on CPU and GPU before timing.
Prefix construction, memory normalization folding, and token-ID device-to-host
transfer are excluded.

The upstream benchmark also separates model parameters from staging buffers.

## Claim decomposition

BENCH-001 contains at least two distinct scientific questions.

### A — residency

~~~text
Where do model parameters live?
How many parameter bytes remain accelerator-resident?
How much temporary staging storage is required?
~~~

These quantities are deterministic functions of the frozen configuration.

They do not require a physical GPU to calculate.

### B — performance

~~~text
How long does CPU gather take?
How long does H2D take?
How much transfer is hidden by concurrent compute?
What is the resulting model latency?
~~~

These are hardware and workload measurements.

They cannot be inferred from a CPU-only hosted runner.

## Candidate designs

### A — upstream GPU reproduction only

Run the upstream-style CUDA benchmark and do nothing until a GPU is available.

Strengths:

~~~text
high source fidelity
real hardware timing
~~~

Weaknesses:

~~~text
blocks all current work
does not exploit public Actions for deterministic validation
mixes residency and timing progress into one gate
~~~

### B — accounting only

Use Actions to validate parameter placement, transfer-buffer formulas, and
working-set accounting, but never create a GPU measurement lane.

Strengths:

~~~text
fully executable now
cheap
highly reproducible
~~~

Weakness:

~~~text
cannot answer the latency claim
~~~

### C — dual lane

Keep two separate evidence lanes.

~~~text
Lane A:
    deterministic residency / staging accounting
    standard public Actions
    hardware-independent

Lane B:
    source-faithful CUDA measurement
    identified GPU hardware
    real gather / H2D / overlap / model timing
~~~

This keeps progress unblocked without pretending that accounting is timing.

### D — simulation substitute

Use assumed PCIe/host/GPU bandwidth numbers to predict offload latency and treat
the prediction as if it replaced the CUDA benchmark.

Rejected as scientific evidence.

A bandwidth model may still be used as **decision support** to choose sweep
points or identify likely crossover regions.

## Monte Carlo stress test

The committed calculation uses 300,000 samples in each of five weighting
regimes, for 1.5 million simulated engineering decision states.

Frozen design-run result:

| Scenario | A GPU-only | B accounting | C dual-lane | D simulation substitute |
| --- | ---: | ---: | ---: | ---: |
| balanced | 0.00% | 0.15% | 99.85% | 0.00% |
| Actions-heavy | 0.00% | 15.73% | 84.27% | 0.00% |
| hardware-fidelity | <0.01% | <0.01% | 100.00% | 0.00% |
| scope/cost-heavy | 0.00% | 83.28% | 16.72% | <0.01% |
| source-reproduction | <0.01% | <0.01% | 100.00% | 0.00% |

These are engineering decision-support frequencies, not scientific
probabilities.

The scope/cost counterexample is intentionally retained: if the goal were only
to obtain the smallest immediately executable artifact, accounting-only would
be reasonable.

The current roadmap, however, explicitly includes the offload performance
question, so the dual-lane design remains appropriate.

## Deterministic accounting atoms

Let:

~~~text
H    = hidden size
Nh   = query heads
Nkv  = KV heads
Dh   = H / Nh
K    = Nkv * Dh
L    = layers
V    = vocabulary size
B    = batch size
S    = current input length
G    = offload group size
P    = prefetch depth
b    = bytes per memory-table element
~~~

The source benchmark changes the value-side parameters as follows.

Standard:

~~~text
one Wv projection per layer
Wv parameters per layer =
    H*K + optional bias K
~~~

Memory Attention:

~~~text
no Wv projection
+
token memory table =
    V*L*K
~~~

For CPU-offloaded MA:

~~~text
GPU parameter delta vs Standard =
    -L * Wv_parameters_per_layer

CPU parameter delta vs Standard =
    +V*L*K
~~~

Temporary staging buffers are not parameters and must be reported separately.

Pipeline allocation:

~~~text
slots = min(P, ceil(L/G))

GPU staging bytes =
    slots * B*S*G*K*b

pinned CPU staging bytes =
    slots * B*S*G*K*b
~~~

Bulk allocation:

~~~text
GPU staging bytes =
    B*S*L*K*b

pinned CPU staging bytes =
    B*S*L*K*b
~~~

## Persistent KV state boundary

The current upstream Standard and MA benchmark both maintain K and V cache
state.

Therefore BENCH-001 must report KV-cache storage separately and must not claim
an MA-Recall reduction.

MA-Recall belongs to BENCH-002.

## GitHub Actions boundary

Standard public GitHub-hosted runners are used aggressively for:

~~~text
strict spec validation
formula/property tests
large randomized accounting sweeps
integer-overflow guards
cross-platform reproduction
decision-support sensitivity analysis
~~~

Host-only latency measurements on hosted VMs may be diagnostic, but are not
canonical evidence for CUDA offload performance.

## Decision

Adopt **C — dual lane**.

~~~text
BENCH-001A
    deterministic residency / staging accounting

BENCH-001B
    source-faithful CUDA performance measurement
~~~

The A/B names are sub-lanes inside BENCH-001, not separate scientific claims.

A PASS in BENCH-001A does not make BENCH-001B pass.

## Revisit conditions

Reopen this decision if:

~~~text
the upstream benchmark changes its placement or timing scope materially;

a standard GitHub-hosted CUDA runner becomes available to this repository;

the research question is explicitly narrowed to residency accounting only;

the offload implementation changes from token-indexed deterministic lookup to a
different address/transfer model.
~~~

## Boundary

DS-004 selects evidence architecture only.

It does not establish that CPU offload is fast, latency-neutral, or beneficial.
