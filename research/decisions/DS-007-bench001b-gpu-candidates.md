# DS-007 — BENCH-001B GPU candidate priority

> **Status:** ACCEPTED DECISION SUPPORT  
> **Scientific authority:** NONE  
> **Performance authority:** NONE  
> **Purchase authority:** NONE  
> **Decision date:** 2026-09-25

## Question

If BENCH-001B B1 is later approved for real execution, which GPU class should be
probed first?

This decision does **not** start, rent, purchase, or authorize any GPU resource.

## Frozen protocol requirements

BENCH-001B B1 requires:

~~~text
CUDA
BF16
compute capability >= 8.0
flash-attn >= 2.1
source-default profile
isolated / approved execution environment
~~~

The official FlashAttention project states that FlashAttention-2 supports
Ampere, Ada, or Hopper CUDA GPUs and that BF16 requires those architectures.

## Free / opportunistic options

### Kaggle

Kaggle announced that P100 was retired on 2026-09-15 and T4x2 remains the
general notebook accelerator.

T4 is Turing and therefore does not satisfy this repository's frozen
FlashAttention-2/BF16 lane.

Kaggle is rejected for canonical B1.

### Google Colab

Colab provides free GPU access, but Google's own FAQ states that GPU types,
usage limits, and resource availability vary over time.

A Colab session may be useful for opportunistic probe experiments if it happens
to receive a qualifying GPU.

It is not the default canonical B1 execution path because the hardware class is
not reproducibly selectable in the free tier.

## Current paid candidate observations

Current public provider information was checked on 2026-09-25.

### Runpod Secure Cloud

Runpod published Secure Cloud rates verified in August 2026:

~~~text
RTX A5000 24 GB:
    $0.27 / hour

A40 48 GB:
    $0.49 / hour

A100 80 GB:
    $1.59 / hour
~~~

Runpod describes Secure Cloud as single-tenant / data-center infrastructure,
while Community Cloud uses vetted third-party capacity.

For research evidence, Secure Cloud is preferred over Community Cloud when the
price difference is acceptable because it reduces execution-environment
ambiguity.

### Lambda

Lambda currently lists:

~~~text
A100 40 GB:
    $1.99 / GPU-hour
~~~

Lambda remains a valid fallback provider but is not the first candidate under
the current cost/reproducibility trade-off.

## Hardware notes

NVIDIA documents the A40 as:

~~~text
NVIDIA Ampere architecture
48 GB GDDR6 ECC
~~~

NVIDIA documents the RTX A5000 as an Ampere-generation workstation GPU with
24 GB GDDR6 ECC.

FlashAttention-2 explicitly supports Ampere-class GPUs and BF16 on Ampere.

## VRAM uncertainty

BENCH-001A provides deterministic lower-level resource accounting, but it does
not predict full CUDA peak allocation.

The source-default accounting already separates:

~~~text
model parameters
KV cache
MA staging buffers
transfer payload
~~~

but activation storage, CUDA allocator behavior, FlashAttention workspace, and
implementation-specific temporary allocations require real measurement.

Therefore:

~~~text
24 GB
    potentially viable
    lower-cost
    greater OOM / allocator-margin uncertainty

48 GB
    substantially safer first probe
    still inexpensive

80 GB
    strongest headroom
    higher cost
    better fallback for OOM-risk or source-similarity priority
~~~

No claim is made that 24 GB is sufficient until a real probe succeeds.

## Candidate set

~~~text
A
    free / opportunistic notebook

B
    Runpod Secure RTX A5000 24 GB

C
    Runpod Secure A40 48 GB

D
    Runpod Secure A100 80 GB

E
    Lambda A100 40 GB
~~~

## Council decomposition

The candidate decision separates:

~~~text
protocol qualification
VRAM headroom
execution isolation
reproducibility
cost efficiency
setup friction
source proximity
availability
~~~

The most important trade-off is not simply:

~~~text
cheapest GPU
vs
fastest GPU
~~~

It is:

~~~text
minimum-cost resource
that is likely to complete the frozen benchmark
without weakening evidence quality
~~~

## Monte Carlo stress test

The committed decision-support calculation runs five weighting regimes with
300,000 samples each:

~~~text
balanced
OOM-risk-heavy
budget-heavy
reproducibility-heavy
source-similarity-heavy
~~~

Frozen design-run result:

| Scenario | Preferred candidate | Approx. win rate |
| --- | --- | ---: |
| balanced | Runpod Secure A40 48GB | 67.82% |
| OOM-risk-heavy | Runpod Secure A100 80GB | 52.57% |
| budget-heavy | Runpod Secure A40 48GB | 67.76% |
| reproducibility-heavy | Runpod Secure A40 48GB | 53.88% |
| source-similarity-heavy | Runpod Secure A100 80GB | 54.21% |

These are engineering decision-support frequencies, not probabilities of
success.

## Decision

Default **first probe candidate**:

~~~text
Runpod Secure A40 48 GB
~~~

Reason:

~~~text
qualifies architecturally
48 GB gives materially more headroom than 24 GB
current Secure Cloud hourly price remains low
single-tenant Secure Cloud is preferable for evidence provenance
~~~

Fallback:

~~~text
If A40 probe fails due memory/headroom:
    A100 80 GB

If a Human explicitly prioritizes minimum spend and accepts higher OOM risk:
    A5000 24 GB may be probed first
~~~

## Human approval gate

This decision is **not authorization to spend money**.

Before any paid B1 execution:

~~~text
Human chooses the concrete provider / GPU / price
Human explicitly approves the spend
capture-kit probe runs first without benchmark timing
Human reviews probe.json
only then may --execute be used
~~~

## Revisit conditions

Reopen DS-007 if:

~~~text
provider prices materially change
A40 availability disappears
a free guaranteed Ampere/Ada/Hopper resource becomes available
the source-default protocol changes
the capture probe shows unexpected VRAM/runtime constraints
real A40 execution fails
~~~

## Sources

Runpod current pricing:
https://www.runpod.io/pricing

Runpod Secure vs Community guidance:
https://www.runpod.io/blog/configuring-runpod

Lambda pricing:
https://lambda.ai/pricing

Google Colab resource FAQ:
https://research.google.com/colaboratory/faq.html

Kaggle P100 retirement / T4x2:
https://www.kaggle.com/discussions/product-announcements/735239

FlashAttention official repository:
https://github.com/Dao-AILab/flash-attention

NVIDIA A40:
https://www.nvidia.com/ja-jp/data-center/a40/

NVIDIA RTX A5000:
https://www.nvidia.com/ja-jp/products/workstations/rtx-a5000/
