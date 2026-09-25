# Claim Map

> **Status:** STAGING RESEARCH  
> **Scientific authority:** NONE  
> **Implementation authority:** NONE

This document turns source claims and local hypotheses into explicit testable conditions.

The rule is:

~~~text
SOURCE CLAIM
    |
OUR INTERPRETATION
    |
TESTABLE CONDITION
    |
EXPERIMENT
    |
OBSERVATION
    |
ASSESSMENT
~~~

A valid experiment can PASS while the associated claim remains INCONCLUSIVE.

## Assessment vocabulary

~~~text
NOT_TESTED
CONSISTENT
INCONSISTENT
INCONCLUSIVE
~~~

These labels describe local evidence relative to a stated claim.

They are not universal verdicts on a paper or architecture.

---

# Primary Memory Attention claims

## MA-001 — Value construction

**Source**

Kang, *Memory Attention*, arXiv:2609.28399.

**Source claim / mechanism**

Memory Attention constructs value content from contextual key-side state plus layer-specific token-indexed memory, removing a separate value projection.

~~~text
K0 = X Wk
M  = Memory[token_id]
V  = K0 + Norm(M)
~~~

**Local interpretation**

The first thing to validate is not quality or speed. It is that an independent implementation matches the intended algebra, including the distinction between pre-attention key state and any subsequently transformed attention key.

**Potential confounds**

~~~text
RoPE placement
Q/K normalization
head reshaping
GQA/MQA head dimensions
memory normalization
dtype and rounding
gate variants
~~~

**Test**

~~~text
VAL-001
~~~

**Current status**

~~~text
CONSISTENT
~~~

**Evidence**

VAL-001 canonical reference evidence demonstrates the frozen independent algebra,
including the pre-RoPE content-key boundary and known-answer value construction:

~~~text
evidence/VAL-001/reference/
source commit f941844c985a0857dc93b6cc321fa40427856f08
~~~

This assessment is limited to the frozen numerical contract. It is not a
quality, performance, or upstream-runtime-equivalence claim.

---

## MA-002 — Inference memory normalization can be pre-folded

**Source**

Kang, *Memory Attention*, arXiv:2609.28399.

**Source claim**

At inference, normalization of token-indexed memory can be folded into the stored memory representation, reducing online value construction to memory lookup plus addition.

**Local interpretation**

For fixed trained parameters and inference mode, a pre-normalized memory table should reproduce the online-normalized value construction within a declared numerical tolerance.

**Potential confounds**

~~~text
RMSNorm epsilon
parameter dtype
storage dtype
quantization
normalization implementation
rounding order
~~~

**Test**

~~~text
VAL-001
~~~

**Current status**

~~~text
CONSISTENT
~~~

**Evidence**

VAL-001 shows exact agreement between online RMS-normalization and the frozen
pre-folded memory-table path for the reference fixture:

~~~text
folded_equivalence_max_abs_error = 0.0
~~~

This does not test quantized, lower-precision, or deployment-specific folded
tables.

---

## MA-003 — Additional memory parameters improve reported quality

**Source**

Kang, *Memory Attention*, arXiv:2609.28399.

**Source claim**

Under matched training-token budgets and with additional memory parameters, reported experiments improve language-modeling perplexity and average downstream performance across studied configurations.

**Local interpretation**

The published comparison is evidence for the reported experimental setting, but it does not isolate whether the observed improvement is due to:

~~~text
the Memory Attention construction
additional parameter capacity
different parameter activation pattern
removal of Wv
reuse of K
optimization interactions
~~~

**Required controls**

VAL-002 now freezes a local causal family:

~~~text
C00:
    V = X Wv

C01:
    V = X Wv + M

C10:
    V = X Wk

C11:
    V = X Wk + M

M = Norm(E_layer[token])
~~~

and keeps the historical Value Embeddings mechanism in a separate
source-fidelity lane:

~~~text
V_hist =
    (1 - lambda) * X Wv
    +
    lambda * E_layer[token]
~~~

VAL-002 validates these control definitions algebraically. It does **not**
measure the model-quality claim in MA-003.

Training comparisons must also state exactly which budgets are matched.

**Tests**

~~~text
VAL-002
EXP-001
~~~

**Current status**

~~~text
NOT_TESTED
~~~

---

## MA-004 — CPU offload can reduce accelerator parameter residency

**Source**

Kang, *Memory Attention*, arXiv:2609.28399.

**Source claim**

Token-indexed memory can be stored outside GPU memory because the lookup address
is determined by token identity and layer position.

**Local interpretation**

MA-004 is deliberately narrowed to **parameter placement / accelerator
residency**.

It asks:

~~~text
Can the MA table reside in host memory while the dedicated Wv projection is
absent from accelerator-resident parameters?
~~~

This is an accounting / placement claim and belongs to:

~~~text
BENCH-001A
~~~

Runtime overlap, transfer hiding, and whether offload latency is acceptable are
**not** part of MA-004's assessment. Those belong to MA-005 / BENCH-001B.

A BENCH-001A PASS may therefore move MA-004 without implying any latency result.

**Relevant accounting variables**

~~~text
Wv parameter bytes removed from accelerator residency
MA table bytes placed in host residency
temporary pinned-host / GPU staging bytes
persistent KV-cache bytes reported separately
~~~

Temporary staging and KV cache are not model parameters and must not be merged
into the parameter-residency claim.

**Test**

~~~text
BENCH-001A
~~~

**Current status**

~~~text
CONSISTENT
~~~

**Evidence**

The canonical BENCH-001A reference is stored at:

~~~text
evidence/BENCH-001A/reference/
~~~

Under the frozen source-default accounting contract it reports:

~~~text
Standard Wv accelerator-resident parameter bytes:
    201,326,592

MA-Offload GPU parameter-byte delta vs Standard:
    -201,326,592

MA table host-resident parameter bytes:
    3,145,728,000
~~~

This is consistent with the narrowed MA-004 placement claim.

Temporary staging and KV-cache storage are reported separately and are not
evidence of latency hiding or total deployment-memory savings.

MA-005 remains NOT_TESTED.

---

## MA-005 — MA-Offload latency can approach resident execution in the reported profile

**Source**

Kang, *Memory Attention*, arXiv:2609.28399 and the public profiling implementation.

**Source claim**

The paper reports profiled prefill/decode timings in which its offloaded configuration is close to the compared resident path under the reported hardware and benchmark settings.

**Local interpretation**

This is a benchmark-specific result, not a universal speed claim.

The local test must reproduce the protocol before changing hardware, batch, sequence length, or memory width.

**Protocol gate**

~~~text
BENCH-001B B0
    VALIDATED — PASS
    no performance authority
~~~

**Performance test**

~~~text
BENCH-001B B1
~~~

**Current status**

~~~text
NOT_TESTED
~~~

B0 validates source pins, capability requirements, evidence structure, raw
round statistics, and placement metadata. It does not move MA-005 because no
real CUDA performance bundle has been executed.

---

## MA-006 — Value state may be reconstructable instead of persistently cached

**Source**

Kang, *Memory Attention*, arXiv:2609.28399.

**Source claim**

The paper proposes an MA-Recall direction in which value-side cache storage may be traded for reconstruction from retained state and token-indexed memory.

**Local interpretation**

This is not accepted as a free 50% KV-cache reduction.

The precise reconstruction state must first be frozen.

Questions include:

~~~text
Which K representation is retained?
Is pre-RoPE K required?
Can post-RoPE K be inverted safely?
Are token IDs sufficient?
What layer-local metadata is required?
What precision is required?
What extra memory traffic is introduced?
~~~

**Tests**

~~~text
VAL-001
BENCH-002
~~~

**Current status**

~~~text
NOT_TESTED
~~~

---

## MA-007 — Current public reference cache path stores K and V

**Source**

Public reference implementation: Joluck/memory-attention, MemoryAttention layer.

**Observed implementation behavior**

The current public implementation updates its ordinary attention cache with both flattened key and value state.

**Local interpretation**

The repository should treat MA-Recall as a proposed extension to test independently, not as an already-demonstrated property of the public runtime path.

**Test**

Source-code readback now; executable reconstruction test later.

**Current status**

~~~text
CONSISTENT
~~~

This assessment refers only to the inspected public implementation state.

---

# Control and lineage claims

## CTRL-001 — Token-indexed capacity can be added without proportional dense compute

**Sources**

- *Large Memory Layers with Product Keys*, arXiv:1907.05242.
- *Memory Layers at Scale*, arXiv:2412.09764.
- *STEM*, arXiv:2601.10639.
- *Engram*, arXiv:2601.07372.

**Local interpretation**

Parameter count alone is insufficient to characterize runtime cost.

Relevant resource dimensions include:

~~~text
total parameters
active parameters
dense FLOPs
lookup operations
memory traffic
resident bytes
~~~

**Experiment impact**

Every later comparison must report more than a single parameter count.

**Current status**

~~~text
CONSISTENT
~~~

This means the cited sources explicitly study forms of capacity/compute decoupling; it does not mean all mechanisms are equivalent.

---

## CTRL-002 — Accelerator residency can differ from total parameter count

**Sources**

- Gemma 3n Per-Layer Embeddings.
- RWKV-8 DeepEmbed documentation.
- Memory Attention MA-Offload proposal.

**Local interpretation**

For deployment, a useful model-size report may need at least:

~~~text
total parameters
accelerator-resident parameters
host-resident parameters
active parameters per token
persistent cache bytes
~~~

**Experiment impact**

BENCH-001A should report resident bytes directly rather than infer them from total parameters.

**Current status**

~~~text
CONSISTENT
~~~

The cited systems explicitly describe non-accelerator residency for some parameter capacity.

---

## CTRL-003 — Predictable addressing enables prefetch opportunities

**Sources**

- Engram, arXiv:2601.07372.
- RWKV-8 DeepEmbed documentation.
- Memory Attention, arXiv:2609.28399.

**Local interpretation**

If the address is known from token identity or token-derived information before a later layer consumes the memory, transfer may overlap with computation.

**Important boundary**

~~~text
prefetch possible
!=
prefetch free
!=
prefetch fully hidden
~~~

**Test**

~~~text
BENCH-001B
~~~

**Current status**

~~~text
NOT_TESTED
~~~

---

## CTRL-004 — Persistent KV state can be reduced by changing the attention representation

**Sources**

- Multi-Query Attention, arXiv:1911.02150.
- Cross-Layer Attention, arXiv:2405.12981.
- DeepSeek-V2 / Multi-head Latent Attention, arXiv:2405.04434.

**Local interpretation**

KV-cache shape is an architectural choice, not an untouchable constant.

The cited methods reduce cache cost through:

~~~text
head sharing
layer sharing
latent compression
~~~

MA-Recall would represent a different strategy:

~~~text
reconstruction
~~~

**Experiment impact**

BENCH-002 should compare reconstruction against storage costs, not claim novelty from the general idea that KV state can be reduced.

**Current status**

~~~text
CONSISTENT
~~~

---

## CTRL-005 — Historical Value Embedding is not the same as the clean additive memory control

**Sources**

- KoszarskyB, *Layerwise token value embeddings*, X post (2024), as cited by Memory Attention.
- KellerJordan/modded-nanogpt, 2024-12-04 Value Embeddings record.
- MoVE, arXiv:2601.22887.
- Memory Attention, arXiv:2609.28399.

**Source recovery**

The inspected 2024-12-04 modded-nanogpt record implements:

~~~text
V_hist =
    (1 - lambda) * X Wv
    +
    lambda * E_layer[token]
~~~

with a learned scalar lambda initialized to 0.5.

Therefore the historical mechanism changes both:

~~~text
token-indexed content
and
the mixing / scaling of the ordinary value path
~~~

relative to Standard Attention.

**Local interpretation**

VAL-002 must not use one Value Embedding condition as if it simultaneously
provided exact historical reproduction and a minimal causal intervention.

The primary causal family is:

~~~text
C00:
    V = X Wv

C01:
    V = X Wv + M

C10:
    V = X Wk

C11:
    V = X Wk + M

M = Norm(E_layer[token])
~~~

The historical learned-lambda mechanism is a separate source-fidelity lane.

**Important limitation**

The source factor:

~~~text
Wv -> Wk
~~~

couples removal of the dedicated Wv projection with reuse of K.

Therefore the primary factorial does not by itself distinguish a generic
projection-sharing/removal effect from a K-specific reuse effect.

**Test**

~~~text
VAL-002
EXP-001
~~~

**Current status**

~~~text
CONSISTENT
~~~

**Evidence**

VAL-002 canonical reference evidence demonstrates, within the frozen numerical
contract:

~~~text
historical learned-lambda lane:
    known-answer error 0

local additive C01:
    known-answer error 0

historical vs C01 max abs delta:
    3.5784200537048947

factorial fixed-input interaction max abs:
    2.220446049250313e-16

parent VAL-001 anchors:
    C10 / memory / C11 / rotated K error 0
~~~

Evidence:

~~~text
evidence/VAL-002/reference/
source commit 6fa2003032285de3e01d9930b530ac13baa761f2
~~~

This assessment supports the experimental-control boundary and frozen algebra
only. It does not establish training-quality effects.


---

# Local hypotheses

These are repository hypotheses, not source claims.

## HYP-001 — Residency crossover

There exists a hardware- and workload-dependent region in which CPU-resident token memory reduces accelerator residency without causing unacceptable latency overhead.

**Variables**

~~~text
memory-table width
number of layers
batch size
sequence length
unique-token rate
dtype
host bandwidth
interconnect bandwidth
prefetch depth
~~~

**Test**

~~~text
BENCH-001B
~~~

**Status**

~~~text
NOT_TESTED
~~~

---

## HYP-002 — Reconstruction crossover

There exists a workload region in which avoiding persistent V storage saves more accelerator memory than the reconstruction path costs in memory traffic and latency.

**Test**

~~~text
BENCH-002
~~~

**Prerequisite**

~~~text
VAL-001 reconstruction contract
~~~

**Status**

~~~text
NOT_TESTED
~~~

---

## HYP-003 — Value-source decomposition changes interpretation of quality gains

The 2 x 2 memory/source factorial, plus a separate historical Value Embedding
lane, will materially change how small-model quality differences between
Standard Attention and Memory Attention should be interpreted.

**Test**

~~~text
VAL-002
EXP-001
~~~

**Status**

~~~text
NOT_TESTED
~~~

---

## HYP-004 — Total parameters are a weak deployment summary for lookup-heavy models

For models with substantial token-indexed offloadable memory, total parameter count alone will poorly predict accelerator memory requirements and may poorly predict active per-token compute.

**Tests**

~~~text
BENCH-001A
EXP-001
~~~

**Status**

~~~text
NOT_TESTED
~~~

---

# Claim-to-experiment matrix

| Claim | Type | Primary experiment | Current assessment |
| --- | --- | --- | --- |
| MA-001 | source mechanism | VAL-001 | CONSISTENT |
| MA-002 | source inference claim | VAL-001 | CONSISTENT |
| MA-003 | source quality claim | VAL-002 + EXP-001 | NOT_TESTED |
| MA-004 | source systems claim | BENCH-001A + BENCH-001B | NOT_TESTED |
| MA-005 | source benchmark claim | BENCH-001B | NOT_TESTED |
| MA-006 | source proposed extension | VAL-001 + BENCH-002 | NOT_TESTED |
| MA-007 | implementation observation | source readback | CONSISTENT |
| CTRL-001 | prior-art boundary | all | CONSISTENT |
| CTRL-002 | prior-art boundary | BENCH-001A | CONSISTENT |
| CTRL-003 | systems hypothesis from sources | BENCH-001B | NOT_TESTED |
| CTRL-004 | prior-art boundary | BENCH-002 | CONSISTENT |
| CTRL-005 | experimental-control boundary | VAL-002 + EXP-001 | CONSISTENT |
| HYP-001 | local hypothesis | BENCH-001B | NOT_TESTED |
| HYP-002 | local hypothesis | BENCH-002 | NOT_TESTED |
| HYP-003 | local hypothesis | VAL-002 + EXP-001 | NOT_TESTED |
| HYP-004 | local hypothesis | BENCH-001A + EXP-001 | NOT_TESTED |

## Rule for updates

A claim assessment may change only when the corresponding evidence identifies:

~~~text
experiment/spec identity
source commit
implementation commit
hardware/runtime provenance
measurement protocol
validity checks
structured observations
~~~

README prose, plots, or a successful CI job alone are not sufficient to change a scientific claim assessment.
