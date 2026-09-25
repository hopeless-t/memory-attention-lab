# BENCH-001 — Residency Baseline

> **Status:** ACCOUNTING EXECUTOR VALIDATED — CANONICAL PUBLICATION PENDING  
> **Scientific authority:** benchmark design only  
> **Performance authority:** NONE until a valid CUDA measurement exists  
> **Training authority:** NONE

## Goal

BENCH-001 separates two questions that are easy to blur together:

~~~text
A. residency
   where parameters and temporary transfer state live

B. performance
   what gather / H2D / overlap costs on real hardware
~~~

The first is deterministic accounting.

The second is a hardware measurement.

Neither is allowed to stand in for the other.

## Primary source

The benchmark design follows the current public Memory Attention profiling
implementation, pinned at:

~~~text
repository:
Joluck/memory-attention

main commit:
8176f1feaff6724af670e70d75cd9767f8e38223

profile/bmk.py:
9ce6143fe7df1422b2f617996395d238edd2cc1e

profile/ma_profile.py:
186c386e4d631197e40251c1fc14ad0e46c80318

profile/sweep_bmk.py:
932cf4dcac6fb51135cce441810061ab245a1d51
~~~

The upstream benchmark compares:

~~~text
standard
ma_gpu
ma_offload
~~~

and requires CUDA, BF16 support, and FlashAttention >= 2.1.

## BENCH-001A — deterministic residency accounting

This lane is hardware-independent and is intended to run exhaustively on public
GitHub Actions.

### Symbols

~~~text
H    hidden size
Nh   number of query heads
Nkv  number of KV heads
Dh   H / Nh
K    Nkv * Dh
L    number of layers
V    vocabulary size
B    batch size
S    current input sequence length
G    requested offload group size
Ge   effective group size = min(G, L)
P    prefetch depth
Z    ceil(L/Ge)
bw   bytes per ordinary model-weight element
bm   bytes per MA memory/staging element
bkv  bytes per KV-cache element
~~~

### Value-side parameter delta

In the pinned source benchmark:

~~~text
Standard:
    one Wv projection per layer

MA:
    no Wv projection
    + token-indexed memory table
~~~

For optional QKV bias:

~~~text
Wv_params_per_layer =
    H*K + (K if qkv_bias else 0)
~~~

The fused Memory Attention table contains:

~~~text
MA_table_elements =
    V*L*K
~~~

Therefore the source-scope placement deltas are:

~~~text
ma_offload GPU params - standard GPU params =
    -L * Wv_params_per_layer

ma_offload CPU params - standard CPU params =
    +V*L*K

ma_gpu GPU params - standard GPU params =
    V*L*K - L*Wv_params_per_layer
~~~

Byte deltas must use the actual element width of each allocation rather than
assuming that ordinary model weights, the MA table, and KV cache always share
one dtype:

~~~text
standard Wv bytes =
    L * Wv_params_per_layer * bw

MA table bytes =
    V * L * K * bm

ma_offload GPU parameter-byte delta vs standard =
    -standard Wv bytes

ma_offload CPU parameter-byte delta vs standard =
    +MA table bytes

ma_gpu GPU parameter-byte delta vs standard =
    MA table bytes - standard Wv bytes
~~~

These are parameter **deltas**, not complete model totals.

All common model parameters cancel from the delta.

### Temporary transfer state

The pinned pipeline offloader preallocates bounded pinned-host and GPU slots.

~~~text
effective_group =
    min(G, L)

pipeline_slots =
    min(P, ceil(L/effective_group))

pipeline pinned CPU bytes =
    pipeline_slots * B*S*effective_group*K*bm

pipeline GPU staging bytes =
    pipeline_slots * B*S*effective_group*K*bm
~~~

The exact upstream pipeline clamps a requested group larger than the layer
count to the number of layers before allocating slots. BENCH-001A mirrors that
source behavior rather than extrapolating the requested G literally.

The default decode path uses a bulk gather/H2D buffer:

~~~text
bulk pinned CPU bytes =
    B*S*L*K*bm

bulk GPU staging bytes =
    B*S*L*K*bm
~~~

These buffers are **not model parameters**.

They must never be merged into parameter counts.

### KV cache

The current source benchmark stores both K and V for Standard and Memory
Attention.

For cache capacity C:

~~~text
KV cache elements =
    2 * L * B * C * K
~~~

BENCH-001 records this separately as persistent inference state.

It does not claim any MA-Recall saving.

## BENCH-001B — source-faithful CUDA measurement

This lane measures actual latency only on an explicitly identified CUDA system.

### Variants

~~~text
standard
ma_gpu
ma_offload
~~~

### Modes

~~~text
prefill
decode
~~~

The pinned upstream defaults use:

~~~text
batch size       8
prefill length   2048
decode prefix    2048
hidden size      2048
heads            32
layers           24
vocabulary       32000
group size       1
prefetch depth   4
decode offload   bulk
warmup           5
repeats          30
rounds           5
~~~

These defaults are a source-reproduction anchor, not a universal recommended
configuration.

## Offload mechanisms

### Pipeline

The upstream pipeline path uses:

~~~text
CPU gather
    into pinned host slot

asynchronous H2D
    on dedicated CUDA copy stream

bounded lookahead

wait for memory dependency
    only when K + M is consumed
~~~

The producer can submit transfer work before the later attention layer reaches
the value addition.

### Bulk decode

The default decode path gathers all layer memories for the current token into
one host buffer and performs one bulk H2D transfer.

This trades per-layer coordination for larger up-front transfer.

BENCH-001 should measure both bulk and pipeline decode if hardware time permits.

## Timing scope

The pinned upstream model timing includes:

~~~text
input embedding
all Transformer blocks
Q/K/V or K+M construction
RoPE
KV-cache writes
FlashAttention
gating if enabled
output projection
MLP and residual paths
final RMSNorm
untied LM head
CPU gather and H2D work enabled by the offloader
~~~

It excludes:

~~~text
prefix construction
offline memory normalization folding
token-ID device-to-host transfer
sampling
loss / backward / optimizer
~~~

Both CPU and GPU token IDs are ready before the timed forward.

This exclusion must be visible in every source-faithful result.

A later end-to-end lane may include token-ID movement, but it must use a
different benchmark-scope label.

## Parameter-count scope

The upstream benchmark includes model weights such as:

~~~text
input embedding
untied LM head
Q/K/V/O where applicable
MLP
norms
optional gates
Memory Attention table
~~~

It excludes:

~~~text
KV cache
activations
transfer buffers
deterministic RoPE buffers
CPU staging copies used only to initialize GPU weights
~~~

BENCH-001A follows the same categories.

## Correctness before timing

A CUDA run is invalid unless its offload path first passes the declared
correctness checks.

The pinned source benchmark checks:

~~~text
exact output equality against its reference offloader path

exact written K cache equality

exact written V cache equality

reuse with changed token IDs and revisit of the first input
~~~

This repository may add independent checks, but may not weaken the source
correctness gate silently.

## Measurement outputs

### BENCH-001A

Required structured outputs:

~~~text
standard Wv parameter count / bytes

MA table parameter count / bytes

GPU parameter delta
CPU parameter delta

pipeline slot count
pipeline pinned-host bytes
pipeline GPU staging bytes

bulk pinned-host bytes
bulk GPU staging bytes

KV-cache bytes reported separately
~~~

### BENCH-001B

Required structured outputs include:

~~~text
hardware identity
software versions
benchmark source commit
configuration
timing scope
offload policy

per-round latency samples
median latency
min / max diagnostics

parameter placement
staging bytes

correctness-check result
~~~

Min/max are diagnostics, not confidence intervals.

## GitHub Actions use

Standard public Actions runners are used heavily for:

~~~text
spec validation
formula property tests
large randomized parameter sweeps
cross-platform accounting
overflow / invalid-shape fuzzing
decision-support Monte Carlo
artifact schema validation
~~~

Host-only timing on ephemeral shared VMs is diagnostic only.

It is not canonical CUDA performance evidence.

## Decision support

See:

~~~text
research/decisions/DS-004-bench001-evidence-lanes.md
research/bench001_design_mc.py
~~~

The selected design is dual-lane.

The scope/cost stress case deliberately prefers accounting-only, preventing the
dual-lane decision from being treated as a theorem.

## BENCH-001A executable reference

The deterministic accounting executor is implemented at:

~~~text
src/memory_attention_lab/measurement/residency.py
src/memory_attention_lab/experiments/bench001a.py
specs/BENCH-001A.reference.json
~~~

The reference fixture contains four explicit cases:

~~~text
small known answer
source-default prefill anchor
group-size > layers clamp guard
GQA + bias + mixed element-width guard
~~~

The latest validated branch run also executes:

~~~text
full pytest:
    Ubuntu 24.04 / Python 3.12
    Ubuntu 24.04 / Python 3.13
    macOS 15 / Python 3.12
    macOS 15 / Python 3.13

randomized accounting:
    500,000 cases

claim-partition decision support:
    1.5 million simulated engineering decision states
~~~

All passed on the validated candidate.

### Source-default accounting anchor

For the pinned upstream default dimensions with BF16-sized model/memory/cache
elements and prefill length/cache capacity 2048:

~~~text
Standard Wv parameter bytes:
    201,326,592
    192 MiB

MA table bytes:
    3,145,728,000
    approximately 2.93 GiB

MA-Offload GPU parameter-byte delta vs Standard:
    -201,326,592
    -192 MiB

pipeline GPU staging bytes:
    268,435,456
    256 MiB

bulk GPU staging bytes:
    1,610,612,736
    1.5 GiB

logical H2D payload per full forward:
    1,610,612,736
    1.5 GiB

KV-cache bytes at capacity 2048:
    3,221,225,472
    3.0 GiB
~~~

These are exact deterministic accounting outputs under the frozen configuration.

They are **not** measured CUDA latency, peak allocator usage, or proof that the
transfer can be hidden.

## Next implementation step

Publish the reviewed BENCH-001A evidence from a merged-main run.

Only after BENCH-001A is canonical should BENCH-001B CUDA measurement be
implemented or borrowed.

## Non-claims

A BENCH-001A PASS will not mean:

- CPU offload is fast;
- PCIe transfer is hidden;
- MA-Offload matches GPU-resident latency;
- the source paper's H800 timing is reproduced;
- total deployment memory is reduced by the parameter delta alone;
- MA-Recall reduces KV cache.

A BENCH-001B result will be scoped to its exact hardware and protocol.
