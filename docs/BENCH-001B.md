# BENCH-001B — CUDA Performance Evidence Protocol

> **Status:** B0 PROTOCOL VALIDATED — PASS · B1 NOT EXECUTED  
> **Scientific authority:** protocol validation only  
> **Performance authority:** NONE until real CUDA evidence is reviewed  
> **Paid-resource authority:** NONE

## Purpose

BENCH-001A established deterministic placement and working-set accounting.

BENCH-001B asks the separate empirical question:

> On a qualifying CUDA system, what latency and placement behavior does the
> pinned upstream Memory Attention benchmark actually measure?

The lane is split again so that hardware availability does not block protocol
work.

~~~text
B0
    protocol / capability / evidence validation
    CPU GitHub Actions is sufficient

B1
    real CUDA performance execution
    qualifying GPU required
~~~

A B0 PASS is **not** a performance result.

## Pinned source

~~~text
repository:
Joluck/memory-attention

commit:
8176f1feaff6724af670e70d75cd9767f8e38223

profile/bmk.py blob:
9ce6143fe7df1422b2f617996395d238edd2cc1e
~~~

The upstream benchmark itself requires:

~~~text
torch.cuda.is_available()
CUDA device
torch.cuda.is_bf16_supported()
flash-attn >= 2.1
~~~

It stores per-round timing samples and summary statistics in its JSON output.

## Why standard public Actions cannot produce B1

Standard GitHub-hosted public-repository runners provide CPU VMs, not the CUDA
GPU required by the pinned benchmark.

GitHub's documented GPU larger runner is currently an NVIDIA Tesla T4 with
16 GB VRAM.

NVIDIA lists T4 as compute capability 7.5.

The CUDA Programming Guide lists bfloat16 as requiring compute capability 8.0
or higher.

Therefore the documented GitHub T4 larger runner fails the source benchmark's
BF16 capability floor before performance is considered.

The larger-runner path is also a billed Team/Enterprise organization feature,
not part of the unlimited standard public-runner lane.

## Why a persistent public self-hosted runner is not the default

GitHub explicitly warns about using self-hosted runners with public
repositories because fork-originated workflows can expose the runner machine
to untrusted code.

BENCH-001B therefore does not make a persistent personal/workstation runner its
default execution authority.

A future self-hosted execution would need a separately reviewed isolation and
trigger policy.

## Selected topology

DS-006 selects:

~~~text
frozen B0 protocol
      |
explicit Human approval of a concrete GPU resource
      |
isolated / disposable qualifying CUDA executor
      |
pinned upstream benchmark
      |
hardware.json
bmk_results.json
      |
public GitHub Actions validator
      |
reviewed B1 evidence
~~~

The GPU executor is an evidence producer.

GitHub Actions remains the public evidence-validation and regression layer.

## Paid-resource boundary

No paid GPU instance may be started from this protocol.

If a candidate execution environment has a monetary cost, explicit Human
approval is required before B1 execution.

Developing and validating B0 on standard public Actions does not spend GPU
resources.

## Hardware qualification floor

A B1 bundle must report:

~~~text
CUDA-capable NVIDIA GPU
compute capability >= 8.0
BF16 support = true
flash-attn >= 2.1
positive VRAM capacity
runtime / driver / OS metadata
~~~

The protocol-level compute-capability check is a screening rule.

The stronger runtime requirement remains:

> the pinned upstream setup_device() must actually succeed.

## Source-default profile

The initial B1 profile is named:

~~~text
source_default_latency
~~~

It freezes the upstream default model/workload dimensions while enabling the
upstream correctness guard:

~~~text
mode:
    both

variants:
    standard
    ma_gpu
    ma_offload

check_correctness:
    true

batch:
    8

prefill sequence:
    2048

decode context:
    2048

hidden:
    2048

heads:
    32

KV heads:
    32

layers:
    24

vocab:
    32000

group size:
    1

prefetch depth:
    4

decode offload:
    bulk

GPU MA lookup:
    layerwise

warmup:
    5

repeats per round:
    30

rounds:
    5

timing:
    latency

seed:
    1234
~~~

The complete parsed configuration is frozen in:

~~~text
specs/BENCH-001B.protocol.json
~~~

## Required result rows

The raw upstream JSON must contain exactly:

~~~text
standard   / prefill
ma_gpu     / prefill
ma_offload / prefill

standard   / decode
ma_gpu     / decode
ma_offload / decode
~~~

For each row the validator recomputes:

~~~text
median(round_ms)
min(round_ms)
max(round_ms)
speedup_vs_standard
~~~

and checks the source-specific placement and offload metadata.

## Evidence bundle

### hardware.json

Records:

~~~text
execution id
synthetic flag
source pin

GPU name
compute capability
VRAM
BF16 support

Python
PyTorch
CUDA runtime
driver
flash-attn
OS

profile
command argv
SHA-256 of bmk_results.json
~~~

### bmk_results.json

Must be the raw JSON emitted by the pinned upstream benchmark.

The hardware manifest cryptographically binds to these bytes using SHA-256.

## Validator authority

The CPU-side validator can establish:

~~~text
source identity
profile identity
hardware-floor declaration
BF16 / flash-attn declaration
result-file hash binding
six-row completeness
round-count completeness
finite positive timings
summary-statistic consistency
speedup consistency
parameter-placement consistency
offload-policy consistency
~~~

It cannot cryptographically prove that a claimed GPU name is genuine.

The strongest available evidence is therefore a combination of:

~~~text
pinned executor code
successful upstream capability gate
raw timing rounds
hardware/runtime manifest
hash binding
public validator
human review
~~~

No remote hardware attestation claim is made.

## Synthetic fixture rule

The repository contains a synthetic timing fixture only to Red Team the
validator.

~~~text
tests/fixtures/BENCH-001B/synthetic/
~~~

Synthetic bundles:

~~~text
MUST be rejected by the normal validator path
MAY be admitted only by unit tests with allow_synthetic=True
MUST NEVER become BENCH-001B performance evidence
~~~

The synthetic timing numbers have **zero performance authority**.

## Current execution state

~~~text
BENCH-001A
    FROZEN REFERENCE — PASS

BENCH-001B B0
    PROTOCOL VALIDATED — PASS

BENCH-001B B1 capture kit
    VALIDATED — PASS
    NO PERFORMANCE AUTHORITY

BENCH-001B B1 CUDA execution
    NOT EXECUTED
    NO GPU RESOURCE AUTHORIZED
~~~

## External platform references

GitHub:

- https://docs.github.com/en/actions/reference/runners/github-hosted-runners
- https://docs.github.com/en/actions/reference/runners/larger-runners
- https://docs.github.com/en/actions/concepts/runners/larger-runners
- https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners

NVIDIA:

- https://developer.nvidia.com/cuda/gpus
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html

## Non-claims

A B0 PASS does not establish:

- any measured CUDA latency;
- any speedup;
- latency neutrality;
- H800 reproduction;
- peak GPU-memory behavior;
- PCIe overlap effectiveness;
- CPU-gather cost;
- MA-Recall benefit;
- Memory Attention superiority.


## B0 validation result

Fresh public Actions run:

~~~text
run:
36079694609

source commit:
e75eb00d147ea9a5aa5f49d959a09b0f6a967dff
~~~

Validated on:

~~~text
Ubuntu 24.04 / Python 3.12
Ubuntu 24.04 / Python 3.13
macOS 15 / Python 3.12
macOS 15 / Python 3.13

full pytest:
PASS
~~~

DS-006 execution-topology decision support also passed with 300,000 samples per
scenario:

| Scenario | Selected topology C |
| --- | ---: |
| balanced | 99.30% |
| source fidelity heavy | 99.05% |
| public-repo security heavy | 97.57% |
| paid-resource control heavy | 84.59% |
| automation heavy | 79.95% |

These percentages are engineering decision-support frequencies, not scientific
probabilities.

The next state transition requires an identified qualifying GPU execution
resource. No such resource is authorized by B0.


## B1 capture-kit checkpoint

The B1 capture kit and runbook are validated on standard public Actions.

See:

~~~text
src/memory_attention_lab/experiments/bench001b_capture.py
tests/test_bench001b_capture.py
docs/BENCH-001B-RUNBOOK.md
~~~

The validated default remains **probe-only**.

Real benchmark timing requires explicit `--execute` after the execution
resource has been approved.

This checkpoint does not move MA-005.


## B1 GPU candidate checkpoint

DS-007 records the current pre-approval GPU candidate priority.

~~~text
default first probe candidate:
    Runpod Secure A40 48 GB

OOM/headroom or source-similarity fallback:
    Runpod Secure A100 80 GB

minimum-spend alternative:
    Runpod Secure RTX A5000 24 GB
    only if Human accepts higher headroom risk
~~~

Fresh public Actions decision-support run:

~~~text
run:
36082545945

balanced:
    A40 48 GB 67.82%

OOM-risk-heavy:
    A100 80 GB 52.57%

budget-heavy:
    A40 48 GB 67.76%

reproducibility-heavy:
    A40 48 GB 53.88%

source-similarity-heavy:
    A100 80 GB 54.21%
~~~

These are engineering decision-support frequencies, not probabilities that a
GPU will complete the benchmark.

No provider account, instance, GPU, billing event, or execution is authorized
by this checkpoint.

The next state transition remains:

~~~text
Human selects a concrete resource and current price
Human explicitly approves any spend
capture kit runs probe-only
Human reviews probe.json
only then may --execute be used
~~~


## Google Cloud reevaluation checkpoint

DS-008 reevaluates the proposed Google Cloud / RTX PRO 6000 path.

The product boundary is:

~~~text
Cloud Build
    control-plane candidate only
    not the GPU execution worker

Compute Engine G4
    RTX PRO 6000 Blackwell 96 GB
    exploratory Blackwell lane
    not canonical under the frozen FA2 protocol

Compute Engine A2
    A100 40 GB
    canonical Google candidate

Compute Engine A2 Ultra
    A100 80 GB
    canonical Google headroom/provenance candidate
~~~

The reason G4 is not promoted into the current B1 lane is software fidelity,
not hardware capability:

~~~text
RTX PRO 6000 Blackwell:
    compute capability 12.0
    BF16-capable hardware

current official FlashAttention-2 CUDA support:
    Ampere
    Ada
    Hopper
~~~

Changing the benchmark to FA4 or a patched/source-built SM120 path requires a
separate protocol and must not silently inherit BENCH-001B authority.

Fresh decision-support validation:

~~~text
run:
36084289427

Google Cloud reevaluation:
    1.8 million states PASS

balanced:
    Runpod Secure A40 48 GB 51.54%

budget-heavy:
    Runpod Secure A40 48 GB 95.24%

provenance-heavy:
    GCP A2 Ultra A100 80 GB 47.74%

Japan-proximity-heavy:
    GCP A2 A100 40 GB 58.12%

headroom-heavy:
    GCP A2 Ultra A100 80 GB 51.30%

cloud-credit-heavy:
    GCP A2 Ultra A100 80 GB 58.13%
~~~

These are engineering decision-support frequencies, not benchmark-success
probabilities.

No Google Cloud resource creation is authorized by this checkpoint.
