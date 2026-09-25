# DS-006 — BENCH-001B execution topology

> **Status:** ACCEPTED DESIGN DECISION  
> **Scientific authority:** NONE  
> **Performance authority:** NONE  
> **Paid-resource authority:** NONE  
> **Decision date:** 2026-09-25

## Question

How should BENCH-001B obtain source-faithful CUDA performance evidence without
confusing public GitHub Actions convenience with actual hardware suitability?

## Upstream hard requirements

The pinned upstream benchmark performs these runtime checks:

~~~text
CUDA must be available
device must be CUDA
torch.cuda.is_bf16_supported() must be true
flash-attn >= 2.1
~~~

Pinned source:

~~~text
Joluck/memory-attention
commit:
8176f1feaff6724af670e70d75cd9767f8e38223

profile/bmk.py blob:
9ce6143fe7df1422b2f617996395d238edd2cc1e
~~~

The benchmark records timing rounds, parameter placement, staging buffer sizes,
and source-scope model metadata into JSON.

## Current GitHub runner constraint

Standard GitHub-hosted runners for public repositories are useful for the
validator lane but do not provide the CUDA GPU required by BENCH-001B.

GitHub's documented GPU larger-runner size uses an NVIDIA Tesla T4 with 16 GB
VRAM.

NVIDIA documents Tesla T4 as compute capability 7.5, while the CUDA Programming
Guide documents native bfloat16 support as requiring compute capability 8.0 or
higher.

Therefore the current GitHub T4 larger runner is **not source-faithful for this
benchmark's BF16 hard gate**.

Larger runners are also a billed Team/Enterprise organization feature rather
than a free extension of standard public-repository runners.

## Public self-hosted boundary

GitHub documentation warns that self-hosted runners attached to public
repositories can expose the runner machine to dangerous code from fork-based
workflows.

A persistent general-purpose workstation must therefore not become the default
BENCH-001B execution authority merely for convenience.

## Candidate topologies

### A — GitHub GPU larger runner

~~~text
GitHub-hosted Tesla T4
-> benchmark
-> artifact
~~~

Rejected as the default source-faithful lane because the documented T4 is
compute capability 7.5 and fails the benchmark's BF16 requirement.

### B — Persistent public-repository self-hosted GPU

~~~text
public repo
-> persistent local/cloud GPU runner
-> benchmark
~~~

Technically viable with suitable Ampere-or-newer hardware, but rejected as the
default because of the public-repository runner security boundary.

### C — Isolated approved GPU execution + public Actions validator

~~~text
frozen protocol
      |
Human approves a concrete GPU execution resource
      |
isolated / disposable Ampere-or-newer CUDA executor
      |
upstream source-faithful benchmark
      |
raw benchmark JSON + hardware/runtime manifest + hashes
      |
public GitHub Actions evidence validator
      |
reviewed BENCH-001B evidence
~~~

Selected.

The GPU executor is an evidence producer, not a long-lived repository worker.

### D — Defer all BENCH-001B work until hardware exists

Safe but unnecessarily blocks protocol, schema, provenance, and validator work
that can be completed today on free public Actions.

Rejected as the default.

## Hardware qualification floor

A performance artifact may enter BENCH-001B only if the executor reports and
the source benchmark successfully passes:

~~~text
CUDA available
CUDA device selected
torch.cuda.is_bf16_supported() == true
flash-attn version >= 2.1
upstream setup_device() succeeds
~~~

For protocol-level screening, the GPU must be compute capability >= 8.0.

A successful benchmark execution remains the stronger runtime gate.

## Two-stage BENCH-001B structure

### B0 — protocol / capability qualification

No performance claim.

GitHub Actions may validate:

~~~text
source pin
benchmark configuration
hardware manifest schema
runtime-version manifest
result JSON structure
round count
finite timing samples
summary-statistic recomputation
hash binding
authority labels
~~~

### B1 — CUDA performance evidence

Requires a concrete approved GPU execution resource.

Measures:

~~~text
standard
ma_gpu
ma_offload

prefill
decode

median / min / max / per-round latency
parameter placement
offload working-set metadata
speedup ratio vs Standard
~~~

The source-default run uses the upstream defaults unless a separately named
reduced profile is explicitly frozen.

## Paid-resource rule

This decision does **not** authorize buying or starting a paid GPU instance.

Any paid GPU resource requires explicit Human approval before execution.

Protocol/validator development on free public Actions does not require such
approval.

## Monte Carlo decision support

The committed simulation evaluates:

~~~text
balanced
source-fidelity-heavy
public-repo-security-heavy
paid-resource-control-heavy
automation-heavy
~~~

using 300,000 samples per scenario.

The design run selected option C in every scenario.

This is engineering decision support, not a probability of success or a
provider reliability forecast.

## Decision

Adopt:

~~~text
C — isolated approved GPU execution + public Actions validator
~~~

Proceed now with B0 protocol/schema/validator work.

Do not execute B1 until a qualifying GPU resource is identified and, if it has
cost, explicitly approved.

## References

- GitHub standard runner reference:
  https://docs.github.com/en/actions/reference/runners/github-hosted-runners
- GitHub larger runner reference:
  https://docs.github.com/en/actions/reference/runners/larger-runners
- GitHub larger runner concepts/billing:
  https://docs.github.com/en/actions/concepts/runners/larger-runners
- GitHub self-hosted runner security warning:
  https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners
- NVIDIA CUDA GPU compute capability:
  https://developer.nvidia.com/cuda/gpus
- NVIDIA CUDA floating-point type requirements:
  https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html
