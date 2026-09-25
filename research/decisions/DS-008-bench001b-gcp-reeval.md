# DS-008 — Google Cloud Build / G4 reevaluation

> **Status:** ACCEPTED DECISION SUPPORT  
> **Scientific authority:** NONE  
> **Performance authority:** NONE  
> **Purchase authority:** NONE  
> **Decision date:** 2026-09-25

## Trigger

A possible Google Cloud execution path was proposed using an NVIDIA RTX PRO
6000 Blackwell-class GPU.

The proposal is useful, but the product boundary and attention-kernel boundary
must be separated before changing BENCH-001B.

## Finding 1 — RTX PRO 6000 is Compute Engine G4, not a Cloud Build worker

Google Cloud currently offers the NVIDIA RTX PRO 6000 Blackwell Server Edition
through the Compute Engine **G4** accelerator-optimized machine series.

A full `g4-standard-48` instance provides:

~~~text
GPU:
    NVIDIA RTX PRO 6000 Blackwell Server Edition

GPU count:
    1

GPU memory:
    96 GB GDDR7

vCPU:
    48

host memory:
    180 GiB
~~~

Cloud Build private pools are a different product surface.

The current Cloud Build private-pool schema lists supported worker machine
families from:

~~~text
e2
n2d
c3
~~~

and does not list G4 accelerator-optimized workers.

Therefore:

~~~text
Cloud Build
    != RTX PRO 6000 execution worker

Compute Engine G4
    = RTX PRO 6000 execution worker
~~~

Cloud Build could later be used as an orchestration/control plane that invokes
Compute Engine APIs, but it is not the GPU execution plane itself.

## Finding 2 — G4 hardware qualifies for BF16, but the frozen FA2 software lane does not

NVIDIA lists RTX PRO 6000 Blackwell as compute capability:

~~~text
12.0
~~~

The hardware therefore clears the repository's old numeric
`compute_capability >= 8.0` screen.

However, that screen is insufficient.

The current official FlashAttention repository states that
FlashAttention-2 CUDA support is for:

~~~text
Ampere
Ada
Hopper
~~~

and BF16 in the FA2 lane is specified for those architectures.

RTX PRO 6000 Blackwell is SM120.

Open upstream reports in 2025-2026 document that RTX PRO 6000 / SM120 is not
the normal supported FA2 path.

FlashAttention-4 is explicitly the newer Hopper/Blackwell-oriented path, but
switching the frozen Memory Attention benchmark from its current
FlashAttention-2 dependency to FA4 would change the execution protocol.

Therefore:

~~~text
G4 / RTX PRO 6000 Blackwell
    excellent hardware
    NOT canonical for the currently frozen BENCH-001B FA2 lane
~~~

It can become a separate exploratory Blackwell lane only after a distinct
kernel/protocol decision.

## Finding 3 — Google Cloud still has strong canonical options

Compute Engine A2 provides source-compatible Ampere GPUs.

### A2 Standard

~~~text
machine:
    a2-highgpu-1g

GPU:
    NVIDIA A100 40 GB

GPU count:
    1

host memory:
    85 GB
~~~

A2 Standard is currently listed in Tokyo zones.

### A2 Ultra

~~~text
machine:
    a2-ultragpu-1g

GPU:
    NVIDIA A100 80 GB

GPU count:
    1

host memory:
    170 GB
~~~

Both A100 variants remain inside the official FlashAttention-2
Ampere support envelope.

## Cloud Build role

For a first canonical Google execution, the minimal trusted path is:

~~~text
Human approval
    ->
direct Compute Engine A2 instance
    ->
BENCH-001B capture kit probe-only
    ->
Human review of probe.json
    ->
explicit --execute
    ->
evidence bundle
    ->
public GitHub Actions validator
~~~

Using Cloud Build to create/delete the VM is technically possible as an
orchestration pattern if the build service account is granted Compute Engine
permissions.

However, that would make a CI control plane capable of creating billed GPU
resources.

The repository currently has:

~~~text
paid_resource_authority = false
~~~

so Cloud Build GPU provisioning is **not** enabled by this decision.

A later automation decision can add:

~~~text
Cloud Build
    -> create disposable A2 VM
    -> capture
    -> collect evidence
    -> delete VM
~~~

only after Human approval and explicit budget/cleanup safeguards.

## Current public pricing observations

Google's accelerator-optimized pricing page currently lists approximately:

~~~text
A2 Standard / A100 40 GB:
    a2-highgpu-1g
    on-demand table value about $3.67 / hour

A2 Ultra / A100 80 GB:
    a2-ultragpu-1g
    on-demand table value about $5.07 / hour

G4 / RTX PRO 6000 96 GB:
    g4-standard-48
    on-demand table value about $4.50 / hour
    Spot table value about $1.72 / hour
~~~

Actual billed price depends on region, provisioning model, discounts, and
account configuration.

For comparison, DS-007 recorded materially lower current Runpod Secure prices,
so Google Cloud is not automatically the cost-first choice.

## G4 regional note

Current Google GPU location documentation lists G4 in Tokyo zone:

~~~text
asia-northeast1-b
~~~

A2 Standard is currently listed in Tokyo zones:

~~~text
asia-northeast1-a
asia-northeast1-c
~~~

Therefore regional proximity no longer distinguishes G4 from A2 in Tokyo.

G4 remains outside the current canonical BENCH-001B lane because of the frozen
FlashAttention-2 / Blackwell software-compatibility boundary, not because of
Tokyo availability.

## Candidate policy

### Canonical frozen FA2 lane

Eligible:

~~~text
Runpod Secure A40 48 GB
Runpod Secure A100 80 GB

Google Compute Engine A2 A100 40 GB
Google Compute Engine A2 Ultra A100 80 GB
~~~

### Exploratory Blackwell lane

Not canonical under current B1:

~~~text
Google Compute Engine G4
RTX PRO 6000 Blackwell 96 GB
~~~

A future Blackwell lane must explicitly decide among:

~~~text
FA2 source build / patch
FA4
alternative kernel path
upstream Memory Attention changes
~~~

and must not silently inherit BENCH-001B authority.

## Monte Carlo decision support

After the Blackwell hard gate, four canonical candidates were compared across:

~~~text
source fidelity
VRAM headroom
isolation / provenance
region proximity
cost efficiency
setup reproducibility
availability
orchestration control
~~~

300,000 samples are run per scenario.

Frozen design-run result:

| Scenario | Winner | Approx. win rate |
| --- | --- | ---: |
| balanced | Runpod Secure A40 48 GB | 51.54% |
| budget heavy | Runpod Secure A40 48 GB | 95.24% |
| provenance heavy | GCP A2 Ultra A100 80 GB | 47.74% |
| Japan proximity heavy | GCP A2 A100 40 GB | 58.12% |
| headroom heavy | GCP A2 Ultra A100 80 GB | 51.30% |
| cloud-credit heavy | GCP A2 Ultra A100 80 GB | 58.13% |

These are engineering decision-support frequencies, not probabilities of
benchmark success.

## Decision

Do **not** replace BENCH-001B's canonical executor with G4/RTX PRO 6000 under
the current FA2 protocol.

Instead:

~~~text
Cost-first default:
    retain Runpod Secure A40 48 GB

Google canonical option:
    Compute Engine A2 / A100 40 GB

Google headroom/provenance option:
    Compute Engine A2 Ultra / A100 80 GB

Blackwell research option:
    Compute Engine G4 / RTX PRO 6000 96 GB
    separate future protocol
~~~

Cloud Build may be used later as a **control plane**, not as the GPU worker,
after an explicit Human-approved automation/budget decision.

## Human approval gate

This decision does not authorize any Google Cloud resource creation.

Before a paid Google execution:

~~~text
Human selects:
    project
    zone
    machine type
    provisioning model
    price ceiling

Human explicitly approves spend

capture kit runs probe-only

Human reviews probe.json

only then:
    --execute
~~~

## Sources

Cloud Build private-pool machine schema:
https://docs.cloud.google.com/build/docs/private-pools/private-pool-config-file-schema

Google Compute Engine G4:
https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines

Google GPU locations:
https://docs.cloud.google.com/compute/docs/regions-zones/gpu-regions-zones

Google accelerator-optimized pricing:
https://cloud.google.com/products/compute/pricing/accelerator-optimized

Google Spot VM pricing:
https://cloud.google.com/spot-vms/pricing

NVIDIA compute capability:
https://developer.nvidia.com/cuda/gpus

FlashAttention official repository:
https://github.com/Dao-AILab/flash-attention

FlashAttention SM120 issue context:
https://github.com/Dao-AILab/flash-attention/issues/1987
