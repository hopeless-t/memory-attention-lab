# memory-attention-lab

Independent reproducibility and systems research for **Memory Attention**, with a focus on accelerator residency, offloading, and reconstructable attention state.

> **Status: STAGING RESEARCH**
>
> **NOT CANONICAL · NO IMPLEMENTATION AUTHORITY**
>
> This repository is an independent research project. It is not the official Memory Attention implementation, and experimental findings here do not by themselves establish architectural superiority or imply adoption by any other project.

## Core question

> **How much value-side state in attention must actually remain resident on the accelerator?**

Memory Attention suggests an unusual decomposition of attention state:

```text
context-dependent state   -> compute
token-dependent memory    -> lookup
persistent accelerator state
                         -> maybe offload
cached value state        -> maybe reconstruct
```

This repository tests that decomposition as a small, reproducible systems-research problem.

The goal is not merely to ask whether Memory Attention "works."

The goal is to separate and measure:

```text
COMPUTE
RESIDENCY
BANDWIDTH
RECONSTRUCTION
QUALITY
```

and determine which trade-offs are real under controlled conditions.

## Research origin

This project is motivated by **Memory Attention** (arXiv:2609.28399) and its public reference implementation.

The core construction replaces a separately projected value tensor with a contextual key contribution plus token-indexed learned memory:

```text
Q  = q_proj(X)
K0 = k_proj(X)
M  = memory[token_id]

V  = K0 + Norm(M)
```

The public implementation currently stores both keys and values in its ordinary cache path. This creates an independently testable systems question:

> If a value can be reconstructed from retained key-side state and token-indexed memory, when is storing that value preferable to reconstructing it?

That question is the initial systems focus of this repository.

## Scientific and engineering boundary

This repository does **not** currently claim to:

- reproduce all results from the Memory Attention paper;
- establish that Memory Attention is better than standard attention;
- establish a universal latency advantage;
- establish a universal memory advantage;
- show that CPU or storage offload is always beneficial;
- prove that reconstructed values are cheaper than cached values in deployment;
- infer model-quality gains from parameter-count increases;
- provide production-ready serving infrastructure.

The initial work deliberately starts below those claims.

A paper claim is not a reproduced result.

A successful program execution is not a successful research result.

A benchmark being faster once is not evidence of a general speed advantage.

## Research decomposition

The initial research object is value-side attention state:

```text
VALUE-SIDE STATE
      |
      +-- must be computed contextually
      |
      +-- can be represented by token lookup
      |
      +-- can live outside accelerator memory
      |
      +-- can be reconstructed instead of cached
```

This decomposition is intentionally broader than one implementation.

Memory Attention is the first concrete architecture used to test it.

## Research pipeline

The repository follows an explicit evidence path:

```text
Question
   |
   v
Experiment Spec
   |
   v
Execution
   |
   v
Measurement
   |
   v
Validity Checks
   |
   v
Evidence
   |
   v
Claim Assessment
```

The implementation is not allowed to turn a measurement directly into a scientific conclusion.

## Execution states

Every experiment or benchmark must end in one of four execution states:

```text
PASS
    The run executed and every validity check required by the experiment
    contract passed.

FAIL
    The run executed, but one or more declared validity checks failed.

INVALID
    The specification was rejected before valid execution began.

ERROR
    Runtime, software, hardware, or I/O failure prevented valid evaluation.
```

A `PASS` does **not** mean that a research claim was confirmed.

It means only that the experiment was validly executed under its declared contract.

## Claim-assessment states

Research claims use a separate vocabulary:

```text
NOT_TESTED
CONSISTENT
INCONSISTENT
INCONCLUSIVE
```

For example:

```text
BENCH-001: PASS
claim:     INCONCLUSIVE
```

is a normal and valid outcome.

This separation is intentional.

## Initial research sequence

The first stages are ordered so that the benchmark harness must earn trust before larger model experiments begin.

### VAL-001 — Memory Attention Algebra

Establish the minimal independently inspectable implementation of the Memory Attention value construction.

Primary checks include:

- tensor-shape validity;
- token-indexed memory lookup;
- normalization behavior;
- separation of pre-RoPE key state from the attention key;
- deterministic known-answer tests;
- cached-value versus reconstructed-value equivalence under a controlled precision contract;
- deliberate failure injection.

No performance advantage is claimed at this stage.

### BENCH-001 — Residency Baseline

Compare at least:

```text
standard attention
Memory Attention / GPU-resident memory
Memory Attention / CPU-resident memory
```

Candidate measurements include:

```text
GPU parameter bytes
CPU parameter bytes
peak accelerator memory
prefill latency
decode latency
tokens / second
host -> device bytes
```

Benchmark validity and benchmark outcome remain separate.

A slower configuration can still produce a valid `PASS` benchmark.

### BENCH-002 — Value Reconstruction / MA-Recall

Compare conventional value caching against reconstructing values from retained state.

Conceptually:

```text
ordinary cache

K K K K K
V V V V V
```

versus:

```text
reconstruction-oriented cache

K K K K K
token ids

    |
    v
V = K0 + Memory[token]
```

Candidate measurements include:

```text
persistent cache bytes
reconstruction latency
memory-table traffic
decode latency
peak accelerator memory
host-memory pressure
```

This stage must first demonstrate semantic equivalence within a declared numerical tolerance before performance comparisons are interpreted.

### EXP-001 — Controlled Small-Model Comparison

Training experiments are deferred until the validation and benchmark harnesses are trustworthy.

The first model-quality experiment should separate fairness constraints rather than collapse them into one vague notion of "matched models."

Candidate comparison axes include:

```text
matched training tokens
matched total parameters
matched active parameters
matched accelerator-resident parameters
matched compute budget
```

These are different experiments.

They should not be silently treated as interchangeable.

## Evidence model

A canonical research result should be reconstructable from small, structured artifacts.

Expected evidence objects include:

```text
manifest.json
metrics.json
checks.json
samples.csv
```

A manifest should record enough provenance to identify the actual calculation, including where relevant:

```text
paper version
upstream implementation commit
local implementation commit
experiment spec hash

Python
PyTorch
CUDA / ROCm
FlashAttention / Triton versions

CPU
GPU
RAM

dtype
model dimensions
sequence length
batch size
warmup count
measurement count
seed
```

Large profiling outputs are not canonical Git evidence by default.

Examples of non-canonical artifacts:

```text
Chrome traces
Nsight traces
large profiler captures
model checkpoints
dataset caches
temporary compiled objects
```

These may be retained locally or published separately when an experiment requires them.

## Reproducibility principles

This project begins with several fixed rules:

1. **Spec before result.** The intended experiment must be machine-readable before its output becomes evidence.
2. **Fail closed.** Unknown or invalid experiment fields should be rejected rather than silently ignored.
3. **Known-answer first.** A benchmark harness must first demonstrate that it can recognize correct and deliberately broken behavior.
4. **Figure != evidence.** Plots are for human inspection unless an experiment contract explicitly says otherwise.
5. **Plan != result.** Roadmap entries and hypotheses are not findings.
6. **Source != reproduction.** Citing a paper or running its code does not automatically reproduce its claims.
7. **Performance needs protocol.** Warmup, synchronization, sample count, device state, dimensions, and precision must be reported with timing results.
8. **No premature framework.** Shared abstractions are added only after multiple concrete experiments require them.

## Planned repository shape

The repository follows a small research-lab pattern rather than mirroring the full upstream implementation.

```text
memory-attention-lab/
|
+-- README.md
+-- LICENSE
+-- CITATION.cff
+-- pyproject.toml
|
+-- specs/
|   +-- VAL-001.json
|   +-- BENCH-001.json
|   +-- BENCH-002.json
|
+-- src/
|   +-- memory_attention_lab/
|       +-- attention/
|       |   +-- standard.py
|       |   +-- memory.py
|       |   +-- recall.py
|       |
|       +-- measurement/
|       |   +-- latency.py
|       |   +-- residency.py
|       |   +-- bandwidth.py
|       |
|       +-- experiments/
|       +-- spec.py
|       +-- results.py
|       +-- evidence.py
|       +-- provenance.py
|       +-- cli.py
|
+-- tests/
|
+-- evidence/
|
+-- docs/
|   +-- SCOPE.md
|   +-- THEORY.md
|   +-- CLAIM_MAP.md
|   +-- REFERENCES.md
|   +-- REPRODUCIBILITY.md
|   +-- MEASUREMENT.md
|   +-- VAL-001.md
|   +-- BENCH-001.md
|   +-- BENCH-002.md
|
+-- research/
|   +-- decisions/
|
+-- .github/
    +-- workflows/
```

This is a **shape, not a command to create empty directories**.

A component is added only when an actual experiment requires it.

## Architecture principle

The project prefers:

```text
functional research core
+
thin imperative shell
```

Attention construction, reconstruction logic, measurements, and acceptance checks should remain inspectable.

Filesystem, CLI, Git metadata, plotting, and environment capture should stay outside the mathematical core where possible.

## Relationship to the upstream implementation

The public Memory Attention implementation is treated as:

```text
SOURCE
+
REFERENCE IMPLEMENTATION
```

It is **not** the architecture authority for this repository.

This project should maintain a small independent reference implementation first, then compare behavior against upstream code where useful.

That separation helps detect shared assumptions instead of merely reproducing the same implementation path twice.

## Claim map

The project will maintain a human-readable claim map:

```text
SOURCE CLAIM
     |
     v
OUR INTERPRETATION
     |
     v
TESTABLE CONDITION
     |
     v
EXPERIMENT
     |
     v
OBSERVATION
     |
     v
ASSESSMENT
```

This makes it possible to say precisely which part of a paper claim was tested, which part was not, and what evidence supports the local assessment.

## Roadmap

```text
VAL-001
Memory Attention algebra + reconstruction equivalence
        |
        v
BENCH-001
accelerator / CPU residency baseline
        |
        v
BENCH-002
value reconstruction / MA-Recall
        |
        v
EXP-001
controlled small-model comparison
        |
        v
later questions
quantized memory / storage-backed memory / broader architectures
```

Later roadmap nodes describe intent, not completed results.

Each stage must earn its way into the repository through an explicit experiment contract.

## References

Primary starting points:

- Jiale Kang, **Memory Attention**, arXiv:2609.28399.
- Official public implementation: [Joluck/memory-attention](https://github.com/Joluck/memory-attention).
- Flash Linear Attention: [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention).

A dedicated literature and claim map will track exact source-to-experiment lineage as the project grows.

## License

A repository license will be added explicitly before external code is incorporated.

Upstream code must not be copied into this repository merely for convenience. Any reused code must retain its applicable license and attribution.

## Project principle

> **Do not ask only whether a model can remember more. Ask which state truly needs to stay expensive.**

The objective of this repository is to make Memory Attention and related value-side state trade-offs small enough to inspect, falsify, reproduce, and measure on ordinary hardware.
