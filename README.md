# memory-attention-lab

Independent reproducibility and systems research for **Memory Attention**, with a focus on value construction, accelerator residency, offloading, and reconstructable attention state.

> **Status: STAGING RESEARCH**
>
> **NOT CANONICAL · NO IMPLEMENTATION AUTHORITY**
>
> This repository is an independent research project. It is not the official Memory Attention implementation, and experimental findings here do not by themselves establish architectural superiority or imply adoption by any other project.

## Core question

> **How much value-side state in attention must actually remain resident on the accelerator?**

This repository begins from Memory Attention, but the research question is broader than one implementation.

The working thesis is:

> **Model capacity, active computation, physical residency, and persistent inference state are separate resources and should be measured separately.**

This is a research hypothesis and design principle, not an established result.

## Current status

~~~text
Stage 0 literature / claim lineage
    COMPLETE

VAL-001 Memory Attention algebra
    FROZEN REFERENCE — PASS

VAL-002 value-source decomposition
    NEXT

BENCH-001 residency baseline
    NOT STARTED

BENCH-002 value reconstruction / MA-Recall
    NOT STARTED

EXP-001 controlled small-model comparison
    NOT STARTED
~~~

The reviewed VAL-001 reference evidence is stored under
`evidence/VAL-001/reference/`. Its PASS status validates only the frozen
numerical contract and does not establish quality or performance superiority.

Memory Attention provides a concrete place to test the broader thesis because it changes both how attention values are constructed and where part of the model state may live.

## Why this project exists

Standard self-attention constructs queries, keys, and values from contextual hidden states:

~~~text
Q = X Wq
K = X Wk
V = X Wv
~~~

Memory Attention replaces the dedicated value projection with a contextual key contribution plus layer-specific token memory:

~~~text
Q  = q_proj(X)
K0 = k_proj(X)
M  = memory[token_id]

V  = K0 + Norm(M)
~~~

At inference, the normalization can be folded into the memory table, reducing online value construction to lookup plus addition.

The important change is not merely that a token embedding is added to the value path.

Memory Attention removes the independent value projection and reuses the key-side contextual representation as part of the value.

That creates several independently testable questions:

~~~text
Does token-indexed memory help because it adds capacity?

Does replacing Wv matter independently of added capacity?

Does reusing K as contextual value content matter?

Can token-indexed parameters live outside accelerator memory?

When is reconstructing value state cheaper than caching it?
~~~

The repository exists to separate these questions rather than collapsing them into one benchmark result.

## Upstream context and lineage

Memory Attention did not appear in isolation.

The Memory Attention paper itself explicitly places its question alongside recent lookup-based approaches including:

~~~text
Layerwise Token Value Embeddings
DeepEmbed
Per-Layer Embeddings (PLE)
STEM
Engram
~~~

These approaches differ in mechanism, but they share a useful systems idea:

> learned capacity can sometimes be addressed by token identity or token-derived addresses rather than activated through proportional dense computation.

Memory Attention asks a complementary question: can explicit memory **replace** an existing computation rather than only supplement it?

### Directly relevant nearby mechanisms

The initial literature map distinguishes several nearby mechanisms instead of treating them as equivalent:

~~~text
Value Embedding
    contextual V + token-specific memory
    original value projection remains

Per-Layer Embeddings / DeepEmbed
    layer-local token-indexed parameters
    designed so substantial parameter capacity need not remain on the accelerator

STEM
    token-indexed embedding lookup replaces part of dense FFN computation

Engram
    deterministic token / n-gram-derived addressing
    lookup can be prefetched from host memory

Memory Attention
    contextual K + token-specific memory
    dedicated value projection removed
~~~

This is **context**, not a claim that every item above directly caused Memory Attention.

Where the Memory Attention paper explicitly cites a mechanism, that relationship will be recorded as a source citation.

Broader historical relationships will be labeled as this repository's interpretation.

### Broader historical context

This lab will also track older or adjacent research lines that help frame the same resource trade-offs:

~~~text
persistent learned memory
product-key / large memory layers
attention as key-value retrieval
KV-cache sharing and compression
shared latent K/V representations
~~~

Examples include Persistent Memory, Product-Key Memory and later Memory Layers, Cross-Layer Attention, and Multi-head Latent Attention.

These are not automatically implementation targets.

Their purpose is to provide controls, terminology, and prior-art boundaries.

### Lineage rule

The literature map must distinguish:

~~~text
EXPLICITLY CITED RELATIONSHIP
HISTORICAL PRECEDENT
MECHANISTIC SIMILARITY
OUR INFERENCE
EXPERIMENTAL CONTROL
~~~

Similarity is not evidence of direct influence.

Chronology is not causality.

A nearby architecture is not automatically an ancestor.

## Scope Freeze

The repository uses the following scope boundary.

~~~text
PRIMARY SUBJECT
    Memory Attention

PRIMARY RESEARCH QUESTIONS
    value construction
    parameter capacity
    active computation
    accelerator residency
    memory traffic
    persistent KV state
    reconstructability

LITERATURE SCOPE
    broad enough to recover ancestry, nearby mechanisms,
    and relevant systems controls

IMPLEMENTATION SCOPE
    deliberately narrow
    only implement a comparison when an experiment requires it
~~~

This repository is **not** intended to become a general catalogue of every neural-memory architecture.

The literature may be broad.

The executable experiment set should remain small.

## Scientific and engineering boundary

This repository does **not** currently claim to:

- reproduce all results from the Memory Attention paper;
- establish that Memory Attention is better than standard attention;
- establish that token-indexed memory is universally better than dense parameters;
- establish a universal latency advantage;
- establish a universal memory advantage;
- show that CPU or storage offload is always beneficial;
- prove that reconstructed values are cheaper than cached values in deployment;
- infer model-quality gains from parameter-count increases alone;
- establish direct historical influence where only mechanistic similarity is known;
- provide production-ready serving infrastructure.

The initial work deliberately starts below those claims.

A paper claim is not a reproduced result.

A successful program execution is not a successful research result.

A benchmark being faster once is not evidence of a general speed advantage.

A larger model winning is not evidence that its architecture is better.

## Research decomposition

The project decomposes the design space into six independent axes.

### A. Address

~~~text
hidden-state-dependent address
vs
token-ID / token-derived deterministic address
~~~

### B. Content

~~~text
fully contextual content
vs
static learned memory
vs
a combination of both
~~~

### C. Fusion

~~~text
replace
augment
gate
mix
~~~

### D. Residency

~~~text
accelerator
host RAM
memory-mapped storage
other storage tiers
~~~

### E. Persistence

~~~text
store state now
vs
reconstruct state later
~~~

### F. Budget

~~~text
total parameters
active parameters
FLOPs
accelerator-resident bytes
host-resident bytes
memory traffic
persistent cache bytes
training tokens
~~~

Two systems should not be described as "matched" without saying which of these budgets are actually matched.

## Research pipeline

The repository follows an explicit evidence path:

~~~text
Exploration
   |
   v
Source / prior-art map
   |
   v
Atomic decomposition
   |
   v
Research question
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
~~~

Difficult design choices may use explicit decision-support analysis, including sensitivity analysis or Monte Carlo simulation.

Such calculations are engineering decision support.

They are not physical probabilities and are not scientific evidence for the architecture being studied.

## Execution states

Every executable experiment or benchmark must end in one of four execution states:

~~~text
PASS
    The run executed and every validity check required by the experiment
    contract passed.

FAIL
    The run executed, but one or more declared validity checks failed.

INVALID
    The specification was rejected before valid execution began.

ERROR
    Runtime, software, hardware, or I/O failure prevented valid evaluation.
~~~

A PASS does **not** mean that a research claim was confirmed.

It means only that the experiment was validly executed under its declared contract.

## Claim-assessment states

Research claims use a separate vocabulary:

~~~text
NOT_TESTED
CONSISTENT
INCONSISTENT
INCONCLUSIVE
~~~

For example:

~~~text
BENCH-001: PASS
claim:     INCONCLUSIVE
~~~

is a normal and valid outcome.

This separation is intentional.

## Initial research sequence

The harness must earn trust before larger model experiments begin.

### Stage 0 — Literature and claim lineage

Before implementing comparisons, record:

- the Memory Attention paper's own stated prior context;
- older and adjacent mechanisms relevant to the same resource trade-offs;
- explicit source relationships versus local interpretations;
- which claims are actually testable on ordinary hardware;
- which comparison would distinguish competing explanations.

The detailed lineage belongs in docs/LITERATURE_MAP.md and docs/CLAIM_MAP.md, not in an ever-growing README.

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

### VAL-002 — Value-Source Decomposition

Separate the architectural ingredients that are otherwise easy to confound.

The first controlled family is:

~~~text
Standard
    V = X Wv

Value-Embedding control
    V = X Wv + E[token]

Memory Attention
    V = X Wk + E[token]
~~~

Exact normalization, scaling, and dimensional contracts will be frozen in the experiment specification.

This validation exists to distinguish at least three possible causes of an observed result:

~~~text
additional token-indexed capacity
removal of Wv
reuse of K as contextual value content
~~~

Additional nearby mechanisms such as MoVE-like mixing are candidates only if the first comparison leaves an unresolved question.

They are not required initial implementations.

### BENCH-001 — Residency Baseline

Compare at least:

~~~text
standard attention
Memory Attention / accelerator-resident memory
Memory Attention / host-resident memory
~~~

Candidate measurements include:

~~~text
accelerator parameter bytes
host parameter bytes
peak accelerator memory
prefill latency
decode latency
tokens / second
host -> device bytes
~~~

Benchmark validity and benchmark outcome remain separate.

A slower configuration can still produce a valid PASS benchmark.

### BENCH-002 — Value Reconstruction / MA-Recall

The Memory Attention paper describes MA-Recall as a possible extension that trades repeated retrieval and reconstruction for reduced persistent value-cache storage.

The current public reference implementation still uses an ordinary cache path containing both keys and values.

This repository will therefore treat value reconstruction as a separate experiment, not as an already-established property of deployed Memory Attention.

Conceptually:

~~~text
ordinary cache

K K K K K
V V V V V
~~~

versus:

~~~text
reconstruction-oriented cache

retained reconstruction state
token ids

    |
    v
reconstruct V when required
~~~

Before benchmarking this path, VAL-001 must define exactly which key representation is retained and demonstrate semantic equivalence under a declared numerical contract.

Candidate measurements include:

~~~text
persistent cache bytes
reconstruction latency
memory-table traffic
decode latency
peak accelerator memory
host-memory pressure
~~~

### EXP-001 — Controlled Small-Model Comparison

Training experiments are deferred until the validation and benchmark harnesses are trustworthy.

The first model-quality experiments should separate fairness constraints rather than collapse them into one vague notion of "matched models."

Candidate comparison axes include:

~~~text
matched training tokens
matched total parameters
matched active parameters
matched FLOPs
matched accelerator-resident parameters
matched accelerator memory
~~~

These are different experiments.

They should not be silently treated as interchangeable.

## Evidence model

A canonical research result should be reconstructable from small, structured artifacts.

Expected evidence objects include:

~~~text
manifest.json
metrics.json
checks.json
samples.csv
~~~

A manifest should record enough provenance to identify the actual calculation, including where relevant:

~~~text
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
~~~

Large profiling outputs are not canonical Git evidence by default.

Examples of non-canonical artifacts:

~~~text
Chrome traces
Nsight traces
large profiler captures
model checkpoints
dataset caches
temporary compiled objects
~~~

These may be retained locally or published separately when an experiment requires them.

## Reproducibility principles

This project begins with several fixed rules:

1. **Explore before implementing.** Recover relevant prior art and competing explanations before freezing an experiment.
2. **Decompose before comparing.** Separate mechanisms that could otherwise produce the same observed improvement.
3. **Spec before result.** The intended experiment must be machine-readable before its output becomes evidence.
4. **Fail closed.** Unknown or invalid experiment fields should be rejected rather than silently ignored.
5. **Known-answer first.** A benchmark harness must first demonstrate that it can recognize correct and deliberately broken behavior.
6. **Figure != evidence.** Plots are for human inspection unless an experiment contract explicitly says otherwise.
7. **Plan != result.** Roadmap entries and hypotheses are not findings.
8. **Source != reproduction.** Citing a paper or running its code does not automatically reproduce its claims.
9. **Similarity != ancestry.** Mechanistic resemblance must not be presented as documented historical influence.
10. **Performance needs protocol.** Warmup, synchronization, sample count, device state, dimensions, and precision must be reported with timing results.
11. **Match the budget explicitly.** "Fair comparison" must identify which resource budgets are controlled.
12. **No premature framework.** Shared abstractions are added only after multiple concrete experiments require them.

## Planned repository shape

The repository follows a small research-lab pattern rather than mirroring the full upstream implementation.

~~~text
memory-attention-lab/
|
+-- README.md
+-- LICENSE
+-- CITATION.cff
+-- pyproject.toml
|
+-- specs/
|   +-- VAL-001.json
|   +-- VAL-002.json
|   +-- BENCH-001.json
|   +-- BENCH-002.json
|
+-- src/
|   +-- memory_attention_lab/
|       +-- attention/
|       |   +-- standard.py
|       |   +-- value_embedding.py
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
|   +-- LITERATURE_MAP.md
|   +-- CLAIM_MAP.md
|   +-- REFERENCES.md
|   +-- REPRODUCIBILITY.md
|   +-- MEASUREMENT.md
|   +-- VAL-001.md
|   +-- VAL-002.md
|   +-- BENCH-001.md
|   +-- BENCH-002.md
|
+-- research/
|   +-- decisions/
|
+-- .github/
    +-- workflows/
~~~

This is a **shape, not a command to create empty directories**.

A component is added only when an actual experiment requires it.

## Architecture principle

The project prefers:

~~~text
functional research core
+
thin imperative shell
~~~

Attention construction, reconstruction logic, measurements, and acceptance checks should remain inspectable.

Filesystem, CLI, Git metadata, plotting, and environment capture should stay outside the mathematical core where possible.

## Relationship to upstream implementations

The public Memory Attention implementation is treated as:

~~~text
SOURCE
+
REFERENCE IMPLEMENTATION
~~~

It is **not** the architecture authority for this repository.

This project should maintain a small independent reference implementation first, then compare behavior against upstream code where useful.

Other upstream or adjacent implementations are treated as references only when a specific experiment requires them.

The default is **not** to vendor large upstream repositories.

That separation helps detect shared assumptions instead of merely reproducing the same implementation path twice.

## Claim map

The project will maintain a human-readable claim map:

~~~text
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
~~~

This makes it possible to say precisely which part of a paper claim was tested, which part was not, and what evidence supports the local assessment.

## Roadmap

~~~text
literature / claim lineage
        |
        v
VAL-001
Memory Attention algebra + reconstruction contract
        |
        v
VAL-002
value-source decomposition
        |
        v
BENCH-001
accelerator / host residency baseline
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
quantized memory / storage-backed memory / additional controls
~~~

Later roadmap nodes describe intent, not completed results.

Each stage must earn its way into the repository through an explicit research question and, where executable, an experiment contract.

## Starting references

### Primary subject

- Jiale Kang, **Memory Attention**, arXiv:2609.28399.
- Official public implementation: [Joluck/memory-attention](https://github.com/Joluck/memory-attention).

### Lookup-based capacity and nearby mechanisms

- **Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models** (Engram), arXiv:2601.07372.
- **STEM: Scaling Transformers with Embedding Modules**, arXiv:2601.10639.
- Google DeepMind / Gemma Team, **Gemma 3n** documentation, including Per-Layer Embeddings.
- RWKV project documentation for **DeepEmbed**.
- Layerwise Token Value Embeddings, as referenced by the Memory Attention paper.
- **MoVE: Mixture of Value Embeddings**, arXiv:2601.22887, as a nearby value-memory comparison candidate.

### Broader historical and systems context

- **Augmenting Self-Attention with Persistent Memory**, arXiv:1907.01470.
- **Large Memory Layers with Product Keys**, arXiv:1907.05242.
- **Memory Layers at Scale**, arXiv:2412.09764.
- **DeepSeek-V2**, arXiv:2405.04434, for Multi-head Latent Attention.
- **Reducing Transformer Key-Value Cache Size with Cross-Layer Attention**, arXiv:2405.12981.
- **Value Residual Learning For Alleviating Attention Concentration In Transformers**, arXiv:2410.17897.

A dedicated literature map will record exact source-to-experiment lineage, dates, mechanisms, and authority boundaries.

## License

A repository license will be added explicitly before external code is incorporated.

Upstream code must not be copied into this repository merely for convenience.

Any reused code must retain its applicable license and attribution.

## Project principle

> **Do not assume that model capacity, active computation, physical residency, and persistent state must scale together. Measure them separately.**

The objective of this repository is to make Memory Attention and its nearby value-side state trade-offs small enough to inspect, falsify, reproduce, and measure on ordinary hardware.
