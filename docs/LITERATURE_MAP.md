# Literature Map

> **Status:** STAGING RESEARCH  
> **Scientific authority:** NONE  
> **Implementation authority:** NONE  
> **Purpose:** Recover the upstream context of Memory Attention and map prior mechanisms to testable questions without confusing chronology, similarity, or citation with direct ancestry.

## Mapping rule

This document uses five relationship types:

~~~text
EXPLICITLY_CITED
    The Memory Attention paper or another primary source explicitly cites or discusses the relationship.

HISTORICAL_PRECEDENT
    An older work contains a relevant mechanism or abstraction.

MECHANISTIC_SIMILARITY
    Two systems share a technical pattern, but direct influence is not established.

OUR_INFERENCE
    A local interpretation used to design experiments.

EXPERIMENTAL_CONTROL
    A mechanism is useful as a comparison even if it is not an ancestor.
~~~

Chronology is not causality.

Mechanistic similarity is not evidence of direct influence.

A paper being useful to this repository does not make it part of the Memory Attention authors' claimed lineage.

## Working question

The primary systems question is:

> **How much value-side state in attention must actually remain resident on the accelerator?**

The broader working thesis is:

> **Model capacity, active computation, physical residency, and persistent inference state are separable resources.**

This is a research hypothesis for the lab, not a result attributed to any source.

---

## Line A — Addressable neural memory before Transformers

### A1 — Neural Turing Machines

**SOURCE**

- Graves, Wayne, Danihelka, *Neural Turing Machines* (2014), arXiv:1410.5401.

**ESTABLISHED SOURCE CLAIM**

Neural networks can be coupled to an external memory and interact with it through differentiable attentional addressing.

**RELEVANCE**

This is a historical precedent for separating a compute network from an addressable memory resource.

**RELATIONSHIP**

~~~text
HISTORICAL_PRECEDENT
~~~

No direct ancestry claim to Memory Attention is made here.

### A2 — End-To-End Memory Networks

**SOURCE**

- Sukhbaatar, Szlam, Weston, Fergus, *End-To-End Memory Networks* (2015), arXiv:1503.08895.

**ESTABLISHED SOURCE CLAIM**

A recurrent attention process can read from a possibly large learned external memory and be trained end-to-end.

**RELEVANCE**

The paper reinforces the distinction between computation and a separately addressable learned memory.

**RELATIONSHIP**

~~~text
HISTORICAL_PRECEDENT
~~~

### A3 — Key-Value Memory Networks

**SOURCE**

- Miller et al., *Key-Value Memory Networks for Directly Reading Documents* (2016), arXiv:1606.03126.

**ESTABLISHED SOURCE CLAIM**

Different representations can be used for the addressing stage and the output/content stage of a memory read.

**RELEVANCE**

This is an important conceptual precedent for separating:

~~~text
address representation
from
retrieved content representation
~~~

It should not be confused with the exact K/V tensors of Transformer self-attention.

**RELATIONSHIP**

~~~text
HISTORICAL_PRECEDENT
MECHANISTIC_SIMILARITY
~~~

---

## Line B — Transformer key/value state and inference residency

### B1 — Transformer self-attention

**SOURCE**

- Vaswani et al., *Attention Is All You Need* (2017), arXiv:1706.03762.

**ESTABLISHED SOURCE STRUCTURE**

Standard self-attention uses separately projected queries, keys, and values:

~~~text
Q = X Wq
K = X Wk
V = X Wv
~~~

**RELEVANCE**

Memory Attention directly modifies the origin of the value tensor.

### B2 — Multi-Query Attention

**SOURCE**

- Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need* (2019), arXiv:1911.02150.

**ESTABLISHED SOURCE CLAIM**

Sharing keys and values across query heads can substantially reduce incremental-decoding memory bandwidth with limited quality loss.

**RELEVANCE**

MQA is part of the systems lineage asking whether all conventional KV state must be distinct and resident.

**RELATIONSHIP**

~~~text
HISTORICAL_PRECEDENT
MECHANISTIC_SIMILARITY
~~~

### B3 — Cross-Layer Attention

**SOURCE**

- Brandon et al., *Reducing Transformer Key-Value Cache Size with Cross-Layer Attention* (2024), arXiv:2405.12981.

**ESTABLISHED SOURCE CLAIM**

Sharing K/V state between adjacent layers can further reduce KV-cache size; the paper reports roughly another 2x reduction relative to its MQA comparison while maintaining similar accuracy in the reported experiments.

**RELEVANCE**

CLA reduces persistent state through sharing.

~~~text
CLA:
store fewer distinct K/V states

MA-Recall question:
can some value-side state be reconstructed rather than persistently stored?
~~~

These are different mechanisms targeting a related resource.

**RELATIONSHIP**

~~~text
MECHANISTIC_SIMILARITY
EXPERIMENTAL_CONTROL
~~~

### B4 — Multi-head Latent Attention

**SOURCE**

- DeepSeek-AI, *DeepSeek-V2* (2024), arXiv:2405.04434.

**ESTABLISHED SOURCE CLAIM**

Multi-head Latent Attention compresses KV information into a latent representation to reduce KV-cache cost.

**RELEVANCE**

MLA is another example of persistent inference state being redesigned instead of accepting the conventional K/V representation as fixed.

**RELATIONSHIP**

~~~text
MECHANISTIC_SIMILARITY
EXPERIMENTAL_CONTROL
~~~

---

## Line C — Parametric memory decoupled from dense compute

### C1 — Persistent Memory

**SOURCE**

- Sukhbaatar et al., *Augmenting Self-Attention with Persistent Memory* (2019), arXiv:1907.01470.

**ESTABLISHED SOURCE CLAIM**

Learned persistent memory vectors can be integrated with self-attention and used as a parameterized memory resource.

**RELEVANCE**

This is an early Transformer-era example of adding learned memory capacity outside ordinary per-token contextual projections.

**RELATIONSHIP**

~~~text
HISTORICAL_PRECEDENT
~~~

### C2 — Product-Key Memory

**SOURCE**

- Lample et al., *Large Memory Layers with Product Keys* (2019), arXiv:1907.05242.

**ESTABLISHED SOURCE CLAIM**

A large structured memory can add up to roughly a billion parameters with comparatively small computational overhead through product-key lookup.

**RELEVANCE**

It makes explicit the resource split:

~~~text
parameter capacity
!=
proportional dense compute
~~~

**RELATIONSHIP**

~~~text
HISTORICAL_PRECEDENT
MECHANISTIC_SIMILARITY
~~~

### C3 — Memory Layers at Scale

**SOURCE**

- Berges et al., *Memory Layers at Scale* (2024), arXiv:2412.09764.

**ESTABLISHED SOURCE CLAIM**

Trainable key-value memory layers can scale to very large memory capacity without increasing dense FLOPs proportionally; the paper reports experiments up to 128B memory parameters and 1T pretraining tokens.

**RELEVANCE**

This strengthens the case that total parameter count, activated computation, and memory access should be reported separately.

**RELATIONSHIP**

~~~text
HISTORICAL_PRECEDENT
MECHANISTIC_SIMILARITY
~~~

---

## Line D — Token-indexed layer-local capacity and offload

This line is the closest systems neighborhood to Memory Attention.

### D1 — Value Residual Learning

**SOURCE**

- Zhou et al., *Value Residual Learning For Alleviating Attention Concentration In Transformers* (2024), arXiv:2410.17897.

**ESTABLISHED SOURCE CLAIM**

Value information from earlier layers can be reused or shared across layers to change how value content propagates through deep Transformers.

**RELEVANCE**

This work is useful context for later value-stream reuse and value-embedding designs.

**RELATIONSHIP**

~~~text
HISTORICAL_PRECEDENT
~~~

Direct lineage to every later token-value-embedding implementation must be established source-by-source rather than assumed.

### D2 — Layerwise Token Value Embeddings

**SOURCES**

- Memory Attention cites KoszarskyB (2024), *Layerwise token value embeddings*.
- modded-nanogpt records **Value Embeddings** on 2024-12-04 and credits @KoszarskyB.
- The preserved 2024-12-04 implementation is frozen for VAL-002 source lineage.

**SOURCE-DOCUMENTED MECHANISM**

The preserved implementation keeps the ordinary projected value and mixes it
with a token-indexed embedding using a learnable scalar:

~~~text
V_hist = (1 - lambda) * Vproj + lambda * E[token]
~~~

with lambda initialized to 0.5 in that record.

**RELEVANCE**

This resolves an important distinction for VAL-002:

~~~text
historical/source-fidelity control
    learned mixture:
    (1-lambda) * Vproj + lambda * E[token]

local causal-additive control
    Vproj + M
~~~

These are **not** the same method.

The historical lane preserves prior-work fidelity.

The local additive lane exists only to isolate the effect of adding the same
Memory Attention memory contribution before replacing Vproj with Kcontent.

**RELATIONSHIP**

~~~text
EXPLICITLY_CITED
EXPERIMENTAL_CONTROL
~~~

### D3 — DeepEmbed

**SOURCE**

- RWKV project documentation for RWKV-8 / DeepEmbed.

**SOURCE-DOCUMENTED MECHANISM**

DeepEmbed learns high-dimensional token-indexed vectors inside every model layer. The documentation describes storing these vectors in RAM or SSD and prefetching the required vectors by token index.

**RELEVANCE**

DeepEmbed provides a concrete example of:

~~~text
total parameters
!=
accelerator-resident parameters
~~~

and of deterministic token addressing enabling staged memory residency.

**RELATIONSHIP**

~~~text
EXPLICITLY_CITED
MECHANISTIC_SIMILARITY
~~~

### D4 — Gemma 3n Per-Layer Embeddings

**SOURCE**

- Google, *Introducing Gemma 3n: The developer guide* (2025).

**ESTABLISHED SOURCE CLAIM**

Per-Layer Embeddings allow a significant portion of Gemma 3n parameters to be loaded/computed on CPU while only the core Transformer weights need to remain in accelerator memory.

Google describes E2B/E4B as having 5B/8B total parameters while requiring roughly 2B/4B core Transformer parameters resident on the accelerator.

**RELEVANCE**

PLE is a strong systems precedent for treating:

~~~text
total parameter capacity
accelerator residency
~~~

as different quantities.

**RELATIONSHIP**

~~~text
EXPLICITLY_CITED
MECHANISTIC_SIMILARITY
EXPERIMENTAL_CONTROL
~~~

### D5 — STEM

**SOURCE**

- Sadhukhan et al., *STEM: Scaling Transformers with Embedding Modules* (2026), arXiv:2601.10639.

**ESTABLISHED SOURCE CLAIM**

STEM replaces the FFN up-projection with a static layer-local token-indexed embedding lookup while retaining other dense FFN components.

**RELEVANCE**

STEM goes beyond adding memory capacity: lookup replaces an existing dense computation.

That makes it especially relevant to the Memory Attention question:

> Can explicit memory replace existing computation rather than only supplement it?

**RELATIONSHIP**

~~~text
EXPLICITLY_CITED
MECHANISTIC_SIMILARITY
~~~

### D6 — Engram

**SOURCE**

- Cheng et al., *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models* (2026), arXiv:2601.07372.

**ESTABLISHED SOURCE CLAIM**

Engram introduces deterministic token/N-gram-derived lookup as a complementary sparsity axis to conditional computation. The paper explicitly highlights that deterministic addressing enables prefetching from host memory.

**RELEVANCE**

This yields a key systems atom:

~~~text
address known before layer execution
        |
        v
prefetch can overlap with computation
~~~

**RELATIONSHIP**

~~~text
EXPLICITLY_CITED
MECHANISTIC_SIMILARITY
EXPERIMENTAL_CONTROL
~~~

### D7 — MoVE

**SOURCE**

- Li, *MoVE: Mixture of Value Embeddings — A New Axis for Scaling Parametric Memory in Autoregressive Models* (2026), arXiv:2601.22887.

**ESTABLISHED SOURCE CLAIM**

MoVE adds a learnable value-embedding bank and dynamically mixes retrieved memory into the standard value projection.

**RELEVANCE**

MoVE is a useful sibling/control because it keeps the standard value projection while augmenting value content with learned memory.

~~~text
MoVE-like family:
standard contextual V
+
memory

Memory Attention:
contextual K
+
memory
and no dedicated Wv
~~~

**RELATIONSHIP**

~~~text
MECHANISTIC_SIMILARITY
EXPERIMENTAL_CONTROL
~~~

MoVE is not required for the first implementation stage.

---

## Line E — Memory Attention

### E1 — Memory Attention

**SOURCE**

- Kang, *Memory Attention* (2026), arXiv:2609.28399.
- Public reference implementation: Joluck/memory-attention.

**CORE CONSTRUCTION**

~~~text
Q  = q_proj(X)
K0 = k_proj(X)
M  = memory[token_id]

V  = K0 + Norm(M)
~~~

The attention key is subsequently transformed by optional normalization and positional operations, so the repository must distinguish the pre-attention contextual key representation from the final key representation used by attention.

**SOURCE CLAIMS RELEVANT TO THIS LAB**

The paper reports or proposes that:

1. token-indexed memory can replace the dedicated value projection when contextual key information is retained;
2. normalization can be folded into memory at inference, reducing online value construction to lookup plus addition;
3. deterministic token-indexed memory enables CPU offload and prefetch;
4. experiments with additional memory parameters and matched training-token budgets improve language-modeling and average downstream results in the reported settings;
5. value state may potentially be reconstructed instead of persistently cached, creating an MA-Recall trade-off.

These claims must be tested separately.

**CURRENT IMPLEMENTATION OBSERVATION**

The public reference implementation currently updates its ordinary attention cache with both K and V state.

Therefore this repository treats MA-Recall as a separate proposed systems experiment, not as a demonstrated property of the current public serving path.

---

# Atomic mechanism map

The broad literature can be reduced to six independent design axes.

| Axis | Question | Example values |
| --- | --- | --- |
| Address | How is memory selected? | contextual similarity; token ID; N-gram-derived; learned routing |
| Content | What is retrieved? | contextual state; static learned vector; mixture |
| Fusion | How does memory enter computation? | augment; replace; gate; mix |
| Residency | Where do parameters/state live? | accelerator; host RAM; mmap/storage |
| Persistence | Must inference state be stored? | store; share; compress; reconstruct |
| Budget | Which resource is being held constant? | total params; active params; FLOPs; resident bytes; traffic; cache bytes; tokens |

This table is the main abstraction used to choose controls.

---

# Experiment impact

## VAL-001 — Memory Attention Algebra

Needed to establish the exact local meaning of:

~~~text
pre-RoPE K
attention K
memory vector
constructed V
cached state
reconstruction state
~~~

before any performance claim is tested.

## VAL-002 — Value-Source Decomposition

Initial controlled family:

~~~text
Standard
    V = X Wv

Value-Embedding control
    V = X Wv + E[token]

Memory Attention
    V = X Wk + E[token]
~~~

Purpose:

~~~text
separate additional token-indexed capacity
from removal of Wv
from reuse of K as contextual value content
~~~

## BENCH-001 — Residency Baseline

Tests whether moving token-indexed memory out of accelerator residency changes:

~~~text
accelerator bytes
host bytes
transfer bytes
prefill latency
decode latency
throughput
~~~

## BENCH-002 — Reconstruction / MA-Recall

Tests:

~~~text
persistent storage saved
vs
lookup / transfer / reconstruction cost added
~~~

Only after VAL-001 freezes the precise reconstruction contract.

## EXP-001 — Controlled Small-Model Training

Must separate multiple notions of fairness:

~~~text
matched training tokens
matched total parameters
matched active parameters
matched FLOPs
matched accelerator-resident parameters
matched accelerator bytes
~~~

No single one of these is automatically "the fair comparison."

---

# Hakken questions

## HAKKEN-01 — Which ingredient causes improvement?

If Memory Attention beats a standard model, is the difference caused by:

~~~text
extra token-indexed capacity
removal of Wv
K reuse
optimization differences
different total parameter count
different active parameter count
~~~

VAL-002 exists primarily to attack this ambiguity.

## HAKKEN-02 — When does predictable addressing become a systems primitive?

DeepEmbed, PLE, STEM, Engram, and Memory Attention suggest that deterministic or early-known addresses may permit a memory hierarchy:

~~~text
accelerator
    |
host RAM
    |
mmap / storage
~~~

The experiment must measure transfer and overlap rather than assume offload is free.

## HAKKEN-03 — Is accelerator residency a better deployment metric than total parameters?

A model may contain parameters that do not need continuous accelerator residency.

Candidate reporting tuple:

~~~text
(total_params,
 active_params_per_token,
 accelerator_resident_params,
 host_resident_params,
 persistent_cache_bytes)
~~~

## HAKKEN-04 — What exactly is reconstructable?

"V can be reconstructed" is not sufficiently precise.

The lab must determine which representation is required:

~~~text
pre-RoPE contextual K?
post-RoPE K?
token IDs?
layer ID?
normalization parameters?
precision state?
~~~

This question is a prerequisite for BENCH-002.

## HAKKEN-05 — When does memory traffic dominate removed compute?

Replacing a dense projection with lookup is not automatically faster.

Candidate crossover variables include:

~~~text
batch size
sequence length
vocabulary size
memory width
dtype
host bandwidth
PCIe bandwidth
cache locality
prefetch depth
accelerator generation
~~~

---

# Current boundary

As of Stage 0:

~~~text
Broad literature intake:
    YES

Direct-ancestry claims without explicit evidence:
    NO

Independent minimal Memory Attention implementation:
    NOT YET

Value-embedding control:
    PLANNED

CPU-residency benchmark:
    PLANNED

MA-Recall implementation:
    PLANNED AFTER RECONSTRUCTION CONTRACT

Training reproduction:
    DEFERRED UNTIL VALIDATION HARNESS IS TRUSTED
~~~

The next documents are CLAIM_MAP.md and REFERENCES.md.
