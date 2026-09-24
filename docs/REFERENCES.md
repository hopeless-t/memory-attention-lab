# References

> **Status:** STAGING RESEARCH  
> This file is a source registry, not a claim that every source has been reproduced.

Primary sources are preferred. Secondary explanations may be used for navigation but should not become the sole authority for a scientific claim.

## Primary subject

### [MA-2026] Memory Attention

Jiale Kang. *Memory Attention*. 2026.

- arXiv: https://arxiv.org/abs/2609.28399
- Public implementation: https://github.com/Joluck/memory-attention

**Repository role**

~~~text
PRIMARY SUBJECT
SOURCE CLAIMS
REFERENCE IMPLEMENTATION
~~~

---

## Token-indexed / lookup capacity and nearby mechanisms

### [ENGRAM-2026] Conditional Memory via Scalable Lookup

Xin Cheng et al. *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models*. 2026.

- arXiv: https://arxiv.org/abs/2601.07372

**Repository role**

~~~text
LOOKUP CAPACITY
DETERMINISTIC ADDRESSING
HOST PREFETCH PRECEDENT
EXPERIMENTAL CONTROL
~~~

### [STEM-2026] STEM

Ranajoy Sadhukhan et al. *STEM: Scaling Transformers with Embedding Modules*. 2026.

- arXiv: https://arxiv.org/abs/2601.10639
- Project page: https://infini-ai-lab.github.io/STEM/

**Repository role**

~~~text
TOKEN-INDEXED LAYER-LOCAL CAPACITY
DENSE-COMPUTE REPLACEMENT PRECEDENT
~~~

### [MOVE-2026] MoVE

Yangyan Li. *MoVE: Mixture of Value Embeddings — A New Axis for Scaling Parametric Memory in Autoregressive Models*. 2026.

- arXiv: https://arxiv.org/abs/2601.22887

**Repository role**

~~~text
VALUE-MEMORY SIBLING
EXPERIMENTAL CONTROL CANDIDATE
~~~

### [PLE-2025] Gemma 3n Per-Layer Embeddings

Google. *Introducing Gemma 3n: The developer guide*. 2025.

- Developer guide: https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/
- Preview announcement: https://developers.googleblog.com/en/introducing-gemma-3n/

**Repository role**

~~~text
TOKEN-INDEXED / LAYER-LOCAL CAPACITY
CPU RESIDENCY PRECEDENT
TOTAL-PARAMETER VS ACCELERATOR-RESIDENCY PRECEDENT
~~~

### [DEEPEMBED-2025] RWKV-8 DeepEmbed

RWKV project documentation. *RWKV Architecture / RWKV-8 DeepEmbed*.

- Documentation: https://github.com/RWKV/RWKV-wiki/blob/main/docs/basic/architecture.md

**Repository role**

~~~text
TOKEN-INDEXED LAYER CAPACITY
RAM / SSD RESIDENCY
PREFETCH PRECEDENT
~~~

### [VALUE-RESIDUAL-2024] Value Residual Learning

Zhanchao Zhou, Tianyi Wu, Zhiyun Jiang, Zhenzhong Lan. *Value Residual Learning For Alleviating Attention Concentration In Transformers*. 2024.

- arXiv: https://arxiv.org/abs/2410.17897

**Repository role**

~~~text
VALUE-STREAM REUSE CONTEXT
HISTORICAL PRECEDENT
~~~

### [TOKEN-VALUE-EMBED-2024] Layerwise Token Value Embeddings

The Memory Attention paper cites:

- KoszarskyB. *Layerwise token value embeddings*. X post, 2024.
- The citation describes the announcement accompanying the modded-nanogpt implementation.

The implementation lineage is anchored here to the modded-nanogpt 2024-12-04
Value Embeddings record:

- Repository: https://github.com/KellerJordan/modded-nanogpt
- Record path: records/track_1_short/2024-12-04_ValueEmbed/train_gpt2.py
- Inspected repository commit: bc3a0c2d640d0d73dedaef87eae26148d2e32afb
- Inspected file blob: c3e21231926be6904e79720ffb19895c5493ed1c

The project history lists the 2024-12-04 record as "Value Embeddings" and
credits @KoszarskyB.

The inspected implementation uses a layer-indexed token embedding and a learned
scalar mix with the ordinary value projection:

~~~text
V_hist = (1 - lambda) * (X Wv) + lambda * E_layer[token]
~~~

The inspected record initializes lambda to 0.5.

**Repository role**

~~~text
DIRECTLY CITED PRIOR MECHANISM
HISTORICAL REPRODUCTION LANE
NOT THE SOLE CAUSAL CONTROL
~~~

Later modded-nanogpt value-embedding variants are separate historical revisions
and should not be silently substituted for the frozen 2024-12-04 anchor.

---

## Large parametric memory

### [PERSISTENT-2019] Augmenting Self-Attention with Persistent Memory

Sainbayar Sukhbaatar et al. *Augmenting Self-Attention with Persistent Memory*. 2019.

- arXiv: https://arxiv.org/abs/1907.01470

**Repository role**

~~~text
TRANSFORMER-ERA PERSISTENT MEMORY PRECEDENT
~~~

### [PKM-2019] Large Memory Layers with Product Keys

Guillaume Lample, Alexandre Sablayrolles, Marc'Aurelio Ranzato, Ludovic Denoyer, Hervé Jégou. *Large Memory Layers with Product Keys*. 2019.

- arXiv: https://arxiv.org/abs/1907.05242

**Repository role**

~~~text
PARAMETER CAPACITY VS DENSE COMPUTE PRECEDENT
LARGE LOOKUP MEMORY
~~~

### [MEMORY-LAYERS-2024] Memory Layers at Scale

Vincent-Pierre Berges et al. *Memory Layers at Scale*. 2024.

- arXiv: https://arxiv.org/abs/2412.09764

**Repository role**

~~~text
LARGE-SCALE PARAMETRIC MEMORY
COMPUTE/PARAMETER MATCHING PRECEDENT
~~~

---

## KV-cache and persistent-state redesign

### [MQA-2019] Multi-Query Attention

Noam Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need*. 2019.

- arXiv: https://arxiv.org/abs/1911.02150

**Repository role**

~~~text
KV HEAD SHARING
INCREMENTAL-DECODE MEMORY-BANDWIDTH PRECEDENT
~~~

### [CLA-2024] Cross-Layer Attention

William Brandon et al. *Reducing Transformer Key-Value Cache Size with Cross-Layer Attention*. 2024.

- arXiv: https://arxiv.org/abs/2405.12981

**Repository role**

~~~text
CROSS-LAYER KV SHARING
PERSISTENT-CACHE REDUCTION PRECEDENT
~~~

### [MLA-2024] DeepSeek-V2 / Multi-head Latent Attention

DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*. 2024.

- arXiv: https://arxiv.org/abs/2405.04434

**Repository role**

~~~text
LATENT KV REPRESENTATION
PERSISTENT-CACHE COMPRESSION PRECEDENT
~~~

---

## Earlier addressable-memory context

### [NTM-2014] Neural Turing Machines

Alex Graves, Greg Wayne, Ivo Danihelka. *Neural Turing Machines*. 2014.

- arXiv: https://arxiv.org/abs/1410.5401

**Repository role**

~~~text
ADDRESSABLE EXTERNAL MEMORY PRECEDENT
~~~

### [MEMN2N-2015] End-To-End Memory Networks

Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus. *End-To-End Memory Networks*. 2015.

- arXiv: https://arxiv.org/abs/1503.08895

**Repository role**

~~~text
END-TO-END ATTENTIONAL MEMORY PRECEDENT
~~~

### [KVMEM-2016] Key-Value Memory Networks

Alexander Miller et al. *Key-Value Memory Networks for Directly Reading Documents*. 2016.

- arXiv: https://arxiv.org/abs/1606.03126
- Meta research page: https://ai.meta.com/research/publications/key-value-memory-networks-for-directly-reading-documents/

**Repository role**

~~~text
ADDRESS REPRESENTATION VS OUTPUT REPRESENTATION PRECEDENT
~~~

---

## Transformer foundation

### [TRANSFORMER-2017] Attention Is All You Need

Ashish Vaswani et al. *Attention Is All You Need*. 2017.

- arXiv: https://arxiv.org/abs/1706.03762

**Repository role**

~~~text
STANDARD Q/K/V REFERENCE
~~~

---

## Source-handling rules

For every claim copied into CLAIM_MAP.md:

~~~text
1. identify the primary source;
2. state what the source actually reports or proposes;
3. separate source fact from local interpretation;
4. record whether the relationship is explicit citation, historical precedent,
   mechanistic similarity, local inference, or experimental control;
5. do not upgrade a secondary summary into primary authority;
6. do not infer direct ancestry from chronology or similarity;
7. pin implementation commits when executable reproduction begins.
~~~

## Open source-resolution tasks

~~~text
REF-RQ-001 — RESOLVED FOR INITIAL VAL-002 DESIGN
The Memory Attention citation is KoszarskyB's 2024 X-post announcement.
The executable implementation anchor is the modded-nanogpt 2024-12-04
Value Embeddings record pinned above.

REF-RQ-002
When executable comparisons begin, pin exact upstream commits for:
    Memory Attention
    any Value-Embedding control
    DeepEmbed if used
    Engram if used

REF-RQ-003
Record licenses before incorporating any upstream code.
~~~
