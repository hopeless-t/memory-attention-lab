# VAL-002 canonical reference evidence

> **Status:** REVIEWED REFERENCE CANDIDATE  
> **Scientific authority:** VAL-002 value-path decomposition only  
> **Quality authority:** NONE  
> **Performance authority:** NONE

This directory preserves the small structured evidence generated from the
merged VAL-002 source commit.

## Generation provenance

~~~text
source commit:
6fa2003032285de3e01d9930b530ac13baa761f2

design commit:
283927518dae597aec51746b2b2cf08e8b66f166

GitHub Actions run:
36076383535

workflow:
.github/workflows/val002.yml

artifact:
val002-reference-evidence

artifact id:
10840460671

artifact digest:
sha256:877de02288f44cb77b7a9ba5333c150344473c5b1bf3a0f86ddd6ac4ad0af8c3

VAL-002 executable spec SHA-256:
87d2332bcfb907821317351d63a6d23eed545c7fce30ce1d3e2a0301a142deb5

parent VAL-001 spec SHA-256:
af5b18a75336604f1f0121a1c2b86b8fa8b72af412f2fb5b91b850606c83b99e
~~~

The generated JSON files in this directory are copied from that main-branch
artifact without changing their bytes.

Their Git blob identities were checked against the artifact bytes before
publication:

~~~text
manifest.json
5fa382cf93c9fd2cda79d586fcf6623486de4a1d

metrics.json
ed18d3a997d13a528dc6687bb40274d5e9ec18f6

checks.json
7adfbfaf564532d2faf0c12279eaf22d380a4bb4

observations.json
f1bb72da0c1c14ec049654a67444df8e9123dee6
~~~

The publication commit is intentionally not written into the generated
manifest. The manifest identifies the source commit that performed the
calculation, avoiding circular provenance.

## Reviewed result

~~~text
VAL-002: PASS

known-answer:
    memory   error 0
    C00      error 0
    C01      error 0
    C10      error 0
    C11      error 0
    HIST-VE  error 0
    rotated contextual Wk error 0

parent VAL-001 anchors:
    C10 content K  error 0
    memory         error 0
    C11 value      error 0
    rotated K      error 0
    fixture identity PASS

factorial fixed-input interaction:
    max abs 2.220446049250313e-16

historical Value Embeddings vs local additive C01:
    max abs delta 3.5784200537048947
    required minimum 0.1

wrong post-RoPE value construction:
    max abs delta 1.1667631545428359
    required minimum 0.1

Wv perturbation guard:
    PASS

Wk perturbation guard:
    PASS
~~~

The source commit also passed the full test suite on:

~~~text
Ubuntu 24.04 / Python 3.12
Ubuntu 24.04 / Python 3.13
macOS 15 / Python 3.12
macOS 15 / Python 3.13
~~~

with 21 tests per matrix lane.

## What this evidence establishes

Within the frozen numerical fixture, the evidence is consistent with:

1. the declared C00/C01/C10/C11 value-path construction;
2. the use of exactly the same Memory Attention memory contribution in C01 and
   C11;
3. C11 reproducing the already-frozen VAL-001 Memory Attention construction;
4. the historical learned-lambda Value Embeddings lane being a different
   construction from the local additive C01 control;
5. the Wv and Wk source lanes responding to their own projection perturbations;
6. use of pre-RoPE contextual K rather than positional/scoring K for C11.

## What this evidence does not establish

It does not establish:

- model-quality improvement;
- training-dynamics equivalence;
- parameter-matched fairness;
- a generic advantage from removing Wv;
- a K-specific reuse advantage in trained models;
- latency improvement;
- accelerator-memory reduction;
- offload efficiency;
- MA-Recall benefit;
- Memory Attention superiority.

The factorial zero interaction is a fixed-input algebraic result for this
additive construction. It must not be generalized to training dynamics.
