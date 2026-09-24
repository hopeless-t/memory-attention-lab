# VAL-001 canonical reference evidence

> **Status:** REVIEWED REFERENCE CANDIDATE  
> **Scientific authority:** VAL-001 numerical contract only  
> **Performance authority:** NONE  
> **Training authority:** NONE

This directory preserves the small structured evidence generated from the
merged VAL-001 source commit.

## Generation provenance

~~~text
source commit:
f941844c985a0857dc93b6cc321fa40427856f08

GitHub Actions run:
36074243522

workflow:
.github/workflows/val001.yml

artifact:
val001-reference-evidence

artifact id:
10839104066

artifact digest:
sha256:e24a4b43346a3263110c9f98f60d71fade951c9ecf7ce19be4b5290d993eae47

spec SHA-256:
af5b18a75336604f1f0121a1c2b86b8fa8b72af412f2fb5b91b850606c83b99e
~~~

The JSON files in this directory are copied from that generated artifact
without changing their numerical content.

The later commit that stores these files is intentionally **not** written into
the generated manifest. The manifest identifies the source code that performed
the calculation, avoiding circular provenance.

## Reviewed result

~~~text
VAL-001: PASS

known-answer normalized memory:
PASS / max abs error 0

known-answer constructed values:
PASS / max abs error 0

known-answer rotated keys:
PASS / max abs error 0

folded-memory equivalence:
PASS / max abs error 0

RoPE roundtrip:
PASS / max abs error 2.220446049250313e-16

pre-RoPE value guard:
PASS
wrong post-RoPE construction delta = 1.1667631545428359
required minimum delta = 0.1
~~~

The source commit also passed the validation test suite on:

~~~text
Ubuntu 24.04 / Python 3.12
Ubuntu 24.04 / Python 3.13
macOS 15 / Python 3.12
macOS 15 / Python 3.13
~~~

with 11 tests per matrix lane.

## Authority boundary

This reference supports only the frozen VAL-001 numerical contract.

It does **not** establish:

- language-model quality;
- training equivalence;
- FlashAttention or Triton correctness;
- latency improvement;
- CPU-offload benefit;
- MA-Recall deployment benefit;
- a 50% end-to-end memory reduction;
- Memory Attention architectural superiority.

A canonical reference is a reproducible comparison anchor, not a universal
scientific verdict.
