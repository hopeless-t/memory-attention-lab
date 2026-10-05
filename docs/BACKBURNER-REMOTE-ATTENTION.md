# Backburner remote-attention control

**Status:** research intake / non-canonical  
**Date:** 2026-10-05

## Relevance

Memory Attention Lab separates model capacity, accelerator residency, persistent inference state, and reconstructability. Backburner provides concrete systems prior art for a nearby question: old KV pages can live on another device, and the computation that consumes them can move with those pages.

References:

- https://github.com/StayLameBro/backburner
- https://github.com/Niko1221/Strata
- https://pc.watch.impress.co.jp/docs/news/2145651.html

## Mechanism to preserve as a control

Backburner changes the helper device's role with context pressure:

```text
lower context:
  host early layers -> helper tail layers

higher context:
  host all layers
  helper stores old KV pages
  host sends Q
  helper computes attention over remote old keys
  host merges partial (O, max, sum)
```

That is useful here because it cleanly separates several questions that are often collapsed into "KV offload":

1. **remote residency only** — state is remote but compute stays local;
2. **remote state + remote consuming compute** — avoid pulling the full state back;
3. **reconstruction** — store less state and rebuild values when needed;
4. **compression/quantization** — retain state in a smaller representation;
5. **eviction** — discard state and accept a semantic limit.

## Proposed benchmark extension

Add a future systems control alongside BENCH-002:

### RA-001 — Remote Attention Partition

Compare, under the same logical attention state:

```text
A. local KV + local attention
B. remote KV + transfer-back + local attention
C. remote KV + remote partial attention + merge
D. reduced persistent state + value reconstruction
```

Measure separately:

- accelerator-resident bytes;
- host-resident bytes;
- remote-resident bytes;
- bytes crossing the link per token;
- query/result bytes versus full-KV transfer bytes;
- prefill latency;
- decode latency;
- synchronization overhead;
- reconstruction overhead;
- output equivalence under the declared precision contract.

## Key research question

> When is it cheaper to move **the query and reduction result** to/from stored state than to move the stored state itself?

This is directly relevant to the lab's broader distinction between persistent state and reconstructability.

A useful decision surface is:

```text
remote-partition beneficial iff
  transfer(Q + reduced_result)
  + remote_compute
  + merge
  < transfer(KV)
    + local_compute
```

The real experiment should retain the terms separately rather than trusting this simplified inequality.

## Strata connection

Strata provides the complementary prior-art axis: application-directed VRAM/RAM/SSD placement and CPU/GPU overlap. Together the projects suggest a broader systems taxonomy:

```text
where state lives
x
where consuming compute runs
x
whether state is retained, compressed, or reconstructed
```

That three-axis taxonomy is more useful to Memory Attention Lab than a binary "offloaded/not offloaded" label.

## Non-claims

This note does not claim that remote attention is faster in general, that Backburner's Apple-specific implementation transfers to CUDA/ROCm systems, or that Memory Attention should adopt remote execution. It records an experimentally separable control that can prevent future benchmark conclusions from conflating residency, transfer, and reconstruction.