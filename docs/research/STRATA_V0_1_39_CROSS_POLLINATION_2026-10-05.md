# Strata v0.1.39 cross-pollination for memory-attention-lab — 2026-10-05

## Relevant upstream atoms

Strata v0.1.39 separates short, long and very-long prompt regimes instead of assuming one attention/prefill policy is optimal everywhere. It also keeps a byte-budgeted streamed working set rather than a fixed slot count.

## Research candidates

- Model attention/context as a **bounded active set** with explicit promotion, demotion and reuse distance.
- Measure regime knees instead of forcing a universal policy: short context, long context, very-long context.
- Budget cache/ring structures in bytes/tokens rather than entry counts when entry sizes differ.
- Treat prefill-like bulk ingestion and decode-like iterative recall as different phases with separate objectives.
- Record whether gains come from algorithm changes, resident-set changes or transfer overlap.

## Candidate state model

```text
MemoryItem {
  bytes
  token_span
  access_probability
  reuse_distance
  retrieval_cost
  eviction_cost
  semantic_criticality
}
```

Hypothesis: the useful memory question is not 'how much history exists?' but 'which smallest sufficient working set must be resident for this phase?'

## Boundary

Strata benchmarks are motivation only; no claim is made that GPU expert-cache behavior transfers directly to cognitive or agent-memory behavior.
