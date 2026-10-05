# Capability Residency Crossover — systems analogy

Date: 2026-10-05
Status: research intake / no architecture claim

## Trigger

Source discussion: https://gigazine.net/news/20261005-deepseek-harness/

The harness discussion exposes a broader systems principle that matches this repository's core distinction:

```text
capacity != active computation != physical residency != persistent state
```

The same separation can be applied outside attention to agent capabilities.

## Analogy

Memory Attention asks how much value-side state must remain accelerator-resident.

A lifecycle-managed harness asks an analogous question:

> How much capability state must remain runtime-resident for the agent to preserve useful responsiveness and correctness?

Candidate capability states:

```text
HOT   active runtime and state resident
WARM  small retained state; fast resume
COLD  reconstructable/checkpointed state only
OFF   not loaded
```

## Shared decomposition

For both model memory and harness capabilities, separate:

```text
addressability
content/state
reconstruction cost
resident bytes
transfer bytes
persistent bytes
active compute
latency
reuse probability
```

A large logical capacity may therefore exist without all of it occupying the fastest tier simultaneously.

## Research question

Can the lab's residency/reconstruction accounting produce a reusable crossover model of the form:

```text
keep resident when:
  expected future reuse benefit > retention cost

reconstruct/offload when:
  retention cost > expected retrieval/reconstruction cost
```

For a state object `i`:

```text
C_keep(i) ~= resident_bytes_i * residency_shadow_price * hold_time

C_rebuild(i) ~= P(reuse_i) * (
    transfer_cost_i
  + reconstruction_compute_i
  + latency_penalty_i
  + failure_risk_i
)
```

This is a decision model, not an empirical finding.

## Useful distinction

The analogy should not collapse different layers:

```text
accelerator-resident tensor
host-resident tensor
memory-mapped model state
agent tool process
remote provider connection
subagent/model instance
```

They share a residency/reconstruction trade-off but have different failure semantics and measurement units.

## Candidate contribution back to the lab

The harness case can serve as a non-neural control example for the lab's broader thesis:

> logical capacity and physical residency are separable resources.

If the same accounting vocabulary remains useful across both attention memory and tool/model/provider residency, that is evidence for a reusable systems abstraction—not evidence that the mechanisms are otherwise equivalent.

## Cross-repo hooks

- `finite-ram-lab`: memory-budget and residency mechanics.
- `harness-component-economics`: verified-success cost of residency decisions.
- `mvca`: restore/recovery/authority invariants.
- `next-generation-github`: demand-paged capability graph.

## Non-claim

This note records a mechanistic systems analogy. It does not imply that Memory Attention techniques directly implement agent-harness paging, or vice versa.
