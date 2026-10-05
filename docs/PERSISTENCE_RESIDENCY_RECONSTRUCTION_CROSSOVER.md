# Persistence, residency, and reconstruction crossover

Status: systems analogy / research question, not a Memory Attention result

Source intake: https://note.com/npaka/n/n341b20a052c6

The durable-agent discussion exposes the same systems distinction already central here:

```text
persistent state
!= active computation
!= physical residency
```

For agent systems, a small durable semantic state can survive while model/context state is reconstructed only on demand. For Memory Attention, the analogous question is whether persistent inference information must remain accelerator-resident or can be reconstructed/retrieved from cheaper tiers.

This does **not** imply a shared mechanism. It suggests a reusable accounting vocabulary:

```text
logical persistence
physical residency
reconstruction cost
access frequency
latency budget
traffic / bandwidth
```

Candidate BENCH-002 reporting extension:

- persistent logical bytes;
- accelerator-resident bytes;
- host/storage-resident bytes;
- reconstruction/retrieval bytes per decode step;
- reconstruction latency;
- break-even access frequency where reconstruction loses to residency.

Useful invariant:

> **Persisted information does not imply continuously resident representation.**

This note is an accounting crossover only; it makes no architecture-quality, latency, or model-quality claim.