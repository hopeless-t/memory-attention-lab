# Finite RAM Transfer — Attention State Obligation vs Residency

Status: **RESEARCH TRANSFER / BENCH-002 DESIGN INPUT / NO PERFORMANCE CLAIM**

Source research:
- `hopeless-t/finite-ram-lab`
- B461 obligation/residency separation
- B462 hosted physical peak proxy
- B463 coupled numerical residency
- B483-B486 temporary-state biopsy and exact repair
- B494-B500 evidence-bound Governor/application line

## 1. Shared systems question

Memory Attention already asks:

> How much value-side state must actually remain resident on the accelerator?

Finite RAM provides a directly reusable experimental distinction:

> **Logical information obligation is not the same thing as simultaneously resident representation.**

For attention-state research:

```text
required future value information
!=
full V representation resident now
!=
persistent cache bytes
!=
accelerator-resident bytes
```

This does not establish that reconstruction is cheaper.
It defines what must be measured separately.

## 2. Future-sufficient attention state

Let:

- `O_t` = information required for future exact/accepted attention behavior,
- `R_t` = physical representation resident on the accelerator,
- `P_t` = persistent inference state retained somewhere,
- `C_t` = state reconstructed on demand.

A reconstruction strategy is scientifically interesting only if it can show:

`O_t preserved`

while changing:

`R_t`, `P_t`, memory traffic, or reconstruction compute.

The primary semantic gate must occur before resource interpretation.

## 3. B461-B486 methods that transfer

### 3.1 Reference / treatment representation schedules

Finite RAM B461/B462 compared:

- all intermediate representations retained,
- future-sufficient streamed reconstruction.

BENCH-002 analogue:

```text
REFERENCE
  retain the frozen cache/state representation

TREATMENT
  retain only the declared reconstruction-sufficient state
  reconstruct the omitted representation when required
```

The treatment must define exactly which key/value/token/memory representation is retained.

### 3.2 Exactness / fidelity class first

Classify the semantic gate before measuring speed or memory:

- **EXACT** — output equality under the frozen numerical contract,
- **BOUNDED_ERROR** — declared numerical tolerance,
- **EMPIRICAL_CAPABILITY** — task/model quality only.

Do not transfer exactness between classes.

A lower peak with failed semantic equivalence is not a residency win.

### 3.3 Fresh-process / order-balanced measurement

Physical peak measurements should use fresh process or equivalently isolated runs where feasible.

Alternate reference/treatment order to reduce temporal/allocator confounding.

Record:

- baseline memory,
- peak memory,
- normalized peak growth,
- latency,
- bytes transferred,
- reconstruction count,
- output digest or quality gate.

### 3.4 Failure-state biopsy

Finite RAM B483-B486 found that a representation-saving design can hide a large temporary allocation.

Therefore BENCH-002 should instrument stages rather than report only end-of-run peak.

Candidate stages:

```text
retained-state materialization
lookup / host->device transfer
reconstruction
attention consumption
temporary centering/normalization
release
```

If treatment unexpectedly exceeds reference peak, freeze and biopsy the first divergent stage.

## 4. Proposed BENCH-002 protocol delta

### Reference arm

Freeze the repository's ordinary validated cache/state path.

Measure the declared resident and persistent representations.

### Treatment arm

Implement one reconstruction candidate only after VAL-001 identifies a sufficient state contract.

Possible retained coordinates may include, depending on the validated algebra:

- token ids,
- pre-RoPE or otherwise explicitly identified key-side state,
- token-indexed memory address/state,
- layer identity,
- minimal metadata needed for exact reconstruction.

The list is a hypothesis until VAL-001/VAL-002 prove the required relation.

### Hard gates

1. semantic/fidelity gate,
2. cache/state identity gate,
3. output or downstream task check,
4. physical peak measurement,
5. latency and traffic accounting,
6. reproducibility across matched repetitions.

## 5. PROBE vs OPTIMIZE

Adopt the Finite RAM mode separation.

- **PROBE** intentionally varies residency/reconstruction conditions to identify knees and failure mechanisms.
- **OPTIMIZE** chooses a configuration using evidence already collected.
- **OBSERVE** measures without policy intervention.

Invariant:

`OPTIMIZE data != PROBE data`.

Do not tune on a run and count that same run as independent validation.

## 6. Candidate finite-residency frontier

A useful result should report a vector rather than one winner score:

```text
(
  semantic fidelity,
  accelerator peak,
  host peak,
  persistent cache bytes,
  host->device traffic,
  reconstruction FLOPs,
  prefill latency,
  decode latency
)
```

Then identify Pareto points for the tested environment.

No single point should be treated as universal.

## 7. Environment-bound policy

If a later residency Governor is built, bind it to at least:

- GPU/accelerator identity,
- driver/runtime,
- model revision,
- precision/quantization,
- attention implementation,
- sequence/context regime,
- reconstruction implementation.

Finite RAM's B500 host-bound policy is a methodological precedent, not a threshold source.

## 8. What does not transfer

Do not import:

- GitHub-hosted RAM byte thresholds,
- q2/q4/q7 policies,
- CRT arithmetic claims,
- CPU mmap behavior as GPU allocator behavior.

The transferable asset is the **experimental contract**.

## 9. Claim ceiling

`FINITE_RAM_ATTENTION_RESIDENCY_TRANSFER_PROTOCOL_DEFINED`

No Memory Attention benchmark result is changed by this document.
