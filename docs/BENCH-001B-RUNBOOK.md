# BENCH-001B B1 Capture Runbook

> **Status:** EXECUTION KIT VALIDATED — PASS  
> **Performance evidence:** NONE until a real approved GPU execution completes  
> **Paid-resource authority:** NONE

This runbook describes how to capture a BENCH-001B evidence bundle **after** a
qualifying GPU resource already exists and has been explicitly approved.

It does not describe how to purchase, provision, start, or register a GPU
resource.

## Preconditions

The operator must already have:

1. an isolated or disposable NVIDIA CUDA environment;
2. explicit Human approval if that environment costs money;
3. a clean checkout of `Joluck/memory-attention` at:

~~~text
8176f1feaff6724af670e70d75cd9767f8e38223
~~~

4. the upstream project's required Python environment, including:
   - PyTorch with CUDA;
   - flash-attn >= 2.1;
   - the FLA package expected by the pinned upstream source;
5. this lab repository available separately for the capture/validation tool.

The output directory must be **outside** the frozen upstream checkout.

## What the capture kit does not do

The capture kit does not:

- create a cloud instance;
- start or stop a cloud instance;
- select a paid provider;
- register a GitHub self-hosted runner;
- accept GitHub secrets;
- upload evidence;
- commit evidence;
- merge evidence;
- infer that a GPU resource is approved.

It only inspects and uses an already-running environment.

## Step 1 — Probe only

Probe is the default mode.

~~~bash
python -m memory_attention_lab.experiments.bench001b_capture \
  --protocol /path/to/memory-attention-lab/specs/BENCH-001B.protocol.json \
  --upstream-dir /path/to/memory-attention \
  --out-dir /path/outside/upstream/bench001b-bundle
~~~

Without `--execute`, the tool must **not** run the benchmark.

It verifies:

~~~text
upstream HEAD
clean upstream worktree
profile/bmk.py Git blob

CUDA availability
BF16 support
compute capability >= 8.0
flash-attn >= 2.1

GPU name
VRAM
driver/runtime metadata
~~~

and writes:

~~~text
probe.json
~~~

`probe.json` has no performance authority.

## Step 2 — Human checkpoint

Before adding `--execute`, confirm:

~~~text
the resource is approved
the resource cost is accepted
the source checkout is frozen
the output directory is external to the source checkout
the probe succeeded
~~~

If any of those are false, stop.

## Step 3 — Execute the source-default profile

Only after the Human checkpoint:

~~~bash
python -m memory_attention_lab.experiments.bench001b_capture \
  --protocol /path/to/memory-attention-lab/specs/BENCH-001B.protocol.json \
  --upstream-dir /path/to/memory-attention \
  --out-dir /path/outside/upstream/bench001b-bundle \
  --execution-id <human-readable-unique-id> \
  --execute
~~~

The kit derives the benchmark command from the frozen B0 protocol.

It does not maintain a second independent copy of the source-default benchmark
configuration.

The real execution runs the pinned upstream `profile/bmk.py` with:

~~~text
--check-correctness
--json <absolute output path>
~~~

and all frozen source-default dimensions.

## Step 4 — Bundle contents

A completed candidate bundle contains:

~~~text
probe.json
benchmark.stdout.txt
benchmark.stderr.txt
bmk_results.json
hardware.json
validation.json
~~~

### bmk_results.json

This is the raw upstream benchmark output.

The capture kit must not rewrite its timing rows.

### hardware.json

Records:

~~~text
execution id
source pin
GPU identity / capability / VRAM / BF16
Python / torch / CUDA / driver / flash-attn / OS
exact benchmark argv
SHA-256 of raw bmk_results.json
~~~

### validation.json

This is the B0 CPU-side validator result.

A PASS means only that the bundle obeys the frozen protocol.

It does not make the benchmark result canonical.

## Step 5 — Publication boundary

A real bundle should then be reviewed separately.

Before any claim movement:

1. run the public validator again on a clean environment;
2. verify source and result hashes;
3. inspect raw round samples;
4. check hardware/runtime provenance;
5. compare repeated runs if the experiment contract requires them;
6. publish through a dedicated evidence PR;
7. move MA-005 only if the reviewed evidence actually supports its scoped claim.

## Security boundary

Do not attach a persistent personal workstation as a general public-repository
self-hosted runner merely to execute this benchmark.

The selected topology is:

~~~text
isolated approved GPU executor
    -> evidence bundle
    -> public Actions validator
~~~

not:

~~~text
public repository
    -> unrestricted persistent personal GPU runner
~~~

## Current state

~~~text
B0 protocol:
    VALIDATED — PASS

B1 capture kit:
    VALIDATED — PASS
    4 environments / 53 tests per lane

B1 real CUDA execution:
    NOT EXECUTED

paid GPU:
    NOT AUTHORIZED BY THIS REPOSITORY
~~~


## Validation checkpoint

The capture kit was validated on public GitHub Actions before any real GPU
execution:

~~~text
run:
36080460403

source commit:
1428a9dc2b40a61e8ec2af6e9536a193c87566fd

Ubuntu 24.04 / Python 3.12:
PASS

Ubuntu 24.04 / Python 3.13:
PASS

macOS 15 / Python 3.12:
PASS

macOS 15 / Python 3.13:
PASS

full suite:
53 tests PASS

execution-topology decision support:
1.5 million states PASS
~~~

This validates capture-kit behavior and safety boundaries only.

No CUDA timing was executed.
