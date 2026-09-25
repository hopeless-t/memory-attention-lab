# BENCH-001A canonical reference evidence

> **Status:** REVIEWED REFERENCE CANDIDATE  
> **Scientific authority:** deterministic residency / staging accounting only  
> **Performance authority:** NONE  
> **CUDA latency authority:** NONE

This directory preserves the structured evidence generated from the merged
BENCH-001A source commit.

## Generation provenance

~~~text
source commit:
bba45d936ba8f04e02e8e63dae106e6641ed1585

GitHub Actions run:
36078561778

workflow:
.github/workflows/bench001-accounting.yml

artifact:
bench001a-reference-evidence

artifact id:
10841415772

artifact digest:
sha256:7c674002fc6d3403111f4f1f34640c4701796b0ed87ec28d06363beeb1443963

reference spec SHA-256:
088be700dbce22acbcf49976acb4cad5a4e40657aef7b9a6c14f9214016b7773

parent BENCH-001 design Git blob SHA-1:
c1fcab785da2358b61ac7006bc4b015f459bc869
~~~

The generated JSON files were copied without changing their bytes.

Git blob identities:

~~~text
manifest.json
b47a43aa8552df4c7db3c0f5adfc4ca172fd25e2

metrics.json
6fd8489d72bb2b8420e0a7db7edbf6835081c55d

checks.json
8d6652ba8dbf38780fa7b018dd42193191e80280

observations.json
a18618903f4a95fd7760071121ef782fb7f33323
~~~

The later publication commit is intentionally not written into the generated
manifest. The manifest identifies the source code that performed the
calculation, avoiding circular provenance.

## Validation envelope

The merged source commit was validated by:

~~~text
Ubuntu 24.04 / Python 3.12
Ubuntu 24.04 / Python 3.13
macOS 15 / Python 3.12
macOS 15 / Python 3.13

full pytest suite:
PASS

randomized accounting:
500,000 cases PASS

claim-partition decision support:
1.5 million simulated engineering states PASS
~~~

## Source-default accounting result

For the frozen source-default prefill case:

~~~text
Standard Wv parameter bytes:
201,326,592 B
192 MiB

MA table bytes:
3,145,728,000 B
approximately 2.93 GiB

MA-Offload GPU parameter-byte delta vs Standard:
-201,326,592 B
-192 MiB

pipeline GPU staging:
268,435,456 B
256 MiB

bulk GPU staging:
1,610,612,736 B
1.5 GiB

logical H2D payload per full forward:
1,610,612,736 B
1.5 GiB

KV cache at capacity 2048:
3,221,225,472 B
3.0 GiB
~~~

The parameter delta, staging working set, transfer payload, and KV cache are
different resource categories.

They must not be added or compared as if they were one kind of memory.

## Authority boundary

This reference supports the claim that, under the frozen source-scope
accounting contract, placing the MA table in host memory removes the dedicated
Wv parameter bytes from accelerator-resident parameters.

It does **not** establish:

- measured peak GPU memory;
- CUDA allocator behavior;
- CPU gather latency;
- H2D latency;
- overlap effectiveness;
- latency neutrality;
- H800 benchmark reproduction;
- MA-Recall KV-cache savings;
- model-quality improvement;
- universal deployment-memory reduction.

Those require separate evidence lanes.
