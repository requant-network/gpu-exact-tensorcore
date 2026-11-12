# Application reuse and modular-join closure

Requant Network, 12 November 2026. This supplement resolves the GF(13) application-reuse and wider-field join questions from the [historical report](FINAL_REPORT.md).

## GF(13) block recurrence

The controlled application is `X(t+1) = A X(t) mod 13`, with a fixed dense matrix A and changing block vectors. It executes 8 or 32 dependent steps. Each step includes encoding the changing state, exact TensorCore multiplication, modular reconstruction and the transpose needed by the next step. Cached-A and reencoded-A paths are compared for both INT8 and FP4; no approximate result is accepted. Independent CPU modular products validate every final state. This is a block-Krylov workload, not a full Wiedemann solver.

Two matrix seeds, five warm-ups and thirty samples per case use alternating variant order. Captured device timing excludes setup; the separate offload path includes initial matrix/state upload and final state download. Plan/allocation and independent verification are outside the steady-state timing. Selected resident medians, milliseconds:

| Seed | N | Block columns | Steps | INT8 reencode A | INT8 cached A | FP4 reencode A | FP4 cached A |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261112 | 256 | 128 | 8 | 0.0776 | 0.0590 | 0.0897 | 0.0796 |
| 20261112 | 1024 | 256 | 8 | 0.3798 | 0.2112 | 0.2899 | 0.2130 |
| 20261112 | 1024 | 256 | 32 | 1.4871 | 0.8145 | 1.1409 | 0.8307 |
| 20261212 | 256 | 128 | 8 | 0.0774 | 0.0590 | 0.0898 | 0.0798 |
| 20261212 | 1024 | 256 | 8 | 0.3794 | 0.2110 | 0.2903 | 0.2133 |
| 20261212 | 1024 | 256 | 32 | 1.4900 | 0.8169 | 1.1415 | 0.8321 |

Caching A is useful. After that reuse is applied to both implementations, FP4 has no resident advantage in these tested complete recurrences: at N=1024 it is close to INT8, and at N=256 it is slower. A stand-alone small-field GEMM speedup therefore does not establish an application-level FP4 advantage. Some transfer-inclusive FP4 observations are lower because its input encoding is smaller; those are scope-specific results in the CSV, not evidence of faster dependent arithmetic.

## Batched wider-field reconstruction

The added [join kernel](cuda/joins.cu) reduces all INT8 limb-product partials in one pass rather than launching a separate modular join per pair. It preserves the original accumulation bounds, reduces each partial modulo p before multiplying by its radix weight, and uses signed 64-bit intermediates within the established bounds. The cost is retaining all limb-pair partials instead of one reusable partial matrix. Exact independent CPU oracles check every output element at p=65521 and p=2013265921.

| Seed | N | Prime | Streamed join | Batched join | Streamed partial MiB | Batched partial MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261112 | 256 | 65521 | 0.0528 | 0.0426 | 0.25 | 2.25 |
| 20261112 | 256 | 2013265921 | 0.1040 | 0.0835 | 0.25 | 4.00 |
| 20261112 | 1024 | 65521 | 0.4567 | 0.4537 | 4.00 | 36.00 |
| 20261112 | 1024 | 2013265921 | 0.9527 | 0.9440 | 4.00 | 64.00 |
| 20261212 | 256 | 65521 | 0.0527 | 0.0427 | 0.25 | 2.25 |
| 20261212 | 256 | 2013265921 | 0.1041 | 0.0836 | 0.25 | 4.00 |
| 20261212 | 1024 | 65521 | 0.4575 | 0.4532 | 4.00 | 36.00 |
| 20261212 | 1024 | 2013265921 | 0.9525 | 0.9441 | 4.00 | 64.00 |

Batching helps the small N=256 case by roughly one fifth. At N=1024 the resident gain is approximately one percent or less, while partial-buffer memory grows substantially. Use the batched form only when launch overhead matters and its memory cost is acceptable; this evidence does not justify replacing the streaming path for all sizes.

The [full median/p95 table](results/followup-summary.csv) includes offload results. Run [reuse_study.py](reuse_study.py) and [join_study.py](join_study.py), then repeat with seed 20261212 and `--output results/reuse-replication.json` / `--output results/join-replication.json`. Both use [the same native modular plan](experiments/modular_plan.py) and the pinned original library. `python verify_followups.py` checks all source bindings, exact-result flags, sample counts and memory accounting. The original artifact checks remain required.

Residue-number/extension-field constructions and production sparse solvers need distinct reconstruction and application designs. They are not inferred from these results. Hardware conclusions apply to the measured single RTX 5060 Ti, not heterogeneous multi-GPU execution or energy efficiency.

## Server timing boundary

GPU campaigns run serially on the same RTX 5060 Ti. Clocks, power limits and unrelated server services are unchanged. Measurements describe this server and retained workload; small percentage differences are not treated as universal improvements. No energy measurement or heterogeneous multi-GPU throughput is inferred.
