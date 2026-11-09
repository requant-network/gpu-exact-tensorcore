# Research protocol

A research project by [Requant Network](https://requant.network).

## Required experiments

1. Implement arbitrary-precision CPU modular GEMM and independently verify signed limb decomposition, reconstruction and accumulation bounds, including adversarial boundary cases.
2. Build a pinned SM120 CUTLASS FP4 path with FP32 output and a native INT8 Tensor Core comparator; validate every output against independent integer arithmetic.
3. Execute complete modular matrix products at multiple shapes and field widths; compare SIMT, INT8 and FP4 including conversion, reduction, resident and transfer-inclusive timing.
4. Test accumulation chunking and encoding alternatives; measure full pipeline rather than extrapolating GEMM throughput. Retain precision failures and corrected implementations.
5. Publish two independent input cohorts, five warmups/thirty observations per fixed case, raw outputs or independently checkable evidence, compiler/environment/source provenance and final bounded conclusions.

## Evidence and completion gates

Original code is MIT licensed; upstream code and patches retain attribution and applicable licenses. Execute GPU timing campaigns sequentially on the available RTX 5060 Ti 16 GB. Keep independent generator seeds distinct from repeated timing samples. Preserve unsuccessful trials, negative findings, exact integer/proof checks, raw source/result hashes, compiler flags and hardware/software metadata. No energy claim without sampled measurements.

Completion requires all experiments above, executable CPU/GPU studies, independently recheckable retained evidence, derived tables, a detailed English README, METHODS.md, RESULTS.md and FINAL_REPORT.md. Public repository topics/homepage identify Requant Network. Latest published-head CPU/reference/artifact CI must pass. A documented negative experiment may complete a hypothesis test; a missing required experiment may not.
