# Exact Tensor Core Arithmetic

Exact modular matrix products on SM120: integer limbs, FP4 encodings and full-pipeline cost.

A research project by [Requant Network](https://requant.network).

**Status: research in progress. No performance result is claimed yet.**

## Research question and scope

Implement arbitrary-precision CPU modular GEMM and independently verify signed limb decomposition, reconstruction and accumulation bounds, including adversarial boundary cases.

Build a pinned SM120 CUTLASS FP4 path with FP32 output and a native INT8 Tensor Core comparator; validate every output against independent integer arithmetic.

Execute complete modular matrix products at multiple shapes and field widths; compare SIMT, INT8 and FP4 including conversion, reduction, resident and transfer-inclusive timing.

Test accumulation chunking and encoding alternatives; measure full pipeline rather than extrapolating GEMM throughput. Retain precision failures and corrected implementations.

Publish two independent input cohorts, five warmups/thirty observations per fixed case, raw outputs or independently checkable evidence, compiler/environment/source provenance and final bounded conclusions.

## Experimental approach

Start from independent exact CPU controls and pinned upstream references. Build hardware-specific paths with validation enabled, then compare complete workloads rather than isolated instruction rates. Retain both independent input cohorts and unsuccessful/negative outcomes.

Fixed-input studies use five warm-ups and thirty measured observations where practical. Stochastic solves and proof commitments use independent inputs and separately identify repeated latency observations. Correctness checks, allocation, compilation, transfer and setup boundaries will be documented per timer.

## Reproduction and evidence

Executable studies and commands are added as experiments become runnable. The committed [research protocol](RESEARCH.md) defines the required experiments and completion gates; reports will distinguish completed measurements from pending work. No GPU benchmark is implied by this initial protocol checkpoint.

## Intended deliverables

- Independent CPU reference and negative correctness controls.
- Executable GPU/native implementations with exact upstream revisions.
- Raw seeded results, derived tables and source/result/build provenance.
- Complete English methods, findings, reproduction commands and limitations.
- CPU/reference/artifact CI on the final published commit.

## License and contributions

Original research code is [MIT licensed](LICENSE). Third-party implementations retain their own licenses. Contributions should include the source revision, environment, exact command, correctness evidence, complete timing boundaries and retained negative results.
