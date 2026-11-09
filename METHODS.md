# Methods

A research project by [Requant Network](https://requant.network).

## Scope and hypotheses

The primary question is whether lossless digit decomposition makes native FP4 useful for exact modular GEMM on SM120. The comparator includes native INT8 in two balanced bases and an original scalar SIMT kernel. A second question is whether a fast partial product survives the cost of encoding, multiple limb products, modular recombination, dispatch and transfers. Deliberately unsafe accumulation is a separate correctness experiment.

The studies use synthetic centered residues modulo 13, 257, 65521 and 2013265921. None is a mining benchmark or a complete proof-system implementation. No economic or energy conclusion follows from a matrix operation alone.

## Implementations and exactness

The native bridge uses the pinned NVIDIA CUTLASS SM120 block-scaled configuration with a 128×128×128 thread-block tile and a 1×1×1 cluster. Both its accumulator and output are FP32. Replacing a lower-precision output is essential: exact accumulation would not suffice if the output then rounded to BF16. Block scales are fixed to one; the FP4 digit is stored as half its integer value; an epilogue multiplier of four restores integer products.

Base 13 assigns each residue to a representable signed digit. Encoding iterates `x = (x - digit) / base`, with a fixed limb count sufficient for the entire centered field range. At modulus 13, the first congruent digit is sufficient because the carry contributes zero modulo p; that specialized contract is retained separately. INT8 uses balanced base 128 or 256 and native cuBLAS INT8/INT32 GEMM. B is stored transposed so both input digit planes are contiguous along K. Each pair of digit planes has a separate product. Results are reduced modulo p before multiplying their modular base-power weight, which keeps the widest intermediate within signed INT64 for the tested fields.

The sufficient FP4 bound is `K * 64 < 2^24`, and the INT8 bound is `K * max_digit^2 < 2^31`. The first is conservative and excludes unsafe integer magnitudes; it is not an assertion that every larger dot product must fail. The separate adversarial control constructs a measured failure beyond it. Splitting K into bounded pieces and reducing partial sums restores exact outputs. Constant inputs in the unsafe tests are distinguishable from the independently seeded workload cohorts.

The original SIMT baseline assigns one output entry per thread and reduces the accumulator modulo p on every K step. It is deliberately simple and exact. It is not a claim about the fastest possible SIMT implementation. Speedups relative to it must retain that qualification.

## Independent oracle

Small cases use arbitrary-precision Python integer multiplication. Larger cases use unsigned base-65536 planes and float64 BLAS. Fields up to 16 bits need one plane product; the wider field needs four. Zero upper planes are omitted from the CPU comparator. Each dot product is an integer with `K * 65535^2 < 2^53`, so individual products and all positive partial sums are exactly representable. Every plane product is converted to INT64, reduced modulo p, and recombined with modular weights. This oracle is first checked against arbitrary-precision arithmetic. Plain INT64 GEMM is unsuitable for the widest field because its unreduced dot products overflow.

All GPU outputs are compared entry by entry with this independent oracle. Tolerance comparisons are not used. Seeded random matrices include deterministic first-row boundary values. Source, input, output and native-library hashes accompany the records. Input digests identify cohorts; thirty observations of one input are not thirty independent arithmetic workloads.

## Shapes and timers

The fixed shapes are 128×128×128, 256×256×256, 512×256×512 and 1024×1024×1024, written M×N×K. Two generator seeds, 20261109 and 20261209, are evaluated separately. Each timed fixed case has five warmups and thirty observations.

Resident direct timing uses CUDA events around GPU encoding, zeroing the result, every native limb GEMM, and every modular reduction/recombination. It includes dispatch gaps visible on the stream. Per-mode working-buffer allocation, planning and constant-scale setup are outside that timer and recorded separately as host wall time. Initial persistent input/output allocation, global library/context initialization and NVRTC compilation are outside the per-mode setup timer. Captured timing covers the same operations inside a CUDA graph, whose output is checked before timing. Transfer-inclusive host wall time additionally includes input H2D, output D2H and host output allocation, with persistent device buffers. It does not include device allocation or plan construction.

The CPU comparison includes its own decomposition, BLAS products and modular recombination, with six OpenBLAS threads. CPU and GPU timing campaigns run only after the heavy CUDA build ends, and GPU experiments run sequentially. Correctness-only preflight wall times are not comparative results. No clocks, power limits or thermal policy are changed.

## Analysis and reproduction limits

[summarize_results.py](summarize_results.py) emits per-seed medians, ranges and interpolated p95 values, retaining every underlying observation. It does not pool repeated observations into an independent-input confidence interval. The hardware/software record identifies native CUDA 13.3, the CuPy runtime/NVRTC versions, the GPU, CPU and thread settings.

The tested scale factors are constant and exact, the fields and shapes are bounded, and chunking is validated separately from the main timing cohort. Performance at much larger matrices, other devices, alternative schedules, fused reduction, batched limb GEMMs and optimized SIMT implementations remains unmeasured. These are limits on extrapolation, not omitted observations within the declared fixed workload.
