# Related work and contribution boundaries

A research project by [Requant Network](https://requant.network).

Exact finite-field linear algebra built on numerical BLAS is established work. Dumas, Giorgi and Pernet describe this approach in [Dense Linear Algebra over Finite Fields: the FFLAS and FFPACK packages](https://arxiv.org/abs/cs/0601133). This study uses a bounded integer-valued float64 oracle in that tradition; it does not claim to invent exact arithmetic through floating-point hardware.

The device implementation builds on NVIDIA's [pinned SM120 NVFP4 example](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu). NVIDIA supplies the block-scaled GEMM machinery. The research changes its output contract to FP32, adds lossless signed or modular residue encodings, proves sufficient accumulation bounds, and measures the entire modular pipeline against native INT8 and an explicit SIMT baseline.

## A useful distinction: integer encoding and residue encoding

For the wider fields, a value must be represented by several base-13 digits before weighted reconstruction. The implementation needs the carry planes. For modulus 13 itself, every table entry d(r) satisfies `d(r) = r mod 13`; therefore the carry contributes zero to the final field result. One digit per input is sufficient, even when that digit is not the original centered integer.

For any two input vectors, replacing each coordinate by a congruent digit preserves their dot product modulo 13. Storing half-digits in FP4 and multiplying the FP32 GEMM output by four then computes the exact integer digit dot product under the stated accumulation bound. Reducing this result modulo 13 gives the original modular product. This argument explains the specialized experiment; it does not establish research priority for the encoding.

The experimentally relevant question is the measured cost after all digit products, conversion and recombination. Peak FP4 throughput alone does not decide that question. The final reports distinguish ordinary wider-field limb arithmetic from the specialized native-small-field case.

## Limits of the contribution

The work provides source, pinned builds, exact output checks, explicit precision failures, chunked recovery, replicated input cohorts and end-to-end timing boundaries. It does not provide a new mining protocol, a general proof-system accelerator, an optimal CPU implementation or a proof that these schedules are the fastest possible SM120 kernels. Residue-number reconstruction, larger extension fields, fused recombination and application-level integration are separate follow-up experiments.
