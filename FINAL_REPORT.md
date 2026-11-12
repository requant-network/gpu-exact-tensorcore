# Final report

A research project by [Requant Network](https://requant.network).

## Conclusion

Exact FP4 modular arithmetic works on consumer Blackwell under explicit digit and accumulation contracts. The most useful positive result in this bounded study is the native modulus-13 encoding: its single-digit resident pipeline is 1.425–1.427× faster than the faster tested INT8 encoding at 1024³. The wider-field experiments favor INT8 base 256. These two outcomes explain where the representation matters more than a peak-throughput specification.

The complete evidence includes 40,370,176 exact output checks, two independent input cohorts, four shapes, four moduli, four implementations, direct/captured/transfer-inclusive measurements, precision failures and exact chunked recovery. The measured scope is a modular matrix product. It does not establish a profitable mining workload or a speedup for an integrated proof system.

## Mathematical interpretation

For the wider moduli, inputs use lossless centered-integer limb expansions. Every digit pair contributes a separate product with its base-power weight. The GPU reduces each partial modulo p before weighted recombination so the widest join remains within signed INT64. FP4's representable signed alphabet makes a radix-13 construction possible, but the resulting number of products grows faster than the INT8 construction in the tested fields.

Modulus 13 admits a different contract. The lookup digit is congruent to its input modulo 13, and the omitted carry is already zero in that field. A single FP4 product is therefore enough; this is a modular encoding, rather than a truncated integer approximation. The independent CPU checks and complete GPU products verify that specialization. It should not be extrapolated to a wider modulus without a new reconstruction argument.

Both paths need accumulation bounds. Exact input representation alone is insufficient: the retained unsafe FP4 and INT8 cases produce actual errors. Chunking and modular reduction recover the mathematical result. The tests use exact equality, which makes those failures visible.

## Performance interpretation

The resident timer includes conversion, every native product and recombination. It measures a useful persistent-data workload, while the separate transfer-inclusive timer measures a host offload with persistent device allocations. The modulus-13 compute saving is only tens of microseconds; transferring the 1024³ inputs and result takes the complete call to roughly four milliseconds. Integration should therefore preserve data residency when possible rather than assuming that a resident advantage survives an offload.

For the wider fields, the tested FP4 construction loses to INT8 even before transfers. More limb pairs and modular joins outweigh its arithmetic path. The current evidence favors optimizing the INT8 pipeline for those fields. Its lead is specific to this implementation; alternative schedules or fused reconstruction require their own measurements.

The naive SIMT and bounded BLAS oracle serve explicit baselines. Their measured costs do not support a claim against optimized CPU finite-field packages or all possible SIMT kernels. Thirty timing observations of one input characterize repeated execution of that input. They do not create thirty independent mathematical instances, and the report keeps the two input seeds separate.

## Completed protocol gates

| Required experiment | Retained evidence |
|---|---|
| Independent integer arithmetic and encoding bounds | `cpu_study.py`, `experiments/exact.py`, CPU record and arbitrary-precision checks inside the GPU driver |
| Native SM120 FP4 and native INT8 comparator | Attributed pinned CUTLASS bridge, real preflight and complete exact GPU products |
| Complete modular products, multiple widths/shapes and timer boundaries | 128 final measured cases, 30 observations per timer, full raw JSON and derived CSV |
| Encoding alternatives and chunked accumulation | INT8 bases 128/256, lossless FP4 limbs, native-modulus FP4 residue map, actual unsafe failures and corrected chunks |
| Replication, provenance, conclusions and CPU CI | Two input seeds, source/native-binary hashes, environment, reproducible summaries and this report |

## Follow-up priorities

First, test the native-small-field encoding in a real application that can keep its operands on the device. Its observed resident advantage is the strongest positive lead, but application relevance and offload amortization remain separate questions.

Second, fuse or batch encoding and modular joins in the wider-field INT8 path. The current result supports that implementation direction more directly than another FP4 peak-throughput comparison. Third, study residue-number or extension-field constructions only with a complete reconstruction proof and measured conversion cost; the small-field result does not supply those automatically.

Larger shapes, different GPUs, varied scale factors, energy sampling and production integration are outside the completed protocol. They remain follow-up work rather than implied results. [Results](RESULTS.md), [Methods](METHODS.md), [related work](RELATED_WORK.md) and the [protocol](RESEARCH.md) provide the complete scope and reproduction links.

## Follow-up closure, 12 November 2026

The [follow-up report](FOLLOWUP_REPORT.md) resolves the concrete priorities above with additional experiments, resource gates and updated conclusions. Its findings supersede the open-work wording in this historical campaign.
