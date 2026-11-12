"""Generate bounded conclusions and tables from the two final measured input cohorts."""
import csv,hashlib,json,pathlib,statistics

root=pathlib.Path('.')
data=[json.loads(p.read_text()) for p in sorted((root/'results').glob('gpu-study-*.json'))]
assert len(data)==2 and {r['seed'] for r in data}=={20261109,20261209}
assert all(len(r['cases'])==64 and not r['validation_only'] for r in data)
assert all(len(c['resident_graph_ms'])==30 and len(c['transfer_inclusive_wall_ms'])==30 for r in data for c in r['cases'])
rows=list(csv.DictReader((root/'results/summary.csv').open()))
assert len(rows)==128

def values(p,mode,key):return [float(r[key]) for r in rows if r['m']=='1024' and int(r['p'])==p and r['mode']==mode]
def interval(v,places=4):
    assert len(v)==2
    return f'{min(v):.{places}f}–{max(v):.{places}f}'

table=['| Modulus | Path | Resident graph median, ms | Transfer-inclusive median, ms |','|---|---|---:|---:|']
for p in (13,257,65521,2013265921):
    for mode in ('int8','int8-base128','fp4','simt'):
        table.append(f"| {p} | {mode} | {interval(values(p,mode,'resident_graph_ms_median'))} | {interval(values(p,mode,'transfer_inclusive_wall_ms_median'))} |")
table='\n'.join(table)
ratios=[]
for r in rows:
    if r['m']=='1024' and r['p']=='13' and r['mode']=='fp4':
        baseline=min(float(v['resident_graph_ms_median']) for v in rows if v['seed']==r['seed'] and v['m']=='1024' and v['p']=='13' and v['mode'] in ('int8','int8-base128'))
        ratios.append(baseline/float(r['resident_graph_ms_median']))
assert min(ratios)>1
outputs=sum(c['exact_outputs'] for r in data for c in r['cases'])
precision=json.loads((root/'results/precision-controls.json').read_text())
assert len(precision['cases'])==2 and all(c['unsafe_mismatches']>0 for c in precision['cases'])
dispatch=json.loads((root/'results/dispatch-profile.json').read_text())
assert dispatch['profile_exit_code']==0
assert any('SM120' in k['name'] and 'BLOCKSCALED' in k['name'] for k in dispatch['kernels'])
assert sum(k['instances'] for k in dispatch['kernels'] if k['name'].startswith('nvjet_sm120_bii_mma_'))==2
results=f'''# Results

A research project by [Requant Network](https://requant.network).

## Completed workload

Two independent input cohorts contain 32 matrix pairs across four shapes and four moduli. Four implementations produce 128 measured cases and {outputs:,} exactly checked output entries. Every resident direct, captured and transfer-inclusive case retains five warmups/thirty observations. The CPU oracle is timed once per matrix pair; its samples are shared by the four corresponding mode records.

All outputs match the independent integer oracle without tolerance. All 128 captured paths also have their output checked before graph timing. The CPU controls retain 405,504 signed-encoding checks, 36 modular products and 2,000 additional native-modulus residue checks. These are execution counts, not independent random-model populations.

## Full pipeline at 1024×1024×1024

Each interval below spans the two input-cohort medians. It is not a confidence interval. Resident graph time includes device encoding, all limb products and modular joins. Transfer-inclusive time additionally includes H2D input, D2H output and host output allocation, with persistent device buffers. Setup/planning is separate in the full CSV.

{table}

![Measured resident and transfer-inclusive pipelines](figures/modular-gemm.png)

The full per-shape, per-seed data, direct-dispatch observations, setup values, min/max and p95 summaries are in [summary.csv](results/summary.csv). Raw data are [cohort 20261109](results/gpu-study-20261109.json) and [cohort 20261209](results/gpu-study-20261209.json).

## Native dispatch evidence

An independent Nsight Systems activity capture of the unchanged two-shape preflight observes the CUTLASS SM120 block-scaled FP4 kernel and two `nvjet_sm120_bii_mma_*` integer-library kernels. Their exact names, counts, profiler version and native binary hash are retained in [dispatch-profile.json](results/dispatch-profile.json). This confirms the observed native kernel families for that preflight; it is not a separate SASS audit of every timing shape. Profiled durations are excluded from the comparative tables, and the full trace remains private because it can contain host/process paths.

## Positive and negative findings

At modulus 13, a single congruent FP4 digit replaces each input; base-13 carries vanish modulo 13. The 1024³ resident FP4 pipeline is {interval(ratios,3)}× faster than the faster measured INT8 base for the corresponding input. That advantage is specific to this residue encoding and workload. Transfer-inclusive latency is around four milliseconds, so the saved tens of microseconds form a small fraction of the complete offload.

For moduli 257, 65521 and 2013265921, INT8 base 256 is faster than the tested FP4 limb pipeline at 1024³. FP4 needs more digit-pair products and joins. Its raw arithmetic capability therefore does not establish a wider-field pipeline win. INT8 base 128 is retained as an encoding comparator and is slower than base 256 in these 1024³ cases.

The scalar SIMT baseline is much slower in this workload. It performs a modular division at every K step and is not an optimized state-of-the-art SIMT implementation. CPU comparisons likewise refer to the documented bounded BLAS oracle, including its own temporaries and modular joins, rather than to every possible CPU finite-field library.

## Precision controls

The deliberately unsafe FP4 dot product loses an integer unit in one output. The deliberately unsafe INT8 accumulation overflows all 16,384 output entries. Bounded K chunks restore all 16,384 entries in each experiment. The raw [precision-control record](results/precision-controls.json) retains the unsafe values, expected values, failure counts and corrected output hashes. Chunking correctness is established; the large unsafe cases are not a separate chunking-throughput benchmark.

## Provenance and limits

The device is an RTX 5060 Ti, with native CUDA 13.3.73 and the recorded CuPy/NVRTC environment. GPU jobs are sequential and comparative timing starts after the heavy CUDA build ends. Clocks and power limits are unchanged. The earlier four-plane CPU comparator was corrected to omit zero upper planes before the final timing campaign; the final narrow-field oracle uses one BLAS product, and the wide field uses four.

The study covers these fields, shapes, constant scale factors and implementations. No energy, complete-miner, application-throughput or production-proof claim is made. See [Methods](METHODS.md), [related work](RELATED_WORK.md), the [final interpretation](FINAL_REPORT.md) and [environment record](results/environment.json).
'''
final=f'''# Final report

A research project by [Requant Network](https://requant.network).

## Conclusion

Exact FP4 modular arithmetic works on consumer Blackwell under explicit digit and accumulation contracts. The most useful positive result in this bounded study is the native modulus-13 encoding: its single-digit resident pipeline is {interval(ratios,3)}× faster than the faster tested INT8 encoding at 1024³. The wider-field experiments favor INT8 base 256. These two outcomes explain where the representation matters more than a peak-throughput specification.

The complete evidence includes {outputs:,} exact output checks, two independent input cohorts, four shapes, four moduli, four implementations, direct/captured/transfer-inclusive measurements, precision failures and exact chunked recovery. The measured scope is a modular matrix product. It does not establish a profitable mining workload or a speedup for an integrated proof system.

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
'''
final += '\n## Follow-up closure, 12 November 2026\n\nThe [follow-up report](FOLLOWUP_REPORT.md) resolves the concrete priorities above with additional experiments, resource gates and updated conclusions. Its findings supersede the open-work wording in this historical campaign.\n'
(root/'RESULTS.md').write_text(results,encoding='utf-8',newline='\n')
(root/'FINAL_REPORT.md').write_text(final,encoding='utf-8',newline='\n')
provenance=dict(source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['generate_report.py','summarize_results.py']},
    artifacts_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['RESULTS.md','FINAL_REPORT.md','results/summary.csv']},
    timer_interpretation='Per-input-cohort summaries; no pooling into independent-instance confidence intervals.')
(root/'results/report-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8',newline='\n')
print('Generated final exact-arithmetic results and bounded conclusions')
