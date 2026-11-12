"""GF13 block-Krylov recurrence with cached operands and complete exact checks."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import cupy as cp
import numpy as np
from gpu_study import oracle
from experiments.modular_plan import Plan


def main():
    assert __debug__
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=20261112);parser.add_argument('--output',default='results/reuse-study.json')
    args=parser.parse_args();rng=np.random.default_rng(args.seed);cases=[]
    stream=cp.cuda.Stream(non_blocking=True);stream.use()
    for n,columns,steps in ((256,128,8),(1024,256,8),(1024,256,32)):
        a=rng.integers(-6,7,(n,n),np.int64);initial=rng.integers(-6,7,(columns,n),np.int64)
        expected=initial.copy()
        for _ in range(steps):expected=oracle(a,expected,13).T.copy()
        da,bt=cp.asarray(a),cp.asarray(initial);initial_gpu=bt.copy();plans={mode:Plan(da,bt,13,mode) for mode in ('int8','fp4')}
        samples={};memory={};graphs={};runs={}
        for mode,plan in plans.items():
            for cached in (False,True):
                name=mode+('_cached_a' if cached else '_reencode_a')
                def run(plan=plan,cached=cached):
                    cp.copyto(bt,initial_gpu)
                    for _ in range(steps):plan.run(reuse_a=cached);cp.copyto(bt,plan.out.T)
                run();stream.synchronize();assert np.array_equal(cp.asnumpy(bt),expected)
                stream.begin_capture();run();graphs[name]=stream.end_capture();graphs[name].launch(stream);stream.synchronize()
                assert np.array_equal(cp.asnumpy(bt),expected)
                runs[name]=run
                samples[name]={'resident_graph_ms':[],'offload_wall_ms':[]};memory[name]=plan.partial.nbytes+plan.ea.nbytes+plan.eb.nbytes+plan.out.nbytes
        for iteration in range(35):
            names=list(graphs)
            if iteration%2:names.reverse()
            for name in names:
                start,end=cp.cuda.Event(),cp.cuda.Event();start.record();graphs[name].launch(stream);end.record();end.synchronize()
                elapsed=cp.cuda.get_elapsed_time(start,end);assert np.array_equal(cp.asnumpy(bt),expected)
                begin=time.perf_counter();da.set(a);initial_gpu.set(initial);runs[name]();actual=bt.get();stream.synchronize()
                full=(time.perf_counter()-begin)*1000;assert np.array_equal(actual,expected)
                if iteration>=5:samples[name]['resident_graph_ms'].append(elapsed);samples[name]['offload_wall_ms'].append(full)
        del graphs,runs
        for plan in plans.values():plan.close()
        cases.append(dict(n=n,columns=columns,steps=steps,p=13,
                          input_sha256=hashlib.sha256(a.tobytes()+initial.tobytes()).hexdigest(),
                          output_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),every_final_state_exact=True,
                          logical_work_buffer_bytes=memory,samples=samples))
        print(n,columns,steps,{name:float(np.median(x['resident_graph_ms'])) for name,x in samples.items()},flush=True)
    paths=['reuse_study.py','experiments/modular_plan.py','gpu_study.py','experiments/exact.py','cuda/modular.cu','cuda/joins.cu','cuda/fp4_bridge.cu']
    record=dict(utc=datetime.now(timezone.utc).isoformat(),seed=args.seed,warmups=5,samples=30,cases=cases,
                source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
                binary_sha256=hashlib.sha256(Path('build/libexact.so').read_bytes()).hexdigest(),
                scope='Controlled GF13 block Krylov recurrence X(t+1)=A*X(t), including reencoding changing states, modular reduction and transpose/copy. Cached-A excludes only unchanged operand encoding. All recurrences match an independent bounded CPU BLAS modular oracle. Resident graph includes state reset and all steps; offload adds A/initial-state uploads and final-state download. Setup, capture and CPU checking excluded. Dense synthetic application, not a deployed proof or a production application speedup.')
    Path(args.output).write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
