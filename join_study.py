"""Qualify a single modular join over all INT8 limb products."""
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
    parser.add_argument('--seed',type=int,default=20261112);parser.add_argument('--output',default='results/join-study.json')
    args=parser.parse_args();rng=np.random.default_rng(args.seed);cases=[]
    stream=cp.cuda.Stream(non_blocking=True);stream.use()
    for n in (256,1024):
      for p in (65521,2013265921):
        a=rng.integers(-p//2,p//2+1,(n,n),np.int64);bt=rng.integers(-p//2,p//2+1,(n,n),np.int64)
        a[0]=np.resize(np.array([-p//2,p//2,0,1,-1],np.int64),n)
        expected=oracle(a,bt,p);da,db=cp.asarray(a),cp.asarray(bt)
        plans={name:Plan(da,db,p,batched=batched) for name,batched in [('streamed_join',False),('batched_join',True)]}
        graphs={};samples={name:{'resident_graph_ms':[],'offload_wall_ms':[]} for name in plans}
        for name,plan in plans.items():
            plan.run();stream.synchronize();assert np.array_equal(cp.asnumpy(plan.out),expected)
            stream.begin_capture();plan.run();graphs[name]=stream.end_capture()
            graphs[name].launch(stream);stream.synchronize();assert np.array_equal(cp.asnumpy(plan.out),expected)
        for iteration in range(35):
            names=list(plans)
            if iteration%2:names.reverse()
            for name in names:
                plan=plans[name];start,end=cp.cuda.Event(),cp.cuda.Event();start.record();graphs[name].launch(stream);end.record();end.synchronize()
                elapsed=cp.cuda.get_elapsed_time(start,end);assert np.array_equal(cp.asnumpy(plan.out),expected)
                begin=time.perf_counter();da.set(a);db.set(bt);plan.run();actual=plan.out.get();stream.synchronize()
                full=(time.perf_counter()-begin)*1000;assert np.array_equal(actual,expected)
                if iteration>=5:samples[name]['resident_graph_ms'].append(elapsed);samples[name]['offload_wall_ms'].append(full)
        memory={name:plan.partial.nbytes for name,plan in plans.items()}
        for plan in plans.values():plan.close()
        cases.append(dict(n=n,p=p,every_output_exact=True,partial_buffer_bytes=memory,samples=samples,
                          input_sha256=hashlib.sha256(a.tobytes()+bt.tobytes()).hexdigest(),
                          output_sha256=hashlib.sha256(expected.tobytes()).hexdigest()))
        print(n,p,{name:float(np.median(x['resident_graph_ms'])) for name,x in samples.items()},flush=True)
    paths=['join_study.py','experiments/modular_plan.py','gpu_study.py','experiments/exact.py','cuda/modular.cu','cuda/joins.cu','cuda/fp4_bridge.cu']
    record=dict(utc=datetime.now(timezone.utc).isoformat(),seed=args.seed,warmups=5,samples=30,cases=cases,
                source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
                binary_sha256=hashlib.sha256(Path('build/libexact.so').read_bytes()).hexdigest(),
                scope='Same conversion and native INT8 Tensor Core products. Streamed reconstruction writes every partial into the output; batched reconstruction retains all partials and joins once. Conversion/GEMM/reconstruction are timed together. Extra temporary memory retained explicitly. Every entry matches independent CPU modular arithmetic. No energy claim.')
    Path(args.output).write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
