"""Full modular GEMM: GPU encodings, exact native partial GEMMs and modular joins."""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import cupy as cp
import numpy as np
from experiments.exact import multiply, bound


def oracle(a, bt, p):
    """Four exact bounded float64 integer GEMMs, not an overflowing int64 GEMM."""
    k = a.shape[1]
    assert k * 65535**2 < 2**53
    aa, bb = a % p, bt % p
    result = np.zeros((a.shape[0],bt.shape[0]),np.int64)
    for i in range(2):
        for j in range(2):
            x = ((aa >> (16*i)) & 65535).astype(np.float64)
            y = ((bb >> (16*j)) & 65535).astype(np.float64)
            partial = x @ y.T
            assert np.all(partial == np.rint(partial))
            result = (result + (partial.astype(np.int64) % p) * pow(65536,i+j,p)) % p
    return result


def main():
    assert __debug__
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=20261109)
    parser.add_argument('--validation-only',action='store_true')
    parser.add_argument('--output',default='results/gpu-study.json')
    args = parser.parse_args()
    library = ctypes.CDLL(str(Path('build/libexact.so').resolve())); ptr = ctypes.c_void_p
    library.rq_fp4_create.argtypes=[ctypes.c_int]*3+[ptr]*3; library.rq_fp4_create.restype=ptr
    library.rq_fp4_run.argtypes=[ptr]*5; library.rq_fp4_run.restype=ctypes.c_int
    library.rq_fp4_destroy.argtypes=[ptr]; library.rq_error.restype=ctypes.c_char_p
    library.rq_i8_create.restype=ptr; library.rq_i8_run.argtypes=[ptr]+[ctypes.c_int]*3+[ptr]*4; library.rq_i8_run.restype=ctypes.c_int
    library.rq_i8_destroy.argtypes=[ptr]
    handle=library.rq_i8_create(); assert handle
    module=cp.RawModule(code=Path('cuda/modular.cu').read_text(),options=('--std=c++17',))
    kernels={name:module.get_function(name) for name in ('encode_i8','encode_fp4','reduce_i8','reduce_fp4','simt')}
    rng=np.random.default_rng(args.seed); cases=[]
    # Check the optimized CPU oracle against arbitrary-precision arithmetic first.
    for p in (257,65521,2013265921):
        a=rng.integers(-p//2,p//2+1,(5,19),np.int64);bt=rng.integers(-p//2,p//2+1,(7,19),np.int64)
        assert np.array_equal(oracle(a,bt,p),np.array(multiply(a.tolist(),bt.T.tolist(),p),np.int64))
    stream=cp.cuda.Stream(non_blocking=True)
    stream.use()

    def samples(function):
        for _ in range(5):function()
        stream.synchronize()
        result=[]
        for _ in range(30):
            start,end=cp.cuda.Event(),cp.cuda.Event();start.record();function();end.record();end.synchronize()
            result.append(cp.cuda.get_elapsed_time(start,end))
        return result

    for m,n,k in ((128,128,128),(256,256,256),(512,256,512)):
        for p in (257,65521,2013265921):
            half=p//2;a=rng.integers(-half,half+1,(m,k),np.int64);bt=rng.integers(-half,half+1,(n,k),np.int64)
            a[0,:]=np.resize(np.array([-half,half,0,1,-1],np.int64),k)
            bt[0,:]=np.resize(np.array([half,-half,1,0,-1],np.int64),k)
            started=time.perf_counter();expected=oracle(a,bt,p);cpu_ms=(time.perf_counter()-started)*1000
            da,db=cp.asarray(a),cp.asarray(bt);out=cp.zeros((m,n),cp.int64)
            input_hash=hashlib.sha256(a.tobytes()+bt.tobytes()).hexdigest()
            cpu_samples=[]
            if not args.validation_only:
                for _ in range(5):oracle(a,bt,p)
                for _ in range(30):
                    start=time.perf_counter();result=oracle(a,bt,p);cpu_samples.append((time.perf_counter()-start)*1000)
                    assert np.array_equal(result,expected)
            for mode,base in (('simt',0),('int8',256),('int8-base128',128),('fp4',13)):
                setup=time.perf_counter();plan=None
                if base:
                    maximum=half;limbs=0
                    while maximum:
                        limbs+=1;maximum=(maximum+(8 if base==13 else base//2))//base
                    assert bound(k,base)<(2**24 if base==13 else 2**31)
                    unit=2 if base==13 else 1
                    ea=cp.empty((limbs,m*k//unit),cp.uint8 if unit==2 else cp.int8)
                    eb=cp.empty((limbs,n*k//unit),ea.dtype)
                    partial=cp.empty((m,n),cp.float32 if unit==2 else cp.int32)
                    if base==13:
                        plan=library.rq_fp4_create(m,n,k,ea.data.ptr,eb.data.ptr,partial.data.ptr);assert plan,library.rq_error()
                    weights=[pow(base,i,p) for i in range(2*limbs-1)]
                stream.synchronize();setup_ms=(time.perf_counter()-setup)*1000

                def run():
                    if not base:
                        kernels['simt'](((m*n+127)//128,),(128,),(da,db,out,np.int32(m),np.int32(n),np.int32(k),np.int64(p)))
                        return
                    fp=base==13;name='encode_fp4' if fp else 'encode_i8'
                    for original,encoded,size in ((da,ea,m*k),(db,eb,n*k)):
                        arguments=(original,encoded,np.int32(size),np.int32(limbs))
                        if not fp:arguments+=(np.int32(base),)
                        kernels[name](((size//unit+255)//256,),(256,),arguments)
                    out.fill(0)
                    for i in range(limbs):
                        for j in range(limbs):
                            if fp:
                                status=library.rq_fp4_run(plan,ea[i].data.ptr,eb[j].data.ptr,partial.data.ptr,stream.ptr)
                            else:
                                status=library.rq_i8_run(handle,m,n,k,ea[i].data.ptr,eb[j].data.ptr,partial.data.ptr,stream.ptr)
                            assert status==0,(mode,status,library.rq_error())
                            kernels['reduce_fp4' if fp else 'reduce_i8'](((m*n+255)//256,),(256,),
                                (partial,out,np.int32(m*n),np.int64(p),np.int64(weights[i+j])))

                run();stream.synchronize();actual=cp.asnumpy(out)
                assert np.array_equal(actual,expected),(m,n,k,p,mode,np.flatnonzero(actual!=expected)[:4])
                record=dict(m=m,n=n,k=k,p=p,mode=mode,limbs=limbs if base else 0,exact_outputs=actual.size,
                    input_sha256=input_hash,output_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),setup_wall_ms=setup_ms,
                    cpu_oracle_wall_ms=cpu_ms,cpu_oracle_samples_ms=cpu_samples)
                if not args.validation_only:
                    record['resident_direct_ms']=samples(run)
                    try:
                        stream.begin_capture();run();graph=stream.end_capture()
                    except Exception as error:
                        # A runtime may reject capture; preserve the limitation, never infer a graph result.
                        if stream.is_capturing():stream.end_capture()
                        graph=None;record['graph_error']=str(error)
                    if graph is not None:
                        record['resident_graph_ms']=samples(lambda:graph.launch(stream))
                    transfer=[]
                    for _ in range(5):
                        da.set(a);db.set(bt);run();out.get();stream.synchronize()
                    for _ in range(30):
                        start=time.perf_counter();da.set(a);db.set(bt);run();result=out.get();stream.synchronize()
                        transfer.append((time.perf_counter()-start)*1000)
                        assert np.array_equal(result,expected)
                    record['transfer_inclusive_wall_ms']=transfer
                cases.append(record);print('PASS',m,n,k,p,mode,flush=True)
                if plan:library.rq_fp4_destroy(plan)
    library.rq_i8_destroy(handle)
    files=['gpu_study.py','experiments/exact.py','cuda/modular.cu','cuda/fp4_bridge.cu']
    record=dict(utc=datetime.now(timezone.utc).isoformat(),seed=args.seed,cases=cases,validation_only=args.validation_only,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
        binary_sha256=hashlib.sha256(Path('build/libexact.so').read_bytes()).hexdigest(),
        notes='Resident includes GPU conversion, all limb GEMMs and modular reduction. Transfer-inclusive adds H2D input and D2H output, with persistent allocations. Setup separately. CPU oracle uses bounded float64 integer products, checked against arbitrary precision. No energy claim.')
    Path(args.output).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':main()
