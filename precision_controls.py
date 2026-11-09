"""Retain actual unsafe accumulator failures and exact chunked recoveries."""
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import cupy as cp
import numpy as np


def main():
    assert __debug__
    lib=ctypes.CDLL(str(Path('build/libexact.so').resolve()));ptr=ctypes.c_void_p
    lib.rq_fp4_create.argtypes=[ctypes.c_int]*3+[ptr]*3;lib.rq_fp4_create.restype=ptr
    lib.rq_fp4_run.argtypes=[ptr]*5;lib.rq_fp4_run.restype=ctypes.c_int
    lib.rq_fp4_destroy.argtypes=[ptr]
    lib.rq_i8_create.restype=ptr;lib.rq_i8_run.argtypes=[ptr]+[ctypes.c_int]*3+[ptr]*4;lib.rq_i8_run.restype=ctypes.c_int
    lib.rq_i8_destroy.argtypes=[ptr];lib.rq_error.restype=ctypes.c_char_p
    handle=lib.rq_i8_create();assert handle
    stream=cp.cuda.get_current_stream();cases=[];p=2013265921;m=n=128
    for mode,k,chunk in (('fp4',1048704,32768),('int8',131200,65536)):
        fp=mode=='fp4';unit=2 if fp else 1
        a=cp.full((m,k//unit),0x66 if fp else -128,cp.uint8 if fp else cp.int8);b=a.copy()
        expected=np.full((m,n),k*(64 if fp else 16384),np.int64)
        if fp:
            a[0,0]=0x61;b[0,0]=0x61
            expected[0,:]-=56;expected[:,0]-=56;expected[0,0]+=49
        partial=cp.empty((m,n),cp.float32 if fp else cp.int32)

        def product(aa,bb,length):
            if fp:
                plan=lib.rq_fp4_create(m,n,length,aa.data.ptr,bb.data.ptr,partial.data.ptr);assert plan,lib.rq_error()
                status=lib.rq_fp4_run(plan,aa.data.ptr,bb.data.ptr,partial.data.ptr,stream.ptr)
                stream.synchronize();lib.rq_fp4_destroy(plan)
            else:
                status=lib.rq_i8_run(handle,m,n,length,aa.data.ptr,bb.data.ptr,partial.data.ptr,stream.ptr)
                stream.synchronize()
            assert status==0,(mode,status)

        product(a,b,k);unsafe=cp.asnumpy(partial).astype(np.int64)
        mismatches=int(np.count_nonzero(unsafe!=expected))
        assert mismatches>0, 'The deliberately unsafe arithmetic must exhibit a measured failure'
        reduced=cp.zeros((m,n),cp.int64)
        for start in range(0,k,chunk):
            length=min(chunk,k-start)
            aa=cp.ascontiguousarray(a[:,start//unit:(start+length)//unit])
            bb=cp.ascontiguousarray(b[:,start//unit:(start+length)//unit])
            product(aa,bb,length)
            reduced=(reduced+partial.astype(cp.int64)%p)%p
        actual=cp.asnumpy(reduced)
        assert np.array_equal(actual,expected%p)
        cases.append(dict(mode=mode,m=m,n=n,k=k,p=p,chunk=chunk,unsafe_mismatches=mismatches,
            first_unsafe=int(unsafe[0,0]),first_expected=int(expected[0,0]),chunked_exact_outputs=actual.size,
            output_sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
        print('PASS',mode,'retained',mismatches,'unsafe failures; chunked recovery exact',flush=True)
    lib.rq_i8_destroy(handle)
    record=dict(utc=datetime.now(timezone.utc).isoformat(),cases=cases,performance_claim=False,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['precision_controls.py','cuda/fp4_bridge.cu']},
        binary_sha256=hashlib.sha256(Path('build/libexact.so').read_bytes()).hexdigest())
    Path('results/precision-controls.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':main()
