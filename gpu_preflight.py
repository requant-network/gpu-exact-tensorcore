"""Exercise actual SM120 FP4 and integer GEMMs before the full arithmetic campaign."""
import ctypes,hashlib,json,pathlib
import cupy as cp
import numpy as np

assert __debug__
lib=ctypes.CDLL(str(pathlib.Path('build/libexact.so').resolve()));ptr=ctypes.c_void_p
lib.rq_fp4_create.argtypes=[ctypes.c_int]*3+[ptr]*3;lib.rq_fp4_create.restype=ptr
lib.rq_fp4_run.argtypes=[ptr]*5;lib.rq_fp4_run.restype=ctypes.c_int
lib.rq_fp4_destroy.argtypes=[ptr];lib.rq_error.restype=ctypes.c_char_p
lib.rq_i8_create.restype=ptr;lib.rq_i8_run.argtypes=[ptr]+[ctypes.c_int]*3+[ptr]*4;lib.rq_i8_run.restype=ctypes.c_int
lib.rq_i8_destroy.argtypes=[ptr]
handle=lib.rq_i8_create();assert handle;cases=[]
for m,n,k in ((128,128,128),(256,128,256)):
    a=cp.full(m*k//2,0x11,cp.uint8);b=cp.full(n*k//2,0x11,cp.uint8);out=cp.zeros((m,n),cp.float32)
    plan=lib.rq_fp4_create(m,n,k,a.data.ptr,b.data.ptr,out.data.ptr);assert plan,lib.rq_error()
    status=lib.rq_fp4_run(plan,a.data.ptr,b.data.ptr,out.data.ptr,cp.cuda.get_current_stream().ptr);assert status==0,lib.rq_error()
    cp.cuda.get_current_stream().synchronize();assert np.all(cp.asnumpy(out)==k)
    lib.rq_fp4_destroy(plan)
    ia=cp.ones((m,k),cp.int8);ib=cp.ones((n,k),cp.int8);io=cp.zeros((m,n),cp.int32)
    assert lib.rq_i8_run(handle,m,n,k,ia.data.ptr,ib.data.ptr,io.data.ptr,cp.cuda.get_current_stream().ptr)==0
    cp.cuda.get_current_stream().synchronize();assert np.all(cp.asnumpy(io)==k)
    cases.append(dict(m=m,n=n,k=k,fp4_and_int8_exact=True,outputs=m*n))
lib.rq_i8_destroy(handle)
record=dict(cases=cases,source_sha256={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in ('gpu_preflight.py','cuda/fp4_bridge.cu')},
    binary_sha256=hashlib.sha256(pathlib.Path('build/libexact.so').read_bytes()).hexdigest(),scope='Real native SM120 FP4/INT8 executions with known exact outputs. Correctness preflight, not a performance result.')
pathlib.Path('results').mkdir(exist_ok=True);pathlib.Path('results/gpu-preflight.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
print('Native FP4 and INT8 preflight passed',sum(c['outputs'] for c in cases),'output entries per path')
