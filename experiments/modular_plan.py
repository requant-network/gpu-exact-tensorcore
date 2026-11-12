"""Fixed-shape research plan shared by reuse and modular-join experiments."""
import ctypes
from pathlib import Path
import cupy as cp
import numpy as np
from experiments.exact import bound


class Plan:
    def __init__(self,a,bt,p,mode='int8',batched=False):
        assert p in (13,65521,2013265921) and mode in ('int8','fp4')
        assert not batched or mode=='int8'
        self.a,self.bt,self.p,self.mode,self.batched=a,bt,p,mode,batched
        self.m,self.k=a.shape;self.n=bt.shape[0];assert bt.shape[1]==self.k
        self.base=13 if mode=='fp4' else 256;maximum=p//2;self.limbs=0
        while maximum:self.limbs+=1;maximum=(maximum+(8 if self.base==13 else 128))//self.base
        if mode=='fp4' and p==13:self.limbs=1
        assert bound(self.k,self.base)<(2**24 if mode=='fp4' else 2**31)
        self.stream=cp.cuda.get_current_stream();self.out=cp.empty((self.m,self.n),cp.int64)
        unit=2 if mode=='fp4' else 1;dtype=cp.uint8 if unit==2 else cp.int8
        self.ea=cp.empty((self.limbs,self.m*self.k//unit),dtype)
        self.eb=cp.empty((self.limbs,self.n*self.k//unit),dtype)
        count=self.limbs**2 if batched else 1
        self.partial=cp.empty((count,self.m,self.n),cp.float32 if mode=='fp4' else cp.int32)
        self.weights=[pow(self.base,i,p) for i in range(2*self.limbs-1)];self.dw=cp.asarray(self.weights,dtype=cp.int64)
        self.library=ctypes.CDLL(str(Path('build/libexact.so').resolve()));ptr=ctypes.c_void_p
        lib=self.library
        lib.rq_fp4_create.argtypes=[ctypes.c_int]*3+[ptr]*3;lib.rq_fp4_create.restype=ptr
        lib.rq_fp4_run.argtypes=[ptr]*5;lib.rq_fp4_run.restype=ctypes.c_int;lib.rq_fp4_destroy.argtypes=[ptr]
        lib.rq_i8_create.restype=ptr;lib.rq_i8_run.argtypes=[ptr]+[ctypes.c_int]*3+[ptr]*4;lib.rq_i8_run.restype=ctypes.c_int
        lib.rq_i8_destroy.argtypes=[ptr];lib.rq_error.restype=ctypes.c_char_p
        self.handle=lib.rq_fp4_create(self.m,self.n,self.k,self.ea.data.ptr,self.eb.data.ptr,self.partial.data.ptr) if mode=='fp4' else lib.rq_i8_create()
        assert self.handle,lib.rq_error()
        self.module=cp.RawModule(code=Path('cuda/modular.cu').read_text()+Path('cuda/joins.cu').read_text(),options=('--std=c++17',))
        self.kernels={name:self.module.get_function(name) for name in ('encode_i8','encode_fp4','reduce_i8','reduce_fp4','join_i8')}
        self.encode(self.a,self.ea);self.stream.synchronize()

    def encode(self,source,target):
        fp=self.mode=='fp4';size=source.size;unit=2 if fp else 1
        args=(source,target,np.int32(size),np.int32(self.limbs))
        if not fp:args+=(np.int32(256),)
        self.kernels['encode_fp4' if fp else 'encode_i8'](((size//unit+255)//256,),(256,),args)

    def run(self,reuse_a=False):
        if not reuse_a:self.encode(self.a,self.ea)
        self.encode(self.bt,self.eb)
        if not self.batched:self.out.fill(0)
        for i in range(self.limbs):
            for j in range(self.limbs):
                partial=self.partial[i*self.limbs+j if self.batched else 0]
                if self.mode=='fp4':
                    status=self.library.rq_fp4_run(self.handle,self.ea[i].data.ptr,self.eb[j].data.ptr,partial.data.ptr,self.stream.ptr)
                else:
                    status=self.library.rq_i8_run(self.handle,self.m,self.n,self.k,self.ea[i].data.ptr,self.eb[j].data.ptr,partial.data.ptr,self.stream.ptr)
                assert status==0,self.library.rq_error()
                if not self.batched:
                    name='reduce_fp4' if self.mode=='fp4' else 'reduce_i8'
                    self.kernels[name](((self.out.size+255)//256,),(256,),
                         (partial,self.out,np.int32(self.out.size),np.int64(self.p),np.int64(self.weights[i+j])))
        if self.batched:
            self.kernels['join_i8'](((self.out.size+255)//256,),(256,),
                 (self.partial,self.out,self.dw,np.int32(self.out.size),np.int32(self.limbs),np.int64(self.p)))
        return self.out

    def close(self):
        if self.handle:
            (self.library.rq_fp4_destroy if self.mode=='fp4' else self.library.rq_i8_destroy)(self.handle)
            self.handle=None
