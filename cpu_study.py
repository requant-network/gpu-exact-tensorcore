"""Check lossless encodings, accumulation bounds and exact modular reconstruction."""
import argparse,hashlib,json,pathlib,random
from experiments.exact import FP4_DIGITS,bound,digits,fp4_code,multiply,reconstruct

assert __debug__
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='results/cpu-study.json');args=parser.parse_args()
rng=random.Random(20261109);checks=0;matrices=0
for base in (13,128,256):
    for value in list(range(-65536,65536))+[rng.randrange(-2**31,2**31) for _ in range(4096)]:
        d=digits(value,base);assert reconstruct(d,base)==value
        assert all(x in FP4_DIGITS for x in d) if base==13 else all(-base//2<=x<base//2 for x in d)
        if base==13:assert all(0<=fp4_code(x)<16 for x in d)
        checks+=1
for p in (13,257,65521,2013265921):
    for m,k,n in ((1,1,1),(3,7,5),(8,16,8)):
        a=[[rng.randrange(p) for _ in range(k)] for _ in range(m)];b=[[rng.randrange(p) for _ in range(n)] for _ in range(k)]
        expected=multiply(a,b,p)
        for base in (13,128,256):
            da=[[digits(x if x<=p//2 else x-p,base) for x in row] for row in a]
            db=[[digits(x if x<=p//2 else x-p,base) for x in row] for row in b]
            actual=[[sum(reconstruct(x,base)*reconstruct(y,base) for x,y in zip(row,column))%p for column in zip(*db)] for row in da]
            assert actual==expected;matrices+=1
assert bound(16384,13)<2**24 and bound(16384,256)<2**31
record=dict(encoding_values=checks,modular_products=matrices,fields=[13,257,65521,2013265921],
    fp4_contract='Digit d is stored as d/2 with scale 1; FP32 GEMM output multiplied by 4 restores integer limb dot products. k*64 < 2^24 bounds all partial integer sums; split k when necessary.',
    int8_contract='Balanced signed base-128/base-256 limbs; k*max_digit^2 < 2^31 prevents signed accumulator overflow. Reconstruct modulo p without unbounded GPU integer sums.',
    source_sha256={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in ('cpu_study.py','experiments/exact.py')})
path=pathlib.Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
print(checks,'signed encodings;',matrices,'exact modular products passed')
