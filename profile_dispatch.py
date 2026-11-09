"""Capture native preflight CUDA kernel names; profiled durations are not benchmark results."""
import csv,hashlib,json,pathlib,subprocess,sys
from datetime import datetime,timezone

assert __debug__
output=pathlib.Path('build/dispatch-trace.nsys-rep');assert not output.exists(),'Preserve previous traces'
command=['nsys','profile','--trace=cuda,cublas','--sample=none','--cpuctxsw=none','-o','build/dispatch-trace',sys.executable,'gpu_preflight.py']
with pathlib.Path('build/dispatch-profile.log').open('x') as log:
    profile=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
assert profile.returncode==0, 'Read the retained private profiling log; do not bypass profiling permissions'
stats=subprocess.run(['nsys','stats','--report','cuda_gpu_kern_sum','--format','csv',str(output)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
assert stats.returncode==0,stats.stderr
lines=stats.stdout.splitlines();header=next(i for i,line in enumerate(lines) if 'Time (%)' in line and 'Name' in line)
reader=csv.reader(lines[header:]);names=next(reader);kernels=[]
for row in reader:
    if len(row)!=len(names):continue
    try:float(row[0])
    except ValueError:continue
    value=dict(zip(names,row));kernels.append(dict(name=value['Name'],instances=int(value.get('Instances',value.get('Num Instances','0')))))
assert kernels,'No captured CUDA kernels'
files=['profile_dispatch.py','gpu_preflight.py','cuda/fp4_bridge.cu']
record=dict(utc=datetime.now(timezone.utc).isoformat(),profile_exit_code=profile.returncode,kernels=kernels,
    profiler_version=subprocess.check_output(['nsys','--version'],text=True).strip(),
    command=[v if v!=sys.executable else 'PYTHON' for v in command],
    source_sha256={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in files},
    binary_sha256=hashlib.sha256(pathlib.Path('build/libexact.so').read_bytes()).hexdigest(),
    performance_claim=False,scope='CUDA activity names from the unchanged two-shape native correctness preflight. Kernel-family evidence only; profiled durations excluded. Full raw trace stays private because it can contain host/process paths.')
pathlib.Path('results/dispatch-profile.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
print('Captured',len(kernels),'native CUDA kernel names')
for item in kernels:print(item['name'][:250])
