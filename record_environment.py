"""Record hardware/toolchain provenance without host names, credentials or local network details."""
import argparse
from datetime import datetime,timezone
import hashlib,importlib.metadata,json,os,pathlib,platform,subprocess
import cupy as cp
from cupy_backends.cuda.libs import nvrtc
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--nvcc',default='nvcc');args=parser.parse_args()
props=cp.cuda.runtime.getDeviceProperties(0)
keys=['totalGlobalMem','major','minor','multiProcessorCount','maxThreadsPerBlock','maxThreadsPerMultiProcessor','sharedMemPerBlock','sharedMemPerMultiprocessor']
record={key:int(props[key]) for key in keys if key in props}
record['gpu_name']=props['name'].decode();record['shared_mem_optin_bytes']=cp.cuda.runtime.deviceGetAttribute(97,0)
record['cpu']=next(line.split(':',1)[1].strip() for line in pathlib.Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
record.update(utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),os=platform.system(),kernel=platform.release(),architecture=platform.machine(),runtime=cp.cuda.runtime.runtimeGetVersion(),driver=cp.cuda.runtime.driverGetVersion(),nvrtc=nvrtc.getVersion(),logical_cpu_count=os.cpu_count())
record['packages']={}
for name in ('numpy','cupy-cuda12x','nvidia-cuda-nvrtc-cu12','nvidia-cuda-runtime-cu12','nvidia-cublas-cu12'):
    try:record['packages'][name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:pass
query='name,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.mem,pcie.link.gen.max,pcie.link.width.max'
record['nvidia_smi_columns']=query.split(',')
record['nvidia_smi_values']=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu='+query,'--format=csv,noheader'],text=True).strip().split(', ')
record['nvcc_version']=subprocess.check_output([args.nvcc,'--version'],text=True)
record['gxx_version']=subprocess.check_output(['g++','--version'],text=True).splitlines()[0]
record['thread_environment']={name:os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')}
record['free_and_total_device_bytes_at_collection']=cp.cuda.runtime.memGetInfo()
record['power_clock_changes']='None requested or performed; no energy efficiency conclusion.'
record['source_sha256']={'record_environment.py':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
pathlib.Path('results/environment.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
print('Recorded hardware/toolchain provenance for',record['gpu_name'])
