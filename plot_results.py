"""Plot per-cohort median ranges; error bars are not confidence intervals."""
import csv,pathlib,statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rows=list(csv.DictReader(open('results/summary.csv',encoding='utf-8')))
fields=(13,257,65521,2013265921);modes=('int8','int8-base128','fp4')
labels=('INT8 / base 256','INT8 / base 128','FP4 / residue or base 13')
colors=('#3a648d','#91a7be','#168c91')
fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout='constrained')
for ax,key,title in zip(axes,('resident_graph_ms_median','transfer_inclusive_wall_ms_median'),('Resident captured pipeline','Host offload with persistent device buffers')):
    for j,(mode,label,color) in enumerate(zip(modes,labels,colors)):
        groups=[[float(r[key]) for r in rows if r['m']=='1024' and int(r['p'])==p and r['mode']==mode] for p in fields]
        assert all(len(v)==2 for v in groups)
        median=[statistics.mean(v) for v in groups];lo=[mid-min(v) for mid,v in zip(median,groups)];hi=[max(v)-mid for mid,v in zip(median,groups)]
        ax.bar([i+(j-1)*.24 for i in range(4)],median,.23,color=color,label=label,yerr=[lo,hi],capsize=2)
    ax.set_xticks(range(4),('13','257','65,521','2,013,265,921'),rotation=14)
    ax.set_title(title,loc='left',fontsize=12);ax.set_xlabel('Modulus');ax.set_ylabel('Milliseconds')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
axes[0].set_yscale('log');axes[0].legend(frameon=False,fontsize=9)
fig.suptitle('Exact modular GEMM on RTX 5060 Ti · 1024×1024×1024',fontsize=16)
fig.supxlabel('Requant Network research · Encoding, every limb GEMM and modular joins included.\nBars: midpoint of two cohort medians; whiskers: their range. Five warmups and thirty observations per cohort.',fontsize=9)
pathlib.Path('figures').mkdir(exist_ok=True)
fig.savefig('figures/modular-gemm.png',dpi=180);fig.savefig('figures/modular-gemm.svg')
svg=pathlib.Path('figures/modular-gemm.svg');svg.write_bytes(svg.read_bytes().replace(b'\r\n',b'\n'))
print('Rendered measured modular-GEMM pipelines')
