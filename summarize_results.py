"""Derive reproducible per-input-cohort timing tables from retained observations."""
import csv,json,pathlib,statistics

rows=[]
for path in sorted(pathlib.Path('results').glob('gpu-study-*.json')):
    record=json.loads(path.read_text());assert not record['validation_only']
    for case in record['cases']:
        row={key:case[key] for key in ('m','n','k','p','mode','limbs','setup_wall_ms')};row['seed']=record['seed']
        for key in ('cpu_oracle_samples_ms','resident_direct_ms','resident_graph_ms','transfer_inclusive_wall_ms'):
            values=case.get(key,[])
            if values:
                assert len(values)==30,(path,key,len(values));ordered=sorted(values)
                row[key+'_median']=statistics.median(values);row[key+'_min']=min(values);row[key+'_max']=max(values)
                at=.95*(len(values)-1);lo=int(at);row[key+'_p95']=ordered[lo]+(at-lo)*(ordered[min(lo+1,len(values)-1)]-ordered[lo])
            else:
                for stat in ('median','min','max','p95'):row[key+'_'+stat]=''
        rows.append(row)
assert rows,'Run both performance cohorts before summarizing'
assert {row['seed'] for row in rows}=={20261109,20261209}
for row in rows:
    matching=[r for r in rows if all(r[key]==row[key] for key in ('seed','m','n','k','p'))]
    simt=next(r for r in matching if r['mode']=='simt')
    baseline=simt['resident_graph_ms_median'] or simt['resident_direct_ms_median']
    measured=row['resident_graph_ms_median'] or row['resident_direct_ms_median']
    row['resident_speedup_vs_naive_simt']=baseline/measured
    row['resident_speedup_vs_cpu_reference']=row['cpu_oracle_samples_ms_median']/measured
with pathlib.Path('results/summary.csv').open('w',encoding='utf-8',newline='') as output:
    writer=csv.DictWriter(output,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
print(len(rows),'per-cohort rows written; repeated observations are not treated as independent inputs')
