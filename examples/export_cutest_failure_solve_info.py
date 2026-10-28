"""
Independent diagnostic export of per-run first-success indices from converted CUTEst JSON.

Targets use f0 and fmin pooled over all aggregators and sample rules for the
specified noise model. Confirm against original plotting code before expecting
exact reproduction of existing charts.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

TAUS = {'tau1':1e-1,'tau2':1e-2,'tau3':1e-3,'tau4':1e-4,'tau5':1e-5}


def finite_min(values):
    a=np.asarray(values, dtype=float)
    a=a[np.isfinite(a)]
    return float(a.min()) if len(a) else np.inf


def first_hit(values, target):
    if not np.isfinite(target):
        return -1
    a=np.asarray(values, dtype=float)
    hits=np.flatnonzero(np.isfinite(a)&(a<=target))
    return int(hits[0]) if len(hits) else -1


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--noise', default='failure_low_p08')
    ap.add_argument('--root', default='converted_results_cutest_parallel')
    ap.add_argument('--outdir', default='audit_results')
    args=ap.parse_args()
    files=sorted(Path(args.root).glob(args.noise+'_*.json'))
    # Expected tree is ROOT/NOISE_AGG_NS/*.json, not ROOT/*.json.
    files=sorted(Path(args.root).glob(args.noise+'_*/*.json'))
    if not files:
        raise SystemExit('No JSON files found under '+str(Path(args.root)/ (args.noise+'_*/')))
    f0_s=defaultdict(list); f0_n=defaultdict(list)
    fmin_s=defaultdict(lambda:np.inf); fmin_n=defaultdict(lambda:np.inf)
    print(f'Pass 1: pooling targets across {len(files)} JSON files',flush=True)
    for idx,p in enumerate(files,1):
        with p.open() as f: d=json.load(f)
        prob=d['problem_name']; runs=d['runs']
        if int(d.get('nruns',len(runs))) != len(runs):
            raise ValueError(f'Inconsistent nruns in {p}')
        for r in runs:
            sv=r['fvals_smooth']; nv=r['fvals_noisy']
            if not sv or not nv:
                raise ValueError(f'Empty values in {p}')
            f0_s[prob].append(float(sv[0])); f0_n[prob].append(float(nv[0]))
        fmin_s[prob]=min(fmin_s[prob],finite_min(d['f_best_smooth_runs']))
        fmin_n[prob]=min(fmin_n[prob],finite_min(d['f_best_noisy_runs']))
        if idx%50==0: print('  first-pass files:',idx,flush=True)
    f0_s={k:float(np.mean(v)) for k,v in f0_s.items()}
    f0_n={k:float(np.mean(v)) for k,v in f0_n.items()}
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    by_rule=defaultdict(list)
    print('Pass 2: reconstructing first successes (0-based)',flush=True)
    for idx,p in enumerate(files,1):
        with p.open() as f: d=json.load(f)
        prob=d['problem_name']; algo=d['aggregator']; rule=d['ns_rule']
        for i,r in enumerate(d['runs']):
            row={'probname':prob,'key':str(algo),'nrun':int(r.get('run',i)),
                 'n':int(d['n']),'noise_model':args.noise,'ns_rule':str(rule),
                 'nf':int(r.get('nf',len(r['fvals_smooth']))),
                 'noisy_neg10000_count':int(np.count_nonzero(np.asarray(r['fvals_noisy'])==-10000.0))}
            for name,tau in TAUS.items():
                row['smooth_'+name]=first_hit(r['fvals_smooth'],fmin_s[prob]+tau*(f0_s[prob]-fmin_s[prob]))
                row['noisy_'+name]=first_hit(r['fvals_noisy'],fmin_n[prob]+tau*(f0_n[prob]-fmin_n[prob]))
            by_rule[str(rule)].append(row)
        if idx%50==0: print('  second-pass files:',idx,flush=True)
    fieldnames=['probname','key','nrun','n','noise_model','ns_rule','nf','noisy_neg10000_count']+[
        k+'_'+name for name in TAUS for k in ('smooth','noisy')]
    for rule,rows in sorted(by_rule.items()):
        output=outdir/f'solve_info_{args.noise}_{rule}.csv'
        with output.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(rows)
        print(f'SAVED {output}: {len(rows)} observations; {len(set(x["probname"] for x in rows))} problems; {len(set(x["key"] for x in rows))} algorithms',flush=True)
    print('NOTE: target f0/fmin is pooled by noise model across rules & aggregators; independently reconstructed, not guaranteed to match plot code.')

if __name__=='__main__':main()
