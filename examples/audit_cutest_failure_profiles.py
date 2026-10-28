"""
Audit small per-run solve-summary CSVs for STORM/CUTEst performance profiles.

Usage:
    python audit_cutest_failure_profiles.py path/to/all_solve_info.csv \
        --success-col smooth_tau2 --out-prefix cutest_failure_invdelta

Expected minimal columns:
    probname, key, nrun, smooth_tau2
where smooth_tau2 >= 0 is the 0-based index of the first successful
objective evaluation and -1 indicates failure. Rename with CLI options.

This script audits/profile-reconstructs only; it does NOT alter raw results.
"""
import argparse
import numpy as np
import pandas as pd


def performance_summary(df, group_columns, values, algorithms, cutoffs=(1, 2, 4, 8, 16, 32)):
    base = values.groupby([df[c] for c in group_columns]).transform('min')
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = values / base
    table = []
    for a in algorithms:
        is_a = df['_algorithm'] == a
        group_ratios = ratio[is_a]
        row = {
            'algorithm': a,
            'observations': int(is_a.sum()),
            'solved': int(np.isfinite(values[is_a]).sum()),
            'undefined_ratios_NaN': int(np.isnan(group_ratios).sum()),
            'infinite_ratios': int(np.isinf(group_ratios).sum()),
        }
        for cutoff in cutoffs:
            row[f'profile_at_{cutoff}'] = float((group_ratios <= cutoff).sum() / is_a.sum())
        table.append(row)
    return pd.DataFrame(table), ratio, base


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('csv', help='Small exported solve-info CSV; not the large raw JSON files')
    p.add_argument('--success-col', default='smooth_tau2')
    p.add_argument('--problem-col', default='probname')
    p.add_argument('--algorithm-col', default='key')
    p.add_argument('--run-col', default='nrun')
    p.add_argument('--out-prefix', default='cutest_failure_audit')
    p.add_argument('--budget-kind', choices=['zero_based_index','evaluation_count','gradient_budget'],
                   default='zero_based_index',
                   help='zero_based_index: old plot arrays; evaluation_count: 1-based evaluations;\n'
                        'gradient_budget: k/(n+1) for 0-based index k; requires --n-col')
    p.add_argument('--n-col', default='n', help='Dimension column for gradient_budget conversion')
    args = p.parse_args()

    cols = [args.problem_col, args.algorithm_col, args.run_col, args.success_col]
    df = pd.read_csv(args.csv, low_memory=False)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f'Missing columns {missing}; available: {list(df.columns)}')
    df = df.copy()
    df['_problem'] = df[args.problem_col].astype(str)
    df['_algorithm'] = df[args.algorithm_col].astype(str)
    df['_run'] = df[args.run_col].astype(str)
    df['_t'] = pd.to_numeric(df[args.success_col], errors='coerce')

    is_success = np.isfinite(df['_t']) & (df['_t'] >= 0)
    is_zero = is_success & (df['_t'] == 0)
    is_failed = np.isfinite(df['_t']) & (df['_t'] < 0)
    is_invalid = ~np.isfinite(df['_t'])

    print('\n=== DATASET ===')
    print('Total rows:', len(df))
    print('Problems:', df['_problem'].nunique(), '| Algorithms:', df['_algorithm'].nunique(),
          '| Unique run labels:', df['_run'].nunique())
    print('Raw first-success values: success >=0:', int(is_success.sum()),
          '| 0:', int(is_zero.sum()), '| failure <0:', int(is_failed.sum()),
          '| NaN/inf:', int(is_invalid.sum()))
    counts = df.groupby(['_problem','_algorithm'])['_run'].agg(['count','nunique'])
    print('Run count per problem/algorithm: min =', int(counts['count'].min()),
          ', max =', int(counts['count'].max()))
    dups = int(df.duplicated(['_problem','_algorithm','_run']).sum())
    print('Duplicate (problem, algorithm, run) rows:', dups)
    if dups:
        print('WARNING: Duplicates must be resolved before interpreting profiles')

    print('\n=== SUCCESS COUNTS BY ALGORITHM ===')
    algo_rows = []
    for algo, part in df.groupby('_algorithm', sort=True):
        t = part['_t']
        algo_rows.append(dict(algorithm=algo, n=len(part), success=int(((t>=0) & np.isfinite(t)).sum()),
                              zero=int((t==0).sum()), positive=int(((t>0) & np.isfinite(t)).sum()),
                              failed=int((t<0).sum()), invalid=int((~np.isfinite(t)).sum())))
    by_alg = pd.DataFrame(algo_rows)
    print(by_alg.to_string(index=False))

    prob = df.groupby('_problem').agg(
        n_observations=('_t','size'),
        valid_successes=('_t',lambda x: int(((x>=0) & np.isfinite(x)).sum())),
        zero_cost_successes=('_t',lambda x: int((x==0).sum())),
        minimum_success_time=('_t',lambda x: x[(x>=0) & np.isfinite(x)].min() if ((x>=0) & np.isfinite(x)).any() else np.nan),
    ).reset_index().rename(columns={'_problem':args.problem_col})
    prob['no_success_from_any_algorithm'] = prob.valid_successes == 0
    prob['all_detected_successes_at_zero'] = (prob.valid_successes > 0) & (prob.valid_successes == prob.zero_cost_successes)
    print('\n=== PROBLEM-LEVEL WARNINGS ===')
    print('Problems with NO successful run:', int(prob.no_success_from_any_algorithm.sum()))
    print('Problems whose ALL detected successes have zero time:', int(prob.all_detected_successes_at_zero.sum()))
    print('Problems with >= 1 successful run:', int((prob.valid_successes > 0).sum()))
    print('Names of up to 20 problems with NO successes:',
          prob.loc[prob.no_success_from_any_algorithm,args.problem_col].head(20).tolist())

    # Legacy as-is computation; 0/0 -> NaN for a problem with zero best budget.
    raw_cost = df['_t'].where(is_success, np.inf).astype(float)
    # Convert 0-based indices to *positive* call costs for a nonzero denominator.
    corrected_cost = raw_cost.copy()
    if args.budget_kind == 'zero_based_index':
        corrected_cost[is_success] = corrected_cost[is_success] + 1.0
    elif args.budget_kind == 'gradient_budget':
        if args.n_col not in df.columns:
            raise SystemExit(f'For gradient_budget, require dimension column {args.n_col!r}; available: {list(df.columns)}')
        dims = pd.to_numeric(df[args.n_col], errors='coerce')
        if not np.all(np.isfinite(dims[is_success]) & (dims[is_success] > 0)):
            raise SystemExit('Dimension n must be a positive finite number for each solved observation')
        corrected_cost[is_success] = corrected_cost[is_success] + 1.0 / (dims[is_success] + 1.0)
    elif args.budget_kind == 'evaluation_count':
        if is_zero.any():
            print('WARNING: Zero solve counts despite positive evaluation-count convention;')
            print('setting them to 1 temporarily only for the reconstructed audit.')
            corrected_cost[is_zero] = 1.0

    algos = sorted(df['_algorithm'].unique())
    for scope_name, scope_cols in [('matched_problem_and_run',['_problem','_run']),
                                   ('problem_only',['_problem'])]:
        raw_summary, raw_ratio, raw_base = performance_summary(df, scope_cols, raw_cost, algos)
        fixed_summary, fixed_ratio, fixed_base = performance_summary(df, scope_cols, corrected_cost, algos)
        print(f'\n=== PERFORMANCE PROFILE: {scope_name} ===')
        print('Using RAW success times:')
        print(raw_summary.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
        print('SUM(profile@1)=', round(raw_summary.profile_at_1.sum(), 6),
              '| SUM(profile@32)=', round(raw_summary.profile_at_32.sum(),6))
        print('Using positive evaluation counts (diagnostic correction):')
        print(fixed_summary.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
        print('SUM(profile@1)=', round(fixed_summary.profile_at_1.sum(), 6),
              '| SUM(profile@32)=', round(fixed_summary.profile_at_32.sum(),6))
        if scope_name == 'matched_problem_and_run':
            df['raw_ratio_matched'] = raw_ratio
            df['corrected_ratio_matched'] = fixed_ratio
        else:
            df['raw_ratio_problem'] = raw_ratio
            df['corrected_ratio_problem'] = fixed_ratio

    print('\n=== THEORETICAL CROSS-CHECK ===')
    nruns = df.groupby(['_problem','_algorithm'])['_run'].nunique()
    if len(nruns)>0 and nruns.min()==nruns.max() and dups==0:
        R = int(nruns.iloc[0])
        P = int(df['_problem'].nunique())
        print('Common number of runs per problem and algorithm:',R)
        print(f'Expected sum of profiles >= 1/{R} = {1/R:.6f} at ratio 1,')
        print('IF the shared fmin comes from these same runs, every problem has at least one',
              'finite successful observation, and all denominators are handled correctly.')
        if int(prob.no_success_from_any_algorithm.sum()):
            print('IMPORTANT: Some problems have NO success. Check fmin/target consistency,',
                  'missing/invalid function values, and whether fmin came from external data.')
    else:
        print('Unequal or duplicate run coverage: simple 1/R bound is not directly applicable.')
    print('Note: corrected ratios are an AUDIT reconstruction, not an automatic repair of the published plots.')

    # Reconstruct the OLDER CUTEst convention: take median of 10 runs for each
    # (problem, algorithm), then make profiles over *problems* rather than runs.
    median_df = df.groupby(['_problem','_algorithm'],as_index=False).agg(
        median_raw=('_t', lambda x: np.median(np.where(np.isfinite(x) & (x>=0),x,np.inf))))
    median_df['_median_corrected'] = median_df['median_raw'].copy()
    if args.budget_kind == 'zero_based_index':
        median_df.loc[np.isfinite(median_df['_median_corrected']),'_median_corrected'] += 1.0
    elif args.budget_kind == 'gradient_budget':
        # For an exact median rebuild, convert each per-run budget before taking its median.
        tmp = df.copy()
        tmp['_corrected_cost'] = corrected_cost
        med_corr = tmp.groupby(['_problem','_algorithm'])['_corrected_cost'].median().rename('_median_corrected')
        median_df = median_df.drop(columns=['_median_corrected']).merge(med_corr.reset_index(), on=['_problem','_algorithm'])
    med_rows = []
    for problem, subset in median_df.groupby('_problem'):
        finite_cost = subset['_median_corrected'][np.isfinite(subset['_median_corrected'])]
        best = float(finite_cost.min()) if len(finite_cost) else np.inf
        for i,row in subset.iterrows():
            cost = row['_median_corrected']
            med_rows.append({'_problem':problem, '_algorithm':row['_algorithm'],
                'median_success_cost':cost,
                'median_ratio':(max(1.0,cost / best) if np.isfinite(cost) and np.isfinite(best) and best > 0 else np.inf)})
    med_stats = pd.DataFrame(med_rows)
    med_summary = med_stats.groupby('_algorithm').agg(
        problems=('_problem','nunique'),
        problems_finite_median=('median_success_cost', lambda x: int(np.isfinite(x).sum())),
        profile_at_1=('median_ratio', lambda x: float((x<=1).mean())),
        profile_at_32=('median_ratio', lambda x: float((x<=32).mean()))).reset_index()
    print('\n=== LEGACY CUTEST DEFINITION: MEDIAN OF 10 RUNS PER PROBLEM ===')
    print(med_summary.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print('This statistic has NO 1/R lower-bound: isolated successful runs can have median=inf.')
    med_stats.to_csv(args.out_prefix+'_legacy_median_per_problem.csv',index=False)

    prob_out = args.out_prefix + '_per_problem.csv'
    rows_out = args.out_prefix + '_per_run.csv'
    summary_out = args.out_prefix + '_per_algorithm.csv'
    prob.to_csv(prob_out,index=False)
    df.to_csv(rows_out,index=False)
    by_alg.to_csv(summary_out,index=False)
    print('\nSaved:', prob_out, rows_out, summary_out)

if __name__ == '__main__':
    main()
