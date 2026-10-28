#!/usr/bin/env python3
"""Audit the exact transformed data and performance-profile routines in
run_cutest_noise_grid_plots_specific_fmin_fixed.py.

Use this on the machine holding converted_results_cutest_parallel; it only
exports small CSV summaries, never copies entire evaluation histories.
"""
import argparse
import csv
import importlib.util
import math
from collections import defaultdict
from pathlib import Path
import numpy as np


def load_plot_module(path):
    spec = importlib.util.spec_from_file_location('cutest_fixed_for_audit', str(path))
    if not spec or not spec.loader:
        raise RuntimeError(f'Cannot load plotting script {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def same_min(a, b):
    return np.isfinite(a) and np.isfinite(b) and np.isclose(a, b, rtol=1e-12, atol=1e-12)


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--plot-script', default='run_cutest_noise_grid_plots_specific_fmin_fixed.py',
                    help='Path to exact plotting script')
    pa.add_argument('--converted-root', default=None, help='Folder of converted CUTEst JSON results')
    pa.add_argument('--noise', default='failure', help='Substring of noise model name; default failure')
    pa.add_argument('--tau', type=float, default=1e-2)
    pa.add_argument('--profile', choices=['smooth', 'noisy'], default='smooth')
    pa.add_argument('--output-dir', default='cutest_failure_diagnostics')
    args = pa.parse_args()
    mod = load_plot_module(Path(args.plot_script).resolve())
    converted = args.converted_root or mod.CONVERTED_ROOT
    records = mod.load_converted_results(str(converted))
    noises = sorted({str(rec['noise_model']) for rec in records if args.noise.lower() in str(rec['noise_model']).lower()})
    if not noises:
        raise SystemExit(f'No noise model matching {args.noise!r}; available: {sorted({r["noise_model"] for r in records})}')
    fmins = mod.compute_problem_fmin(records, args.profile)
    solve = mod.build_solve_budgets(records, tau=args.tau, profile_type=args.profile)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Which sampling/aggregation generated the shared f_min? Also test if
    # the best value is reachable on the evaluation budget axis.
    winners = defaultdict(list)
    min_by_rule = defaultdict(lambda: math.inf)
    per_run_f0 = defaultdict(list)
    for rec in records:
        noise = str(rec['noise_model'])
        if noise not in noises:
            continue
        p, rule, agg = rec['problem_name'], rec['ns_rule'], rec['aggregator']
        for i, run in enumerate(rec['runs']):
            vals = mod.get_values_from_run(run, args.profile)
            if len(vals) == 0 or not np.isfinite(vals).any():
                continue
            vmin = float(np.min(vals[np.isfinite(vals)]))
            min_by_rule[(noise,p,rule)] = min(min_by_rule[(noise,p,rule)],vmin)
            initial = float(vals[0]) if np.isfinite(vals[0]) else math.nan
            per_run_f0[(noise,p)].append(initial)
            if same_min(vmin, fmins.get((noise,p),math.nan)):
                idx = int(np.flatnonzero(np.isfinite(vals) & np.isclose(vals, vmin, rtol=1e-12, atol=1e-12))[0])
                axis = mod.get_budget_axis_from_run(run)
                budget_valid = idx < len(axis) and np.isfinite(axis[idx])
                winners[(noise,p)].append((rule,agg,i,idx,budget_valid))

    problem_rows = []
    group_rows = []
    for noise in noises:
        rules = sorted({rec['ns_rule'] for rec in records if rec['noise_model'] == noise})
        aggs = sorted({rec['aggregator'] for rec in records if rec['noise_model'] == noise})
        for rule in rules:
            problems = sorted({p for agg in aggs for p in solve.get((noise,rule,agg),{})})
            if not problems:
                continue
            all_counts = {a:dict(success=0, zeros=0, gt32=0, hidden_best0=0, at1=0, at32=0) for a in aggs}
            none_solved = 0
            zero_best_problems = 0
            rule_min_wins = 0
            for p in problems:
                vals = {a: np.asarray(solve.get((noise,rule,a),{}).get(p,[]),dtype=float) for a in aggs}
                all_budgets = np.concatenate(list(vals.values())) if vals else np.array([])
                good = all_budgets[np.isfinite(all_budgets)]
                best = float(np.min(good)) if len(good) else math.inf
                if not np.isfinite(best):
                    none_solved += 1
                if best == 0:
                    zero_best_problems += 1
                if any(w[0] == rule for w in winners.get((noise,p), [])):
                    rule_min_wins += 1
                winner = winners.get((noise,p),[])
                r_min = min_by_rule.get((noise,p,rule),math.inf)
                f0s = np.asarray(per_run_f0.get((noise,p), []),dtype=float)
                problem_row = {
                    'noise':noise, 'ns_rule':rule, 'problem':p,
                    'shared_fmin':fmins.get((noise,p), math.nan),
                    'rule_best_smooth_or_noisy':r_min,
                    'best_f_gap_from_shared':r_min-fmins.get((noise,p),math.nan),
                    'any_run_solved_in_this_rule':int(np.isfinite(best)),
                    'best_solve_budget_this_rule':best,
                    'best_budget_is_zero':int(best == 0),
                    'winning_ns_rules_for_shared_fmin':'|'.join(sorted(set(w[0] for w in winner))),
                    'number_of_winners':len(winner),
                    'winner_with_missing_budget_axis':sum(not w[4] for w in winner),
                    'f0_spread':float(np.ptp(f0s[np.isfinite(f0s)])) if np.isfinite(f0s).any() else math.nan,
                }
                for a in aggs:
                    bs=vals[a]
                    success=np.isfinite(bs)
                    zero=bs == 0
                    if best>0 and np.isfinite(best):
                        ratios=np.where(success, np.maximum(1.0, bs/best), np.inf)
                    elif best==0:
                        ratios=np.where(zero,1.0,np.inf)
                    else:
                        ratios=np.full(len(bs), np.inf)
                    nfin=int(np.count_nonzero(success))
                    nzero=int(np.count_nonzero(zero))
                    ngt32=int(np.count_nonzero(np.isfinite(ratios) & (ratios>32)))
                    hidden=int(np.count_nonzero(success & (bs>0))) if best==0 else 0
                    c=all_counts[a]
                    c['success']+=nfin;c['zeros']+=nzero;c['gt32']+=ngt32;c['hidden_best0']+=hidden
                    c['at1']+=int(np.count_nonzero(ratios <= 1))
                    c['at32']+=int(np.count_nonzero(ratios <= 32))
                    problem_row[a+'_runs']=len(bs)
                    problem_row[a+'_success']=nfin
                    problem_row[a+'_ratio_le_1']=int(np.count_nonzero(ratios<=1))
                    problem_row[a+'_ratio_le_32']=int(np.count_nonzero(ratios<=32))
                    problem_row[a+'_finite_hidden_by_zero_best']=hidden
                problem_rows.append(problem_row)
            for a in aggs:
                c=all_counts[a]
                nobs=sum(len(solve.get((noise,rule,a),{}).get(p,[])) for p in problems)
                group_rows.append({
                    'noise':noise,'ns_rule':rule,'aggregator':a,'problems':len(problems),
                    'observations':nobs,'runs_with_finite_solve_budget':c['success'],
                    'runs_with_zero_solve_budget':c['zeros'],
                    'runs_with_ratio_gt32':c['gt32'],
                    'runs_hidden_by_zero_best':c['hidden_best0'],
                    'PP_at_1':c['at1']/nobs if nobs else math.nan,
                    'PP_at_32':c['at32']/nobs if nobs else math.nan,
                    'problems_no_method_succeeded_in_rule':none_solved,
                    'problems_best_solve_budget_zero':zero_best_problems,
                    'problems_shared_fmin_attained_in_rule':rule_min_wins,
                })
            print(f'noise={noise} ns={rule} problems={len(problems)}, none_solved={none_solved}, '
                  f'rule_won_fmin_on={rule_min_wins}, zero_best={zero_best_problems}; '
                  f'sum PP(1)={sum(x["PP_at_1"] for x in group_rows if x["noise"]==noise and x["ns_rule"]==rule):.4f}')

    problem_cols = ['noise','ns_rule','problem','shared_fmin','rule_best_smooth_or_noisy','best_f_gap_from_shared',
                    'any_run_solved_in_this_rule','best_solve_budget_this_rule','best_budget_is_zero',
                    'winning_ns_rules_for_shared_fmin','number_of_winners','winner_with_missing_budget_axis','f0_spread']
    aggs_all = sorted({rec['aggregator'] for rec in records if rec['noise_model'] in noises})
    for a in aggs_all:
        problem_cols.extend([a+'_runs',a+'_success',a+'_ratio_le_1',a+'_ratio_le_32',a+'_finite_hidden_by_zero_best'])
    group_cols = ['noise','ns_rule','aggregator','problems','observations','runs_with_finite_solve_budget',
                  'runs_with_zero_solve_budget','runs_with_ratio_gt32','runs_hidden_by_zero_best',
                  'PP_at_1','PP_at_32','problems_no_method_succeeded_in_rule',
                  'problems_best_solve_budget_zero','problems_shared_fmin_attained_in_rule']
    write_csv(out/'by_problem_and_sampling.csv', problem_rows, problem_cols)
    write_csv(out/'by_algorithm_summary.csv', group_rows, group_cols)
    print(f'\nSaved {out/"by_problem_and_sampling.csv"} and {out/"by_algorithm_summary.csv"}')
    print('Interpretation: zero-solve problems in one sampling rule may have shared f_min from another rule.')
    print('Check the two CSVs for fmin winners, failed solve budgets, and best=0 effects.')


if __name__=='__main__':
    main()
