"""
Streamed audit export matching run_cutest_noise_grid_plots_specific_fmin_fixed.py.

Usage, from repository root:
  python examples/export_cutest_failure_solve_info_matched.py \
      --noise failure_low_p08 --tau tau5 \
      --root converted_results_cutest_parallel --outdir audit_results_matched

The script imports the ORIGINAL plot's solve_budget_for_run() without running
its main(). It preserves per-run f0, noise-specific fmin pooled over all
sampling rules & aggregators, and the original budget axis. No raw files are
changed. It separately calculates a 1-based evaluation-count diagnostic.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np

TAUS = {"tau1": 1e-1, "tau2": 1e-2, "tau3": 1e-3, "tau4": 1e-4, "tau5": 1e-5}
CUTS = (1, 2, 4, 8, 16, 32)


def min_finite(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.min(v)) if v.size else float('inf')


def hit_and_f0(run, fmin, tau, profile_type, plot):
    """Obtain first original-array hit index, faithfully mirroring plot's filters.

    If the plotting routine uses its near-equal-f0 special case, the first
    finite evaluation is considered a hit and the *plot* budget will be zero.
    """
    v = plot.get_values_from_run(run, profile_type)
    budget = plot.get_budget_axis_from_run(run)
    L = min(len(v), len(budget))
    if not L:
        return -1, float('nan'), False
    v = v[:L]
    budget = budget[:L]
    inds = np.flatnonzero(np.isfinite(v) & np.isfinite(budget))
    if not inds.size or not np.isfinite(fmin):
        return -1, float('nan'), False
    f0 = float(v[inds[0]])
    special = abs(f0 - fmin) <= 1e-14 * max(1., abs(f0))
    if special:
        return int(inds[0]), f0, True
    target = fmin + tau * (f0 - fmin)
    hits = inds[np.flatnonzero(v[inds] <= target)]
    return (int(hits[0]) if hits.size else -1), f0, False


def numerical_summary(rows, kind):
    """Profile definition from original plot: min budget across all runs per problem."""
    by_p = defaultdict(list)
    for row in rows:
        by_p[row['probname']].append(row[kind])
    best = {p: min((v for v in values if np.isfinite(v)), default=np.inf)
            for p, values in by_p.items()}
    stats = defaultdict(list)
    for row in rows:
        b = row[kind]
        baseline = best[row['probname']]
        if kind == 'original_plot_budget':
            # This is the exact zero-baseline branch from build_perf_profile_curves().
            if not np.isfinite(b) or not np.isfinite(baseline):
                ratio = np.inf
            elif baseline > 0:
                ratio = max(1., b / baseline)
            else:
                ratio = 1. if b <= 0 else np.inf
        else:
            ratio = b / baseline if np.isfinite(b) and np.isfinite(baseline) and baseline > 0 else np.inf
        row['ratio_' + kind] = ratio
        stats[row['key']].append((b, ratio))
    return best, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--noise', default='failure_low_p08')
    ap.add_argument('--tau', choices=sorted(TAUS), default='tau5')
    ap.add_argument('--root', default='converted_results_cutest_parallel')
    ap.add_argument('--outdir', default='audit_results_matched')
    args = ap.parse_args()

    # The original plot file must be in the same examples/ folder as this script.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_cutest_noise_grid_plots_specific_fmin_fixed as plot

    paths = sorted(Path(args.root).glob(args.noise + '_*/*.json'))
    paths = [p for p in paths if p.name != 'index.json']
    if not paths:
        ap.error('No converted JSON files found for this noise; check --root and --noise')
    print(f'[SOURCE] {len(paths)} files; noise={args.noise}; tau={args.tau}')
    print('[SOURCE] Original plot module:', plot.__file__)

    # Pass 1: fmin is the minimum finite observed objective value across ALL
    # aggregators, sampling rules, and runs, but within the selected noise.
    fmins = {'smooth': {}, 'noisy': {}}
    count = 0
    for p in paths:
        with p.open(encoding='utf-8') as f:
            d = json.load(f)
        if d.get('noise_model') != args.noise:
            raise ValueError(f'Unexpected noise_model in {p}: {d.get("noise_model")}')
        problem = d['problem_name']
        for kind in ('smooth', 'noisy'):
            best = fmins[kind].get(problem, np.inf)
            for run in d['runs']:
                best = min(best, min_finite(plot.get_values_from_run(run, kind)))
            fmins[kind][problem] = best
        count += 1
        if count % 50 == 0:
            print(f'  pass 1: {count}/{len(paths)} JSON files', flush=True)

    # Pass 2: ALWAYS use original plot's per-run f0 and budget_axis.
    groups = defaultdict(list)
    special_counts = defaultdict(int)
    for count, p in enumerate(paths, 1):
        with p.open(encoding='utf-8') as f:
            d = json.load(f)
        problem = d['problem_name']
        agg = str(d['aggregator'])
        ns = str(d['ns_rule'])
        n = int(d['n'])
        if n <= 0:
            raise ValueError(f'Invalid dimension n={n} in {p}')
        runs = d['runs']
        if int(d.get('nruns', len(runs))) != len(runs):
            raise ValueError(f'Run count mismatch in {p}')
        for i, run in enumerate(runs):
            nr = int(run.get('run', i))
            row = dict(probname=problem, key=agg, nrun=nr, n=n, ns_rule=ns,
                       noise_model=args.noise, source_json=str(p), nf=int(run.get('nf', 0)))
            for kind in ('smooth', 'noisy'):
                fmin = fmins[kind][problem]
                idx, f0, special = hit_and_f0(run, fmin, TAUS[args.tau], kind, plot)
                orig = float(plot.solve_budget_for_run(run, fmin, TAUS[args.tau], kind))
                if (idx < 0) != (not np.isfinite(orig)):
                    raise AssertionError(f'Solve-status mismatch for {p}, run {nr}, {kind}')
                row[f'{kind}_{args.tau}'] = idx
                row[f'{kind}_f0'] = f0
                row[f'{kind}_fmin'] = fmin
                row[f'{kind}_plot_budget'] = orig
                row[f'{kind}_eval_count'] = float(idx + 1) if idx >= 0 else float('inf')
                row[f'{kind}_near_equal_f0_fmin'] = int(special)
                if special:
                    special_counts[kind] += 1
            noisy = plot.get_values_from_run(run, 'noisy')
            row['noisy_neg10000_count'] = int(np.count_nonzero(noisy == -10000.))
            row['noisy_starts_neg10000'] = int(len(noisy) > 0 and noisy[0] == -10000.)
            groups[ns].append(row)
        if count % 50 == 0:
            print(f'  pass 2: {count}/{len(paths)} JSON files', flush=True)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for ns, rows in sorted(groups.items()):
        dup_ids = [(r['probname'], r['key'], r['nrun']) for r in rows]
        if len(set(dup_ids)) != len(dup_ids):
            raise ValueError(f'Duplicate problem/algorithm/run in {ns}')
        reports = {}
        for kind in ('smooth', 'noisy'):
            orig_values = [dict(r, original_plot_budget=r[f'{kind}_plot_budget']) for r in rows]
            corr_values = [dict(r, corrected_eval_count=r[f'{kind}_eval_count']) for r in rows]
            best_orig, orig_stats = numerical_summary(orig_values, 'original_plot_budget')
            best_corrected, corr_stats = numerical_summary(corr_values, 'corrected_eval_count')
            for row, original, corrected in zip(rows, orig_values, corr_values):
                row[f'original_ratio_{kind}'] = original['ratio_original_plot_budget']
                row[f'corrected_ratio_{kind}'] = corrected['ratio_corrected_eval_count']
            reports[kind] = (best_orig, orig_stats, corr_stats)
        fieldnames = list(rows[0])
        csv_path = out / f'solve_info_matched_{args.noise}_{ns}_{args.tau}.csv'
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f'\n[SAVED] {csv_path} ({len(rows)} rows, {len(set(r["probname"] for r in rows))} problems)')
        print(f'[NOISY] -10000 at initial evaluation: {sum(r["noisy_starts_neg10000"] for r in rows)}/{len(rows)}')
        for kind, (best_orig, orig_stats, corr_stats) in reports.items():
            print(f'[{kind.upper()}] Problems with original b_star=0: {sum(b == 0 for b in best_orig.values())}/{len(best_orig)}')
            print(f'[{kind.upper()}] Problems with no finite original b_star: {sum(not np.isfinite(b) for b in best_orig.values())}')
            print(f'{"Algorithm":<20} {"Solved":>7} {"Orig @1":>9} {"Orig @32":>10} {"Correct @1":>11} {"Correct @32":>12}')
            for algo in sorted(orig_stats):
                original = orig_stats[algo]
                corrected = corr_stats[algo]
                prop = lambda values, cutoff: sum(ratio <= cutoff for cost, ratio in values) / len(values)
                print(f'{algo:<20} {sum(np.isfinite(cost) for cost,ratio in original):>7} '
                      f'{prop(original,1):>9.4f} {prop(original,32):>10.4f} '
                      f'{prop(corrected,1):>11.4f} {prop(corrected,32):>12.4f}')
    print('\n[NOTE] This preserves the original plot definition, including its zero-budget branch.')
    print('[NOTE] Corrected ratios use positive 1-based evaluation counts and per-problem minima across all runs.')
    print('[NOTE] Such correction does NOT make a noisy fmin based on -10000 scientifically meaningful.')

if __name__ == '__main__':
    main()
