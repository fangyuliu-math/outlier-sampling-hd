"""
Script to make data and performance profiles using results from run_all_problems.py
"""
import json
from math import log
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from run_all_problems import RAW_RESULTS_DIR

ALL_TAUS = {'tau1': 1e-1, 'tau2': 1e-2, 'tau3': 1e-3, 'tau4': 1e-4, 'tau5': 1e-5}


def read_json(infile):
    with open(infile, 'r') as ifile:
        mydict = json.load(ifile)
    return mydict


def load_all_results(problem_names, run_name, raw_results_dir=RAW_RESULTS_DIR):
    # Result is a list, where each entry is a dict corresponding to the data in a single JSON file (which can have
    # several runs of the same solver on the same problem)
    all_results = []
    for probname in problem_names:
        filename = '%s_%s.json' % (run_name, probname)
        this_results = read_json(os.path.join(raw_results_dir, run_name, filename))
        all_results.append(this_results)
    return all_results


def load_all_more_wild_problems(run_name, raw_results_dir=RAW_RESULTS_DIR):
    problem_names = ['more_wild_prob%g' % probnum for probnum in range(1, 54)]  # more_wild_prob1, ..., more_wild_prob53
    return load_all_results(problem_names, run_name, raw_results_dir=raw_results_dir)


def all_results_consistent(all_results_info):
    # Check that all the different sets of 'all_results' have the same problems and nruns
    keys = list(all_results_info.keys())

    if len(keys) == 0:
        print("Warning: checking consistency of empty dict of results")
        return True

    nprobs = len(all_results_info[keys[0]])
    for key in keys:
        if len(all_results_info[key]) != nprobs:
            print("Inconsistency: comparing everything with %s" % keys[0])
            print("Inconsistency: results for %s has a different number of problems" % key)
            return False

        for i in range(nprobs):
            init_key_results = all_results_info[keys[0]][i]
            this_key_results = all_results_info[key][i]

            if init_key_results['objfun_name'] != this_key_results['objfun_name']:
                print("Inconsistency: comparing everything with %s" % keys[0])
                print("Inconsistency: problem %g for %s is problem %s, expect %s" % (i, key, this_key_results['objfun_name'], init_key_results['objfun_name']))
                return False

            if init_key_results['nruns'] != this_key_results['nruns']:
                print("Inconsistency: comparing everything with %s" % keys[0])
                print("Inconsistency: results for %s has a different number of runs for problem %s" % (key, this_key_results['objfun_name']))
                return False

    return True


def get_all_profile_info(all_results_info, tau_levels=ALL_TAUS):
    """
    For a dictionary of different run results (for the same set of problems and same nruns for each)
    determine a suitable fmin and f0, then calculate the number of evaluations required for each run to solve
    each problem to accuracy level tau.

    A problem is solved for accuracy level tau>0 when we find a point such that
        f(x) <= fmin + tau*(f0-fmin)
    We count the number of objective evaluations required to achieve this (-1 if that accuracy never achieved).

    Looking at the 'true' and 'noisy' objective values separately.
    - For 'true' objective values, use f0 = f(x0) and fmin = best f(x) found by any run from the results
    - For 'noisy' objective values, use f0 = average noisy f(x0) across all runs and fmin = best noisy f(x) found
    Note: 'noisy' f0 and fmin are likely not very useful for problems with large outliers.
    """
    # Make sure all the different things we compare have the same underlying problems and nruns
    if not all_results_consistent(all_results_info):
        raise RuntimeError("Trying to build profiles for inconsistent results")

    keys = list(all_results_info.keys())
    if len(keys) == 0:
        raise RuntimeError("Cannot make profiles for empty set of results")

    # First, calculate true/noisy f0 and fmin values
    nprobs = len(all_results_info[keys[0]])
    probnames = []
    nruns_per_prob = np.full(nprobs, 1, dtype=int)  # how many nruns are used for each problem?
    n_per_prob = np.full(nprobs, 0, dtype=int)  # what is the dimension of each problem?
    smooth_f0_values = np.full(nprobs, np.inf)
    smooth_fmin_values = np.full(nprobs, np.inf)
    noisy_f0_values = np.full(nprobs, 0.0)  # will add this_noisy_f0 / (nruns * nsolvers) each time, so start at zero
    noisy_fmin_values = np.full(nprobs, np.inf)
    for key in keys:
        for i in range(nprobs):
            this_key_results = all_results_info[key][i]
            # Only need to fill nruns_per_prob and true_f0 for the first set of results
            if key == keys[0]:
                nruns_per_prob[i] = int(this_key_results['nruns'])
                n_per_prob[i] = int(this_key_results['n'])
                smooth_f0_values[i] = float(this_key_results['run0']['fvals_smooth'][0])
                probnames.append(this_key_results['objfun_name'])
            # Need to fill the remainder each time
            for nrun in range(nruns_per_prob[i]):
                this_true_fmin = float(min(this_key_results['run%g' % nrun]['fvals_smooth']))
                this_noisy_f0 = float(this_key_results['run%g' % nrun]['fvals_noisy'][0])
                this_noisy_fmin = float(min(this_key_results['run%g' % nrun]['fvals_noisy']))
                smooth_fmin_values[i] = min(smooth_fmin_values[i], this_true_fmin)
                noisy_f0_values[i] += this_noisy_f0 / (nruns_per_prob[i] * len(keys))  # after loops finished, this will have the average
                noisy_fmin_values[i] = min(noisy_fmin_values[i], this_noisy_fmin)

    f0_fmin = pd.DataFrame.from_dict({'probname': probnames, 'nruns': nruns_per_prob,
                                      'smooth_f0': smooth_f0_values, 'smooth_fmin': smooth_fmin_values,
                                      'noisy_f0': noisy_f0_values, 'noisy_fmin': noisy_fmin_values})
    f0_fmin.set_index('probname', inplace=True)

    # Now, for each run of each set of results, count how long it took to solve a problem to the given accuracy
    all_solve_info = {'key': [], 'probnum': [], 'probname': [], 'n': [], 'nrun': [], 'nruns': [], 'nevals': [],
                      'this_smooth_fmin': [], 'this_noisy_fmin': []}
    for tau_str in tau_levels.keys():
        all_solve_info['smooth_%s' % tau_str] = []
        all_solve_info['noisy_%s' % tau_str] = []

    for key in keys:
        for i in range(nprobs):
            this_key_results = all_results_info[key][i]
            probname = this_key_results['objfun_name']
            n = int(this_key_results['n'])
            nruns = int(this_key_results['nruns'])
            for nrun in range(nruns):
                smooth_fvals = np.array(this_key_results['run%g' % nrun]['fvals_smooth'])
                noisy_fvals = np.array(this_key_results['run%g' % nrun]['fvals_noisy'])
                nevals = int(this_key_results['run%g' % nrun]['nf'])

                # Append results
                all_solve_info['key'].append(key)
                all_solve_info['probnum'].append(i)
                all_solve_info['probname'].append(probname)
                all_solve_info['n'].append(n)
                all_solve_info['nrun'].append(nrun)
                all_solve_info['nruns'].append(nruns)
                all_solve_info['nevals'].append(nevals)
                all_solve_info['this_smooth_fmin'].append(float(np.min(smooth_fvals)))
                all_solve_info['this_noisy_fmin'].append(float(np.min(noisy_fvals)))

                for tau_str in tau_levels.keys():
                    tau_value = tau_levels[tau_str]
                    smooth_target = f0_fmin.loc[probname]['smooth_fmin'] + tau_value * (f0_fmin.loc[probname]['smooth_f0'] - f0_fmin.loc[probname]['smooth_fmin'])
                    noisy_target = f0_fmin.loc[probname]['noisy_fmin'] + tau_value * (f0_fmin.loc[probname]['noisy_f0'] - f0_fmin.loc[probname]['noisy_fmin'])
                    nevals_smooth = -1
                    nevals_noisy = -1
                    if np.min(smooth_fvals) <= smooth_target:
                        nevals_smooth = np.argmax(smooth_fvals <= smooth_target)  # stops at first 'True' index
                    if np.min(noisy_fvals) <= noisy_target:
                        nevals_noisy = np.argmax(noisy_fvals <= noisy_target)
                    all_solve_info['smooth_%s' % tau_str].append(int(nevals_smooth))
                    all_solve_info['noisy_%s' % tau_str].append(int(nevals_noisy))

    all_solve_info = pd.DataFrame.from_dict(all_solve_info)
    return f0_fmin, all_solve_info


def build_data_profiles(all_solve_info, tau_str, max_budget_in_gradients, average_over_nruns=True):
    """
    Build values used in a data profile
    - If average_over_nruns=True, create one profile per key, showing fraction of *problem runs* solved
    - If average_over_nruns=False, create one profile per key *and run*, showing fraction of problems solved

    For average_over_nruns=False, must have the same nruns for all problems
    """
    if 'smooth_%s' % tau_str not in all_solve_info or 'noisy_%s' % tau_str not in all_solve_info:  # if not valid column
        raise RuntimeError("Invalid tau_str")

    nruns = 1
    if not average_over_nruns:
        if all_solve_info['nruns'].min() != all_solve_info['nruns'].max():
            raise RuntimeError("Cannot do average_over_nruns=False if nruns differs by problem or run name")
        nruns = all_solve_info['nruns'].min()  # set to any value

    xvals = np.linspace(0, max_budget_in_gradients, 100)
    keys = sorted(all_solve_info['key'].unique())
    data_profile_info = {'xvals': xvals, 'nruns': nruns, 'average_over_nruns': average_over_nruns, 'max_budget_in_gradients': max_budget_in_gradients}
    if average_over_nruns:
        for key in keys:
            this_smooth_dp = []
            this_noisy_dp = []
            this_key_results = all_solve_info[all_solve_info['key'] == key]
            smooth_solved_budget = this_key_results['smooth_%s' % tau_str] / (this_key_results['n'] + 1)
            noisy_solved_budget = this_key_results['noisy_%s' % tau_str] / (this_key_results['n'] + 1)
            smooth_solved_budget[this_key_results['smooth_%s' % tau_str] < 0] = np.inf
            noisy_solved_budget[this_key_results['noisy_%s' % tau_str] < 0] = np.inf
            for xval in xvals:
                this_smooth_dp.append(len(smooth_solved_budget[smooth_solved_budget <= xval]) / len(smooth_solved_budget))
                this_noisy_dp.append(len(noisy_solved_budget[noisy_solved_budget <= xval]) / len(noisy_solved_budget))
            this_smooth_dp = np.array(this_smooth_dp)
            this_noisy_dp = np.array(this_noisy_dp)
            data_profile_info[key] = {'smooth': [this_smooth_dp], 'noisy': [this_noisy_dp]}
    else:
        for key in keys:
            data_profile_info[key] = {'smooth': [], 'noisy': []}
            for nrun in range(nruns):
                this_smooth_dp = []
                this_noisy_dp = []
                this_run_results = all_solve_info[(all_solve_info['key'] == key) & (all_solve_info['nrun'] == nrun)]
                smooth_solved_budget = this_run_results['smooth_%s' % tau_str] / (this_run_results['n'] + 1)
                noisy_solved_budget = this_run_results['noisy_%s' % tau_str] / (this_run_results['n'] + 1)
                smooth_solved_budget[this_run_results['smooth_%s' % tau_str] < 0] = np.inf
                noisy_solved_budget[this_run_results['noisy_%s' % tau_str] < 0] = np.inf
                for xval in xvals:
                    this_smooth_dp.append(len(smooth_solved_budget[smooth_solved_budget <= xval]) / len(smooth_solved_budget))
                    this_noisy_dp.append(len(noisy_solved_budget[noisy_solved_budget <= xval]) / len(noisy_solved_budget))
                this_smooth_dp = np.array(this_smooth_dp)
                this_noisy_dp = np.array(this_noisy_dp)
                data_profile_info[key]['smooth'].append(this_smooth_dp)
                data_profile_info[key]['noisy'].append(this_noisy_dp)
    return data_profile_info


def build_perf_profiles(all_solve_info, tau_str, log2_max_ratio=5, average_over_nruns=True):
    """
    Build values used in a performance profile
    - If average_over_nruns=True, create one profile per key, showing fraction of *problem runs* solved
    - If average_over_nruns=False, create one profile per key *and run*, showing fraction of problems solved

    For average_over_nruns=False, must have the same nruns for all problems
    """
    if 'smooth_%s' % tau_str not in all_solve_info or 'noisy_%s' % tau_str not in all_solve_info:  # if not valid column
        raise RuntimeError("Invalid tau_str")

    nruns = 1
    if not average_over_nruns:
        if all_solve_info['nruns'].min() != all_solve_info['nruns'].max():
            raise RuntimeError("Cannot do average_over_nruns=False if nruns differs by problem or run name")
        nruns = all_solve_info['nruns'].min()  # set to any value

    # Get fastest solve for each problem, across all nruns for all keys
    min_solved_budget = {'probname': [], 'smooth_budget': [], 'noisy_budget': []}
    for probname in all_solve_info['probname'].unique():
        min_solved_budget['probname'].append(probname)
        smooth_solve_info = all_solve_info[(all_solve_info['probname']==probname) & (all_solve_info['smooth_%s' % tau_str] >= 0)]
        noisy_solve_info = all_solve_info[(all_solve_info['probname'] == probname) & (all_solve_info['noisy_%s' % tau_str] >= 0)]
        if len(smooth_solve_info) == 0:
            min_solved_budget['smooth_budget'].append(1)  # not solved at all, so putting an arbitrary value
        else:
            min_solved_budget['smooth_budget'].append(smooth_solve_info['smooth_%s' % tau_str].min())
        if len(noisy_solve_info) == 0:
            min_solved_budget['noisy_budget'].append(1)  # not solved at all, so putting an arbitrary value
        else:
            min_solved_budget['noisy_budget'].append(noisy_solve_info['noisy_%s' % tau_str].min())
    min_solved_budget = pd.DataFrame.from_dict(min_solved_budget)
    # print(min_solved_budget.head())
    # print(min_solved_budget.tail())

    # Append min_solved_budget into all_solve_info
    perf_all_solve_info = all_solve_info.merge(min_solved_budget, on='probname', validate='many_to_one')

    xvals = np.logspace(0.0, log2_max_ratio, 100, base=2.0)
    keys = sorted(perf_all_solve_info['key'].unique())
    perf_profile_info = {'xvals': xvals, 'nruns': nruns, 'average_over_nruns': average_over_nruns, 'log2_max_ratio': log2_max_ratio}
    if average_over_nruns:
        for key in keys:
            this_smooth_dp = []
            this_noisy_dp = []
            this_key_results = perf_all_solve_info[perf_all_solve_info['key'] == key]
            smooth_solved_budget = this_key_results['smooth_%s' % tau_str] / this_key_results['smooth_budget']
            noisy_solved_budget = this_key_results['noisy_%s' % tau_str] / this_key_results['noisy_budget']
            smooth_solved_budget[this_key_results['smooth_%s' % tau_str] < 0] = np.inf
            noisy_solved_budget[this_key_results['noisy_%s' % tau_str] < 0] = np.inf
            for xval in xvals:
                this_smooth_dp.append(len(smooth_solved_budget[smooth_solved_budget <= xval]) / len(smooth_solved_budget))
                this_noisy_dp.append(len(noisy_solved_budget[noisy_solved_budget <= xval]) / len(noisy_solved_budget))
            this_smooth_dp = np.array(this_smooth_dp)
            this_noisy_dp = np.array(this_noisy_dp)
            perf_profile_info[key] = {'smooth': [this_smooth_dp], 'noisy': [this_noisy_dp]}
    else:
        for key in keys:
            perf_profile_info[key] = {'smooth': [], 'noisy': []}
            for nrun in range(nruns):
                this_smooth_dp = []
                this_noisy_dp = []
                this_run_results = perf_all_solve_info[(perf_all_solve_info['key'] == key) & (perf_all_solve_info['nrun'] == nrun)]
                smooth_solved_budget = this_run_results['smooth_%s' % tau_str] / this_run_results['smooth_budget']
                noisy_solved_budget = this_run_results['noisy_%s' % tau_str] / this_run_results['noisy_budget']
                smooth_solved_budget[this_run_results['smooth_%s' % tau_str] < 0] = np.inf
                noisy_solved_budget[this_run_results['noisy_%s' % tau_str] < 0] = np.inf
                for xval in xvals:
                    this_smooth_dp.append(len(smooth_solved_budget[smooth_solved_budget <= xval]) / len(smooth_solved_budget))
                    this_noisy_dp.append(len(noisy_solved_budget[noisy_solved_budget <= xval]) / len(noisy_solved_budget))
                this_smooth_dp = np.array(this_smooth_dp)
                this_noisy_dp = np.array(this_noisy_dp)
                perf_profile_info[key]['smooth'].append(this_smooth_dp)
                perf_profile_info[key]['noisy'].append(this_noisy_dp)
    return perf_profile_info


def plot_data_profile(data_profile_info, plot_info, filename, use_smooth_fvals=True, xaxis_in_logscale=False):
    plt.figure()
    plt.clf()

    font_size = 'large'  # x-large for presentations
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    ax = plt.gca()  # current axes
    plot_fun = ax.semilogx if xaxis_in_logscale else ax.plot
    xvals = data_profile_info['xvals']
    for key in plot_info.keys():
        if key not in data_profile_info:
            raise RuntimeError("Don't have data profile info for plot_info key %s" % key)
        this_plot_info = plot_info[key]
        for i, yvals in enumerate(data_profile_info[key]['smooth' if use_smooth_fvals else 'noisy']):
            plot_fun(xvals, yvals, color=this_plot_info['color'], linestyle=this_plot_info['linestyle'],
                     label='_nolegend_' if i > 0 else this_plot_info['label'], linewidth=2.0)

    ax.set_xlabel(r"Solved budget in evaluations / ($n$+1)", fontsize=font_size)
    ax.set_ylabel(r"Proportion problems solved", fontsize=font_size)
    ax.legend(loc='lower right', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.axis([0, np.max(xvals), 0, 1])  # (xlow, xhigh, ylow, yhigh)
    ax.grid()
    plt.savefig(filename, bbox_inches='tight')
    return

def plot_perf_profile(perf_profile_info, plot_info, filename, use_smooth_fvals=True):
    plt.figure()
    plt.clf()

    font_size = 'large'  # x-large for presentations
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    ax = plt.gca()  # current axes
    plot_fun = ax.semilogx
    xvals = perf_profile_info['xvals']
    for key in plot_info.keys():
        if key not in perf_profile_info:
            raise RuntimeError("Don't have data profile info for plot_info key %s" % key)
        this_plot_info = plot_info[key]
        for i, yvals in enumerate(perf_profile_info[key]['smooth' if use_smooth_fvals else 'noisy']):
            plot_fun(xvals, yvals, color=this_plot_info['color'], linestyle=this_plot_info['linestyle'],
                     label='_nolegend_' if i > 0 else this_plot_info['label'], linewidth=2.0)

    ax.set_xlabel(r"Budget / min budget of any solver", fontsize=font_size)
    ax.set_ylabel(r"Proportion problems solved", fontsize=font_size)
    ax.legend(loc='lower right', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.axis([np.min(xvals), np.max(xvals), 0, 1])  # (xlow, xhigh, ylow, yhigh)
    ax.grid()

    # Nicely format x-axis labels
    log_xmax = int(round(log(np.max(xvals), 2.0)))
    xticks = [2 ** y for y in range(log_xmax + 1)]  # 1, 2, 4, 8, ..., max(xvals)
    ax.set_xticks(xticks)
    ax.minorticks_off()  # in newer matploblib versions, minor ticks break label changes for log-scale axes
    # ax.set_xticks(range(1, xticks[-1] + 1), minor=True)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.savefig(filename, bbox_inches='tight')
    return


def main():
    # Load all results by run_name
    all_results_info = {}
    all_results_info['basic_smooth'] = load_all_more_wild_problems('basic_smooth')
    all_results_info['gaussian1_N1'] = load_all_more_wild_problems('gaussian1_N1')

    # Put plotting information here (legend label, line color, etc.)
    plot_info = {}
    plot_info['basic_smooth'] = {'label': 'No noise', 'color': 'C0', 'linestyle': '-'}
    plot_info['gaussian1_N1'] = {'label': 'Gaussian noise', 'color': 'C1', 'linestyle': '-'}
    max_budget_in_gradients = 10  # max x-axis for data profiles
    average_over_nruns = True  # show all nruns in one curve (True), or each nrun in different curve (False)

    # Extract all key information used to build profiles
    f0_fmin, all_solve_info = get_all_profile_info(all_results_info)
    # f0_fmin is a pandas DataFrame with information about per-problem f0 and fmin values used to build profiles
    # all_solve_info is a pandas DataFrame with all the 'how long to solve a problem' for different accuracy levels
    # all_solve_info is what is used to build data/performance profiles
    # print(all_solve_info.head())
    # print(all_solve_info.tail())

    # Build and plot data profiles
    data_profile_info = build_data_profiles(all_solve_info, 'tau1', max_budget_in_gradients, average_over_nruns=average_over_nruns)
    plot_data_profile(data_profile_info, plot_info, 'tmp_data_profile_smooth.png', use_smooth_fvals=True)
    plot_data_profile(data_profile_info, plot_info, 'tmp_data_profile_noisy.png', use_smooth_fvals=False)

    # Build and plot performance profiles
    perf_profile_info = build_perf_profiles(all_solve_info, 'tau1', average_over_nruns=average_over_nruns)
    plot_perf_profile(perf_profile_info, plot_info, 'tmp_perf_profile_smooth.png', use_smooth_fvals=True)
    plot_perf_profile(perf_profile_info, plot_info, 'tmp_perf_profile_noisy.png', use_smooth_fvals=False)
    print("Done")
    return


if __name__ == '__main__':
    main()
