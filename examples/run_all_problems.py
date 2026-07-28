"""
Script to run full set of More & Wild test problems and save results to JSON files
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os

import pybobyqa_sampling

from more_wild import get_problem_as_scalar_objective
from objfun_wrapper import ObjfunWrapper

RAW_RESULTS_DIR = 'raw_results'
if not os.path.isdir(RAW_RESULTS_DIR):
    os.mkdir(RAW_RESULTS_DIR)


def save_dict_to_json(mydict, outfile):
    with open(outfile, 'w') as ofile:
        json.dump(mydict, ofile, indent=4, sort_keys=True)
    return


def load_more_wild_problem(probnum: int, noise_model='smooth'):
    objfun, x0, n, m = get_problem_as_scalar_objective(probnum)
    objfun_wrapped = ObjfunWrapper('more_wild_prob%g' % probnum, objfun, n, x0, bounds=None,
                                   noise_model=noise_model)
    return objfun_wrapped


def run_solver_single_problem(objfun_wrapped: ObjfunWrapper, budget_in_gradients: int, nruns=1,
               nsamples_aggregation='mean', nsamples_function='N1', flag_noise=False):
    # Minimize objfun_wrapped by calling pybobyqa_sampling.solve()
    # Max objective evaluations = budget_in_gradients * (n+1) for an n-dimensional problem
    max_evals = budget_in_gradients * (objfun_wrapped._n + 1)

    # Select aggregator
    if nsamples_aggregation == 'mean':
        nsamples_aggregator = np.mean
    elif nsamples_aggregation == 'median':
        nsamples_aggregator = np.median
    else:
        raise RuntimeError("Unknown nsamples_aggregation '%s'" % nsamples_aggregation)

    # Select nsamples
    if nsamples_function == 'N1':
        # No sampling, N=1 always
        nsamples = lambda delta, rho, iter, nruns: 1
    elif nsamples_function == 'N10':
        # N=10 samples always
        nsamples = lambda delta, rho, iter, nruns: 10
    elif nsamples_function == 'delta_inv':
        # Increasing sampling, N ~ 1/delta
        # BUT, make sure we always ensure N >= 1 and is an integer
        nsamples = lambda delta, rho, iter, nruns: int(max(1.0/delta, 1))
    else:
        raise RuntimeError("Unknown nsamples_function '%s'" % nsamples_function)

    full_results = {}
    full_results['nruns'] = nruns
    full_results['objfun_name'] = objfun_wrapped._name
    full_results['n'] = objfun_wrapped._n
    full_results['budget_in_gradients'] = budget_in_gradients
    full_results['max_evals'] = max_evals
    full_results['noise_model'] = objfun_wrapped._noise_model
    full_results['nsamples_function'] = nsamples_function
    full_results['nsamples_aggregation'] = nsamples_aggregation
    full_results['flag_noise'] = flag_noise if objfun_wrapped.is_noisy() else False  # always False for smooth functions
    for nrun in range(nruns):
        objfun_wrapped.clear()
        soln = pybobyqa_sampling.solve(objfun_wrapped, objfun_wrapped._x0,
                                       bounds=objfun_wrapped.bounds_as_lower_upper_arrays(),
                                       maxfun=max_evals,
                                       nsamples=nsamples,
                                       nsamples_aggregator=nsamples_aggregator,
                                       print_progress=False,
                                       objfun_has_noise=flag_noise if objfun_wrapped.is_noisy() else False)
        full_results['run%g' % nrun] = objfun_wrapped.get_results(vectors_as_numpy=False)
    return full_results


def run_solver_all_problems(run_name: str, probnums_to_run: list, budget_in_gradients: int, noise_model='smooth',
                            nruns=1, nsamples_aggregation='mean', nsamples_function='N1', flag_noise=False,
                            raw_results_dir=RAW_RESULTS_DIR):
    # Call run_solver_single_problem() for all problems in a collection
    for probnum in probnums_to_run:
        print("Solving problem %g" % probnum)
        objfun_wrapped = load_more_wild_problem(probnum, noise_model=noise_model)
        this_problem_results = run_solver_single_problem(objfun_wrapped,
                                                         budget_in_gradients, nruns=nruns,
                                                         nsamples_aggregation=nsamples_aggregation,
                                                         nsamples_function=nsamples_function,
                                                         flag_noise=flag_noise)

        # Save this result to raw_results_dir/run_name/run_name_probname.json (to make it easier to find)
        results_filename = '%s_%s.json' % (run_name, objfun_wrapped._name)
        save_dir = os.path.join(raw_results_dir, run_name)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        save_dict_to_json(this_problem_results, os.path.join(save_dir, results_filename))
        print(" - results saved in %s" % results_filename)
    return


def main():
    #### Actually do a proper run of everything ####

    ## Inputs that need to be the same for every combination, for fair comparisons ##
    budget_in_gradients = 10

    probnums_to_run = list(range(1, 54))  # All More and Wild problems, i.e. [1, ..., 53]
    # probnums_to_run = [1, 2]  # for testing purposes

    nruns = 5  # how many times to run each problem

    ## Inputs to vary to get different comparisons  ##
    # Make sure run_name is unique for every combination
    combination_to_run = 2  # change this to pick the combination to run

    if combination_to_run == 1:
        run_name = 'basic_smooth'  # output filename
        noise_model = 'smooth'  # What noise model to use
        nsamples_function = 'N1'  # How many samples N for every evaluation?
        nsamples_aggregation = 'mean'  # How to combine noisy samples for every evaluation
        flag_noise = True  # set objfun_has_noise=True for noisy problems? (generally better results; not used if noise_model=='smooth')
    elif combination_to_run == 2:
        run_name = 'gaussian1_N1'  # output filename
        noise_model = 'gaussian1'  # What noise model to use
        nsamples_function = 'N1'  # How many samples N for every evaluation?
        nsamples_aggregation = 'mean'  # How to combine noisy samples for every evaluation
        flag_noise = True  # set objfun_has_noise=True for noisy problems? (generally better results; not used if noise_model=='smooth')
    else:
        raise RuntimeError("Unknown combination_to_run '%s'" % combination_to_run)

    ## Do the run ##
    print("**********************")
    print("Running combination %g" % combination_to_run)
    print("budget_in_gradients = %g" % budget_in_gradients)
    print("nruns = %g" % nruns)
    print("run_name = %s" % run_name)
    print("noise_model = %s" % noise_model)
    print("nsamples_function = %s" % nsamples_function)
    print("nsamples_aggregation = %s" % nsamples_aggregation)
    print("flag_noise = %s" % flag_noise)
    print("**********************")

    run_solver_all_problems(run_name, probnums_to_run, budget_in_gradients, noise_model=noise_model,
                            nruns=nruns, nsamples_aggregation=nsamples_aggregation, nsamples_function=nsamples_function,
                            flag_noise=flag_noise)
    print("Done - all results saved in folder %s" % RAW_RESULTS_DIR)
    return


if __name__ == '__main__':
    main()
