"""
Script to run pybobyqa_sampling on a single More & Wild test problem, and plot the results
"""
import matplotlib.pyplot as plt
import numpy as np

import pybobyqa_sampling

from more_wild import get_problem_as_scalar_objective
from objfun_wrapper import ObjfunWrapper


def load_more_wild_problem(probnum: int, noise_model='smooth'):
    objfun, x0, n, m = get_problem_as_scalar_objective(probnum)  # get smooth problem from more_wild.py
    # Put smooth problem in ObjfunWrapper
    objfun_wrapped = ObjfunWrapper('more_wild_prob%g_%s' % (probnum, noise_model), objfun, n, x0, bounds=None,
                                   noise_model=noise_model)
    return objfun_wrapped


def run_solver(objfun_wrapped: ObjfunWrapper, budget_in_gradients: int, nruns=1):
    # Minimize objfun_wrapped by calling pybobyqa_sampling.solve()
    # Max objective evaluations = budget_in_gradients * (n+1) for an n-dimensional problem
    # (why? finite differencing to get an n-dimensional gradient requires n+1 evaluations)
    full_results = {}
    full_results['nruns'] = nruns
    for nrun in range(nruns):
        objfun_wrapped.clear()
        soln = pybobyqa_sampling.solve(objfun_wrapped, objfun_wrapped._x0,
                                       bounds=objfun_wrapped.bounds_as_lower_upper_arrays(),
                                       maxfun=budget_in_gradients * (objfun_wrapped._n + 1),
                                       nsamples=lambda delta, rho, iter, nruns: int(max(1.0/delta, 1.0)),  # number of samples, N=10
                                       nsamples_aggregator=np.mean,
                                       print_progress=True,
                                       objfun_has_noise=True)
        full_results['run%g' % nrun] = objfun_wrapped.get_results(vectors_as_numpy=True)
    return full_results


def plot_results_simple(full_results, save_file='tmp.png', xaxis_in_gradients=True):
    # Plot the objective decrease for a single run of a single problem
    plt.figure()
    plt.clf()

    for nrun in range(full_results['nruns']):
        results = full_results['run%g' % nrun]
        xvals = np.arange(results['nf']) / (results['n'] + 1) if xaxis_in_gradients else np.arange(results['nf'])
        plt.semilogy(xvals, results['fvals_smooth'], 'C0-', label='True value' if nrun==1 else '_nolabel_')
        plt.semilogy(xvals, results['fvals_noisy'], 'C1--', label='Noisy value' if nrun==1 else '_nolabel_')
    plt.xlabel('Budget in gradients (i.e. Objective evaluations / (n+1))' if xaxis_in_gradients else 'Objective evaluations')
    plt.ylabel('Objective value f(x)')
    plt.legend(loc='best')
    plt.grid()

    plt.savefig(save_file, bbox_inches='tight')
    return


def main():
    probnum = 7  # problem to run, integer in [1,53]
    # noise_model = 'smooth'
    noise_model = 'gaussian1'
    nruns = 5
    budget_in_gradients = 500  # small value for testing purposes

    objfun_wrapped = load_more_wild_problem(probnum, noise_model=noise_model)
    results = run_solver(objfun_wrapped, budget_in_gradients, nruns=nruns)
    plot_results_simple(results, save_file='tmp.png', xaxis_in_gradients=True)

    print("Done")
    return


if __name__ == '__main__':
    main()
