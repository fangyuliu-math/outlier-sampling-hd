"""
CUTEst plotting, method-specific smooth fmin, streaming two-pass memory-safe variant.

This script reads converted CUTEst JSON files produced by run_cutest_noise_grid_covert.py.

Expected converted structure:

converted_results_cutest_const10_budget20/
    index.json
    run_names.txt
    problem_names.txt
    gaussian1_mean_const10/
        gaussian1_mean_const10_cutest_ARWHEAD.json
        gaussian1_mean_const10_cutest_BDEXP.json
        ...
    gaussian1_median_const10/
        gaussian1_median_const10_cutest_ARWHEAD.json
        ...

It produces data profiles and performance profiles:

plots_cutest_profiles/
    gaussian1/
        const10/
            gaussian1_const10_tau1_data_profile_smooth.pdf
            gaussian1_const10_tau1_data_profile_noisy.pdf
            gaussian1_const10_tau1_perf_profile_smooth.pdf
            gaussian1_const10_tau1_perf_profile_noisy.pdf
            ...

Main definitions
----------------

For a problem p, algorithm a, and repeated run r, let

    b_{a,r}(p)

be the first budget at which the run satisfies

    f(x_k) <= f_min + tau * (f_0 - f_min).

DATA PROFILE
------------

For algorithm a,

    DP_a(alpha)
      = #{(p,r): b_{a,r}(p) <= alpha}
        / (#problems * #runs).

Thus every individual repeated run contributes one observation.

PERFORMANCE PROFILE
-------------------

For each problem p,

    b_star(p) = min_{a,r} b_{a,r}(p).

Then

    PP_a(alpha)
      = #{(p,r): b_{a,r}(p) <= alpha * b_star(p)}
        / (#problems * #runs).

Definitions used here
---------------------
For a minimization problem, a run is considered solved at tolerance tau if

    f(x_k) <= f_min + tau * (f_0 - f_min),

where f_min is computed from the smooth objective values separately for
(problem, noise model, sampling rule, aggregator), pooled across all runs
of that same setting. The output compares method-specific targets.

Data profile:
    x-axis: budget in evaluations / (n + 1)
    y-axis: proportion of successful (problem, run) pairs.

Performance profile:
    x-axis: solve budget / best solve budget b_star(p), where
            b_star(p) = min_{algorithm, run} b_{algorithm,run}(p)
    y-axis: proportion of (problem, run) pairs solved within that ratio.

Important:
- This script is CUTEst-specific and does not use More--Wild problem numbers.
- It does not depend on run_all_problems_plots.py.
- It is designed to be robust to problem names such as ARWHEAD, BDEXP, etc.
"""

import os
import json
import glob
import csv
from collections import defaultdict
