"""
CUTEst-specific plotting script for converted high-dimensional noisy DFO results.

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

where f_min is the best value found for that problem within the same noise
model, across all loaded methods and repeated runs, using either smooth
values or noisy values depending on the plot type.

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
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# ============================================================
# Path setup
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.basename(SCRIPT_DIR) == "examples":
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
else:
    PROJECT_ROOT = SCRIPT_DIR


# If you want to force a specific converted directory, set this manually, e.g.
# CONVERTED_ROOT = os.path.join(PROJECT_ROOT, "converted_results_cutest_const10_budget20")
# CONVERTED_ROOT = None

# Only read this converted folder.
CONVERTED_ROOT = os.path.join(
    PROJECT_ROOT,
    "converted_results_cutest_parallel"
)

# If CONVERTED_ROOT is None, the script searches these candidates and uses
# the newest existing one.
# CANDIDATE_CONVERTED_ROOTS = [
#     os.path.join(PROJECT_ROOT, "converted_results_cutest_pilot"),
#     os.path.join(PROJECT_ROOT, "examples", "converted_results_cutest_pilot"),
# ]


# Use a new output folder so that the corrected plots do not overwrite
# the previous plots.
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "plots_cutest_profiles_parallel_noise_specific_fmin_multirun_fixed"
)


# ============================================================
# Plot settings
# ============================================================

# Set to None to use all discovered noise models.
# Or explicitly restrict, e.g.
# SELECT_NOISE_MODELS = ["gaussian1", "studentt_df3", "failure_low_p08"]
SELECT_NOISE_MODELS = None


# Set to None to use all discovered sample rules.
# Or explicitly restrict, e.g.
# SELECT_NS_RULES = ["inv_delta()"]
SELECT_NS_RULES = None


# Set to None to use all discovered aggregators.
# Or explicitly restrict, e.g.
# SELECT_AGGREGATORS = ["mean", "median"]
SELECT_AGGREGATORS = None


# Tau values.
# tau3 = 1e-3 is often the most useful stricter tolerance.
TAUS = {
    "tau1": 1e-1,
    "tau2": 1e-2,
    "tau3": 1e-3,
    "tau4": 1e-4,
    "tau5": 1e-5,
}


# If None, use the maximum available budget in the loaded data.
# Otherwise set a number, e.g. 20 or 500.
MAX_BUDGET_IN_GRADIENTS = None
# MAX_BUDGET_IN_GRADIENTS = 200


# Legacy setting kept only so this file stays close to the previous version.
# The corrected data-profile function below now ALWAYS treats every
# (problem, run) pair as one observation, following Lindon's equation.
AVERAGE_OVER_RUNS_FOR_DATA_PROFILE = True  # not used by corrected function


# Use smooth objective values or noisy objective values.
PROFILE_TYPES = ["smooth", "noisy"]


# Use LaTeX text rendering.
# Set False to avoid LaTeX installation issues on Mac / RCP.
USE_TEX = False


# ============================================================
# Label helpers
# ============================================================

def pretty_noise_label(noise: str) -> str:
    label = noise.replace("_", " ")
    label = label.replace("studentt", "Student-t")
    label = label.replace("df", "df")
    label = label.replace("gaussian1", "Gaussian")
    label = label.replace("failure low p08", "Failure low p=0.8")
    label = label.replace(
        "failure uniform1e4 p08",
        "Failure uniform 1e4 p=0.8"
    )
    return label


def pretty_agg_label(agg: str) -> str:
    label_map = {
        "mean": "Mean",
        "median": "Median",
        "trimmed_mean_10": "Tm",
        "mom_K5": "MoM",
    }
    return label_map.get(agg, agg)


def safe_name(s: str) -> str:
    return (
        str(s)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


# ============================================================
# IO helpers
# ============================================================

def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_converted_root():
    """
    Fixed converted-results folder.

    This version only reads:
        converted_results_cutest_parallel

    It will not search other converted_results_cutest* folders.
    """

    print(
        "[DEBUG] Requested converted root:",
        CONVERTED_ROOT
    )

    print(
        "[DEBUG] Exists:",
        os.path.isdir(CONVERTED_ROOT)
    )

    if not os.path.isdir(CONVERTED_ROOT):
        raise FileNotFoundError(
            "The required converted folder does not exist:\n"
            f"{CONVERTED_ROOT}\n\n"
            "Please check whether the converter has written output "
            "to this folder."
        )

    return CONVERTED_ROOT


def load_converted_results(converted_root: str):
    """
    Load all converted JSON files except index.json.

    Returns
    -------
    records : list[dict]
        Each record corresponds to one converted file:
        one problem under one run_name, containing multiple repeated runs.
    """

    pattern = os.path.join(
        converted_root,
        "*",
        "*.json"
    )

    paths = sorted(
        glob.glob(pattern)
    )

    print(
        "[DEBUG] Converted root:",
        converted_root
    )

    print(
        "[DEBUG] Converted JSON files found:",
        len(paths)
    )

    records = []

    for path in paths:

        if os.path.basename(path) == "index.json":
            continue

        try:

            data = read_json(path)

            # Require converted format.
            if "runs" not in data:
                continue

            record = {
                "path":
                    path,

                "run_name":
                    data.get("run_name"),

                "problem_name":
                    data.get("problem_name"),

                "objfun_name":
                    data.get("objfun_name"),

                "noise_model":
                    data.get("noise_model"),

                "aggregator":
                    data.get("aggregator"),

                "ns_rule":
                    data.get("ns_rule"),

                "n":
                    int(
                        data.get(
                            "n",
                            0
                        )
                    ),

                "nruns":
                    int(
                        data.get(
                            "nruns",
                            len(
                                data.get(
                                    "runs",
                                    []
                                )
                            )
                        )
                    ),

                "runs":
                    data.get(
                        "runs",
                        []
                    ),
            }

            records.append(record)

        except Exception as e:

            print(
                f"[WARN] Failed to read {path}: "
                f"{type(e).__name__}: {e}"
            )

    if not records:
        raise RuntimeError(
            f"No converted result files found in "
            f"{converted_root}"
        )

    return records


# ============================================================
# Profile calculations
# ============================================================

def get_values_from_run(
    run: dict,
    profile_type: str
):

    if profile_type == "smooth":

        return np.asarray(
            run.get(
                "fvals_smooth",
                []
            ),
            dtype=float
        )

    if profile_type == "noisy":

        return np.asarray(
            run.get(
                "fvals_noisy",
                []
            ),
            dtype=float
        )

    raise ValueError(
        f"Unknown profile_type: "
        f"{profile_type}"
    )


def get_budget_axis_from_run(run: dict):
    """
    Keep the existing budget definition from the previous plotting code.
    """

    axis = run.get(
        "budget_axis",
        None
    )

    if (
        axis is not None
        and len(axis) > 0
    ):
        return np.asarray(
            axis,
            dtype=float
        )

    nf = int(
        run.get(
            "nf",
            0
        )
    )

    n = int(
        run.get(
            "n",
            0
        )
    )

    if nf <= 0:
        vals = run.get(
            "fvals_smooth",
            []
        )

        nf = len(vals)

    if n > 0:

        return (
            np.arange(
                nf,
                dtype=float
            )
            / float(n + 1)
        )

    return np.arange(
        nf,
        dtype=float
    )


def compute_problem_fmin(
    records,
    profile_type: str
):
    """
    For each noise model and each problem, compute f_min across all
    aggregators, sampling rules, and repeated runs within that noise model.

    Returns
    -------
    fmins : dict

        fmins[(noise_model, problem_name)] = best observed value

    Thus, different noise models have different f_min values for the
    same CUTEst problem.

    This rule is applied to both:
        - smooth profiles, using fvals_smooth;
        - noisy profiles, using fvals_noisy.
    """

    fmins = {}

    for rec in records:

        noise = rec[
            "noise_model"
        ]

        problem = rec[
            "problem_name"
        ]

        vals_all = []

        for run in rec["runs"]:

            vals = get_values_from_run(
                run,
                profile_type
            )

            vals = vals[
                np.isfinite(vals)
            ]

            if len(vals) > 0:

                vals_all.append(
                    float(
                        np.min(vals)
                    )
                )

        if not vals_all:
            continue

        this_min = float(
            np.min(vals_all)
        )

        fmin_key = (
            noise,
            problem
        )

        if fmin_key not in fmins:

            fmins[
                fmin_key
            ] = this_min

        else:

            fmins[
                fmin_key
            ] = min(
                fmins[
                    fmin_key
                ],
                this_min
            )

    return fmins


def solve_budget_for_run(
    run: dict,
    fmin: float,
    tau: float,
    profile_type: str
):
    """
    Return first budget where

        f <= fmin + tau * (f0 - fmin).

    If not solved, return np.inf.

    This is intentionally kept the same as the previous version.
    """

    vals = get_values_from_run(
        run,
        profile_type
    )

    budgets = get_budget_axis_from_run(
        run
    )

    if (
        len(vals) == 0
        or len(budgets) == 0
    ):

        return np.inf

    L = min(
        len(vals),
        len(budgets)
    )

    vals = vals[:L]
    budgets = budgets[:L]

    finite_mask = (
        np.isfinite(vals)
        &
        np.isfinite(budgets)
    )

    if not np.any(
        finite_mask
    ):

        return np.inf

    vals = vals[
        finite_mask
    ]

    budgets = budgets[
        finite_mask
    ]

    f0 = float(
        vals[0]
    )

    if (
        not np.isfinite(f0)
        or not np.isfinite(fmin)
    ):

        return np.inf

    # Keep the existing convention:
    # if there is essentially no observed improvement range,
    # count this run as solved at budget 0.
    if (
        abs(f0 - fmin)
        <=
        1.0e-14
        *
        max(
            1.0,
            abs(f0)
        )
    ):

        return 0.0

    target = (
        fmin
        +
        tau
        *
        (
            f0 - fmin
        )
    )

    solved_idx = np.where(
        vals <= target
    )[0]

    if len(solved_idx) == 0:

        return np.inf

    return float(
        budgets[
            int(
                solved_idx[0]
            )
        ]
    )


def build_solve_budgets(
    records,
    tau: float,
    profile_type: str,
):
    """
    Build solve budgets using a separate f_min for each
    (noise model, problem) pair.

    Returns
    -------
    solve_budgets : dict

        solve_budgets[
            (noise, ns_rule, aggregator)
        ][problem] = list of run budgets
    """

    # Noise-specific f_min:
    #
    # fmins[(noise_model, problem_name)]
    #
    fmins = compute_problem_fmin(
        records,
        profile_type
    )

    print(
        f"[DEBUG] Number of noise-specific fmins "
        f"for {profile_type}: {len(fmins)}"
    )

    solve_budgets = defaultdict(
        lambda: defaultdict(list)
    )

    for rec in records:

        noise = rec[
            "noise_model"
        ]

        ns_rule = rec[
            "ns_rule"
        ]

        agg = rec[
            "aggregator"
        ]

        problem = rec[
            "problem_name"
        ]

        fmin_key = (
            noise,
            problem
        )

        if fmin_key not in fmins:
            continue

        fmin = fmins[
            fmin_key
        ]

        algorithm_key = (
            noise,
            ns_rule,
            agg
        )

        for run in rec[
            "runs"
        ]:

            b = solve_budget_for_run(
                run=run,
                fmin=fmin,
                tau=tau,
                profile_type=profile_type
            )

            solve_budgets[
                algorithm_key
            ][problem].append(b)

    return solve_budgets


def get_available_groups(records):
    """
    Return discovered noise models, ns rules, aggregators.
    """

    noise_models = sorted(
        set(
            r["noise_model"]
            for r in records
        )
    )

    ns_rules = sorted(
        set(
            r["ns_rule"]
            for r in records
        )
    )

    aggregators = sorted(
        set(
            r["aggregator"]
            for r in records
        )
    )

    # Preferred aggregator order.
    preferred = [
        "mean",
        "median",
        "trimmed_mean_10",
        "mom_K5"
    ]

    aggregators = (
        [
            a
            for a in preferred
            if a in aggregators
        ]
        +
        [
            a
            for a in aggregators
            if a not in preferred
        ]
    )

    return (
        noise_models,
        ns_rules,
        aggregators
    )


def filter_selected(
    discovered,
    selected
):

    if selected is None:
        return discovered

    return [
        x
        for x in selected
        if x in discovered
    ]


# ============================================================
# Data profile
# ============================================================

def build_data_profile_curves(
    solve_budgets,
    noise,
    ns_rule,
    aggregators,
    xvals
):
    """
    Build data profile y-values using all repeated runs directly.

    Lindon's multiple-run definition:

        DP_a(alpha)
        =
        # {(p,r): b_{a,r}(p) <= alpha}
        --------------------------------
              #problems * #runs

    Therefore each repeated run contributes one observation.

    With exactly the same number of runs for every problem, this is
    mathematically equivalent to first averaging the success indicators
    over runs for each problem and then averaging over problems.

    The direct implementation is used here so that the code matches
    Lindon's equation explicitly.
    """

    curves = {}

    # Use the union of problems across aggregators.
    problems = sorted(
        set(
            problem
            for agg in aggregators
            for problem in solve_budgets.get(
                (
                    noise,
                    ns_rule,
                    agg
                ),
                {}
            ).keys()
        )
    )

    if not problems:
        return curves

    # --------------------------------------------------------
    # Check that every algorithm/problem pair has the same
    # number of repeated runs.
    # --------------------------------------------------------

    run_counts = []

    for problem in problems:

        for agg in aggregators:

            budgets = solve_budgets.get(
                (
                    noise,
                    ns_rule,
                    agg
                ),
                {}
            ).get(
                problem,
                []
            )

            if not budgets:

                raise RuntimeError(
                    f"Missing runs for "
                    f"noise={noise}, "
                    f"ns={ns_rule}, "
                    f"agg={agg}, "
                    f"problem={problem}"
                )

            run_counts.append(
                len(budgets)
            )

    unique_run_counts = sorted(
        set(run_counts)
    )

    if len(
        unique_run_counts
    ) != 1:

        raise RuntimeError(
            f"Inconsistent repeated-run counts "
            f"for noise={noise}, "
            f"ns={ns_rule}: "
            f"{unique_run_counts}"
        )

    nruns = unique_run_counts[0]

    expected_nobs = (
        len(problems)
        *
        nruns
    )

    # --------------------------------------------------------
    # Build each algorithm's empirical data-profile curve.
    # --------------------------------------------------------

    for agg in aggregators:

        key = (
            noise,
            ns_rule,
            agg
        )

        all_budgets = []

        for problem in problems:

            budgets = np.asarray(
                solve_budgets.get(
                    key,
                    {}
                ).get(
                    problem,
                    []
                ),
                dtype=float
            )

            all_budgets.extend(
                budgets.tolist()
            )

        all_budgets = np.asarray(
            all_budgets,
            dtype=float
        )

        if len(
            all_budgets
        ) != expected_nobs:

            raise RuntimeError(
                f"Unexpected number of observations "
                f"for agg={agg}, "
                f"noise={noise}, "
                f"ns={ns_rule}: "
                f"{len(all_budgets)} "
                f"!= {expected_nobs}"
            )

        yvals = []

        for x in xvals:

            yvals.append(
                float(
                    np.mean(
                        all_budgets <= x
                    )
                )
            )

        curves[
            agg
        ] = np.asarray(
            yvals,
            dtype=float
        )

    return curves


# ============================================================
# Performance profile
# ============================================================

def build_perf_profile_curves(
    solve_budgets,
    noise,
    ns_rule,
    aggregators
):
    """
    Build performance profile ratios and curves using all repeated runs.

    Lindon's multiple-run definition:

    For each problem p,

        b_star(p)
        =
        min_{a,r} b_{a,r}(p),

    where the minimum is taken over all algorithms a and all repeated
    runs r.

    For algorithm a,

        PP_a(alpha)
        =
        # {(p,r): b_{a,r}(p) <= alpha * b_star(p)}
        ------------------------------------------------
                     #problems * #runs

    Therefore every run contributes separately.

    IMPORTANT:
    This version does NOT first take the median solve budget over the
    repeated runs.
    """

    problems = sorted(
        set(
            problem
            for agg in aggregators
            for problem in solve_budgets.get(
                (
                    noise,
                    ns_rule,
                    agg
                ),
                {}
            ).keys()
        )
    )

    if not problems:

        return None, {}

    # --------------------------------------------------------
    # Check repeated-run counts.
    # --------------------------------------------------------

    run_counts = []

    for problem in problems:

        for agg in aggregators:

            budgets = solve_budgets.get(
                (
                    noise,
                    ns_rule,
                    agg
                ),
                {}
            ).get(
                problem,
                []
            )

            if not budgets:

                raise RuntimeError(
                    f"Missing runs for "
                    f"noise={noise}, "
                    f"ns={ns_rule}, "
                    f"agg={agg}, "
                    f"problem={problem}"
                )

            run_counts.append(
                len(budgets)
            )

    unique_run_counts = sorted(
        set(run_counts)
    )

    if len(
        unique_run_counts
    ) != 1:

        raise RuntimeError(
            f"Inconsistent repeated-run counts "
            f"for noise={noise}, "
            f"ns={ns_rule}: "
            f"{unique_run_counts}"
        )

    nruns = unique_run_counts[0]

    expected_nobs = (
        len(problems)
        *
        nruns
    )

    ratios_by_agg = {
        agg: []
        for agg in aggregators
    }

    # --------------------------------------------------------
    # One problem at a time.
    # --------------------------------------------------------

    for problem in problems:

        # ----------------------------------------------------
        # Lindon's b_star(p):
        #
        # minimum solve budget over ALL algorithms AND
        # ALL repeated runs for this problem.
        # ----------------------------------------------------

        all_problem_budgets = []

        for agg in aggregators:

            key = (
                noise,
                ns_rule,
                agg
            )

            budgets = np.asarray(
                solve_budgets.get(
                    key,
                    {}
                ).get(
                    problem,
                    []
                ),
                dtype=float
            )

            finite_budgets = budgets[
                np.isfinite(
                    budgets
                )
            ]

            if len(
                finite_budgets
            ) > 0:

                all_problem_budgets.extend(
                    finite_budgets.tolist()
                )

        # ----------------------------------------------------
        # If no algorithm/run solved this problem,
        # every run gets an infinite performance ratio.
        # ----------------------------------------------------

        if not all_problem_budgets:

            for agg in aggregators:

                ratios_by_agg[
                    agg
                ].extend(
                    [np.inf]
                    *
                    nruns
                )

            continue

        best = float(
            np.min(
                all_problem_budgets
            )
        )

        # ----------------------------------------------------
        # Each run contributes one performance ratio.
        # ----------------------------------------------------

        for agg in aggregators:

            key = (
                noise,
                ns_rule,
                agg
            )

            budgets = np.asarray(
                solve_budgets.get(
                    key,
                    {}
                ).get(
                    problem,
                    []
                ),
                dtype=float
            )

            for b in budgets:

                if not np.isfinite(
                    b
                ):

                    ratio = np.inf

                elif best > 0:

                    ratio = max(
                        1.0,
                        float(
                            b / best
                        )
                    )

                else:

                    # The existing budget convention can produce
                    # b_star(p)=0.
                    #
                    # If b_star(p)=0, then
                    #
                    #     b <= alpha * b_star(p)
                    #
                    # can hold for finite alpha only if b=0.
                    #
                    ratio = (
                        1.0
                        if b <= 0
                        else np.inf
                    )

                ratios_by_agg[
                    agg
                ].append(
                    ratio
                )

    # --------------------------------------------------------
    # Each algorithm should now contain
    #
    #     #problems * #runs
    #
    # observations.
    # --------------------------------------------------------

    for agg in aggregators:

        if len(
            ratios_by_agg[
                agg
            ]
        ) != expected_nobs:

            raise RuntimeError(
                f"Unexpected number of performance "
                f"observations for agg={agg}, "
                f"noise={noise}, "
                f"ns={ns_rule}: "
                f"{len(ratios_by_agg[agg])} "
                f"!= {expected_nobs}"
            )

    # --------------------------------------------------------
    # Find largest finite performance ratio.
    # --------------------------------------------------------

    finite_ratios = []

    for ratios in ratios_by_agg.values():

        finite_ratios.extend(
            [
                r
                for r in ratios
                if np.isfinite(r)
            ]
        )

    if not finite_ratios:

        return None, {}

    max_ratio = max(
        2.0,
        float(
            np.max(
                finite_ratios
            )
        )
    )

    # --------------------------------------------------------
    # Build a dense base-2 logarithmic x-grid.
    #
    # Do not first collapse repeated runs with a median.
    # --------------------------------------------------------

    xvals = np.unique(
        np.concatenate(
            [
                np.array(
                    [1.0]
                ),
                np.logspace(
                    0.0,
                    np.log2(
                        max_ratio
                    ),
                    200,
                    base=2.0
                ),
            ]
        )
    )

    curves = {}

    for agg, ratios in ratios_by_agg.items():

        ratios = np.asarray(
            ratios,
            dtype=float
        )

        yvals = []

        for x in xvals:

            yvals.append(
                float(
                    np.mean(
                        ratios <= x
                    )
                )
            )

        curves[
            agg
        ] = np.asarray(
            yvals,
            dtype=float
        )

    # Useful check:
    # with 29 problems and 10 runs, this should report 290.
    print(
        f"[DEBUG PERF] "
        f"noise={noise}, "
        f"ns={ns_rule}: "
        f"{len(problems)} problems "
        f"x {nruns} runs "
        f"= {expected_nobs} observations; "
        f"max finite ratio="
        f"{max_ratio:.6g}"
    )

    return (
        xvals,
        curves
    )


# ============================================================
# Plotting
# ============================================================

def linestyle_for_agg(agg: str):

    style_map = {
        "mean":
            "-",

        "median":
            "--",

        "trimmed_mean_10":
            "-.",

        "mom_K5":
            ":",
    }

    return style_map.get(
        agg,
        "-"
    )


def plot_curves(
    xvals,
    curves,
    aggregators,
    filename,
    title,
    xlabel,
    ylabel,
    logx=False
):
    """
    Plot one data profile or performance profile.

    For performance profiles (logx=True), use a base-2 log scale and
    explicitly label the x-axis as

        1, 2, 4, 8, 16, ...

    This fixes the previous figure where only 1.00 appeared.
    """

    os.makedirs(
        os.path.dirname(
            filename
        ),
        exist_ok=True
    )

    plt.figure()
    plt.clf()

    plt.rc(
        "text",
        usetex=USE_TEX
    )

    plt.rc(
        "font",
        family="serif"
    )

    ax = plt.gca()

    for idx, agg in enumerate(
        aggregators
    ):

        if agg not in curves:
            continue

        ax.plot(
            xvals,
            curves[agg],
            linestyle=linestyle_for_agg(
                agg
            ),
            linewidth=2.0,
            label=pretty_agg_label(
                agg
            ),
        )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        xlabel
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_ylim(
        0.0,
        1.0
    )

    ax.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.4
    )

    ax.legend(
        loc="best"
    )

    if logx:

        # ----------------------------------------------------
        # Use base 2 for the performance-profile x-axis.
        # ----------------------------------------------------

        ax.set_xscale(
            "log",
            base=2
        )

        xmax = max(
            2.0,
            float(
                np.max(
                    xvals
                )
            )
        )

        # Extend the displayed axis to the next power of 2.
        max_power = int(
            np.ceil(
                np.log2(
                    xmax
                )
            )
        )

        xmax_tick = float(
            2 ** max_power
        )

        ticks = (
            2.0
            **
            np.arange(
                0,
                max_power + 1
            )
        )

        ax.set_xlim(
            1.0,
            xmax_tick
        )

        ax.set_xticks(
            ticks
        )

        ax.set_xticklabels(
            [
                f"{tick:g}"
                for tick in ticks
            ]
        )

        ax.minorticks_off()

    plt.tight_layout()

    plt.savefig(
        filename,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    converted_root = (
        find_converted_root()
    )

    records = (
        load_converted_results(
            converted_root
        )
    )

    (
        discovered_noises,
        discovered_ns_rules,
        discovered_aggs
    ) = get_available_groups(
        records
    )

    noise_models = filter_selected(
        discovered_noises,
        SELECT_NOISE_MODELS
    )

    ns_rules = filter_selected(
        discovered_ns_rules,
        SELECT_NS_RULES
    )

    aggregators = filter_selected(
        discovered_aggs,
        SELECT_AGGREGATORS
    )

    print(
        "=" * 60
    )

    print(
        "CUTEst plotting script"
    )

    print(
        f"Project root:     "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Converted root:   "
        f"{converted_root}"
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Loaded records:   "
        f"{len(records)}"
    )

    print(
        f"Noise models:     "
        f"{noise_models}"
    )

    print(
        f"Sample rules:     "
        f"{ns_rules}"
    )

    print(
        f"Aggregators:      "
        f"{aggregators}"
    )

    print(
        f"Taus:             "
        f"{TAUS}"
    )

    print(
        "=" * 60
    )

    # ========================================================
    # Determine maximum data-profile budget.
    # ========================================================

    if (
        MAX_BUDGET_IN_GRADIENTS
        is None
    ):

        max_budget = 0.0

        for rec in records:

            for run in rec[
                "runs"
            ]:

                axis = (
                    get_budget_axis_from_run(
                        run
                    )
                )

                if len(axis) > 0:

                    finite_axis = axis[
                        np.isfinite(
                            axis
                        )
                    ]

                    if len(
                        finite_axis
                    ) > 0:

                        max_budget = max(
                            max_budget,
                            float(
                                np.max(
                                    finite_axis
                                )
                            )
                        )

        if max_budget <= 0:

            max_budget = 1.0

    else:

        max_budget = float(
            MAX_BUDGET_IN_GRADIENTS
        )

    xvals_data = np.linspace(
        0.0,
        max_budget,
        201
    )

    print(
        f"Data profile max budget: "
        f"{max_budget}"
    )

    # ========================================================
    # Smooth / noisy profiles.
    # ========================================================

    for profile_type in PROFILE_TYPES:

        print(
            f"[INFO] Building profiles "
            f"using {profile_type} values."
        )

        for tau_key, tau in TAUS.items():

            print(
                f"[INFO] "
                f"tau={tau_key} ({tau})"
            )

            solve_budgets = (
                build_solve_budgets(
                    records,
                    tau=tau,
                    profile_type=profile_type
                )
            )

            for noise in noise_models:

                for ns_rule in ns_rules:

                    subdir = os.path.join(
                        OUTPUT_DIR,
                        noise,
                        ns_rule
                    )

                    os.makedirs(
                        subdir,
                        exist_ok=True
                    )

                    # ========================================
                    # Data profile
                    # ========================================

                    data_curves = (
                        build_data_profile_curves(
                            solve_budgets=
                                solve_budgets,

                            noise=
                                noise,

                            ns_rule=
                                ns_rule,

                            aggregators=
                                aggregators,

                            xvals=
                                xvals_data,
                        )
                    )

                    if data_curves:

                        data_file = os.path.join(
                            subdir,
                            (
                                f"{noise}_"
                                f"{ns_rule}_"
                                f"{tau_key}_"
                                f"data_profile_"
                                f"{profile_type}"
                                f".pdf"
                            )
                        )

                        title = (
                            f"{pretty_noise_label(noise)}, "
                            f"{ns_rule}, "
                            f"{tau_key}, "
                            f"{profile_type}"
                        )

                        plot_curves(
                            xvals=
                                xvals_data,

                            curves=
                                data_curves,

                            aggregators=
                                aggregators,

                            filename=
                                data_file,

                            title=
                                title,

                            xlabel=
                                "Budget in evaluations / (n+1)",

                            ylabel=
                                "Proportion of problem-run pairs solved",

                            logx=
                                False,
                        )

                        print(
                            f"[PLOT] "
                            f"{data_file}"
                        )

                    # ========================================
                    # Performance profile
                    # ========================================

                    (
                        xvals_perf,
                        perf_curves
                    ) = (
                        build_perf_profile_curves(
                            solve_budgets=
                                solve_budgets,

                            noise=
                                noise,

                            ns_rule=
                                ns_rule,

                            aggregators=
                                aggregators,
                        )
                    )

                    if (
                        xvals_perf is None
                        or not perf_curves
                    ):

                        print(
                            f"[SKIP PERF] "
                            f"noise={noise}, "
                            f"ns={ns_rule}, "
                            f"tau={tau_key}, "
                            f"profile={profile_type}: "
                            f"no finite solved budgets",
                            flush=True,
                        )

                    if (
                        xvals_perf is not None
                        and perf_curves
                    ):

                        perf_file = os.path.join(
                            subdir,
                            (
                                f"{noise}_"
                                f"{ns_rule}_"
                                f"{tau_key}_"
                                f"perf_profile_"
                                f"{profile_type}"
                                f".pdf"
                            )
                        )

                        title = (
                            f"{pretty_noise_label(noise)}, "
                            f"{ns_rule}, "
                            f"{tau_key}, "
                            f"{profile_type}"
                        )

                        plot_curves(
                            xvals=
                                xvals_perf,

                            curves=
                                perf_curves,

                            aggregators=
                                aggregators,

                            filename=
                                perf_file,

                            title=
                                title,

                            xlabel=
                                "Performance ratio "
                                "(budget / best budget)",

                            ylabel=
                                "Proportion of problem-run pairs solved",

                            logx=
                                True,
                        )

                        print(
                            f"[PLOT] "
                            f"{perf_file}"
                        )

    print(
        "=" * 60
    )

    print(
        "All plots completed."
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()
