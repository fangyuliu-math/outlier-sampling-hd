"""
CUTEst smooth tau2 plotting, method-specific fmin, streaming two-pass memory-safe variant.

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


# Keep old plots untouched; this experiment uses method-specific smooth fmin.
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "plots_cutest_profiles_parallel_smooth_tau2_per_group_fmin"
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


# Examine tau2 first. Other tau values can be restored later.
TAUS = {
    "tau2": 1e-2,
}


# If None, use the maximum available budget in the loaded data.
# Otherwise set a number, e.g. 20 or 500.
MAX_BUDGET_IN_GRADIENTS = None
# MAX_BUDGET_IN_GRADIENTS = 200


# Legacy setting kept only so this file stays close to the previous version.
# The corrected data-profile function below now ALWAYS treats every
# (problem, run) pair as one observation, following Lindon's equation.
AVERAGE_OVER_RUNS_FOR_DATA_PROFILE = True  # not used by corrected function


# Plot only the true (smooth) objective values; noisy profiles are disabled.
PROFILE_TYPES = ["smooth"]


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


def compute_problem_fmin(records, profile_type: str):
    """Return method-specific fmin, pooled over repeated runs only.

    Key: (noise_model, ns_rule, aggregator, problem_name).

    For smooth plots, each distinct experiment setting is compared against
    the best finite true objective value among *its own* ten runs.  We do not
    pool over other sampling rules or aggregators, and do not compute fmin
    independently for each run.
    """
    if profile_type != "smooth":
        raise ValueError("This plotting variant is restricted to smooth values")

    fmins = {}
    for rec in records:
        key = (
            rec["noise_model"],
            rec["ns_rule"],
            rec["aggregator"],
            rec["problem_name"],
        )
        for run in rec["runs"]:
            vals = get_values_from_run(run, "smooth")
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            candidate = float(np.min(vals))
            fmins[key] = min(fmins.get(key, np.inf), candidate)
    return fmins


def save_smooth_fmins(records, fmins, filename):
    """Save each fmin and check how often it equals the start value."""
    rec_by_key = {
        (rec["noise_model"], rec["ns_rule"],
         rec["aggregator"], rec["problem_name"]): rec
        for rec in records
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "noise_model", "ns_rule", "aggregator", "problem_name",
            "fmin_smooth", "runs", "runs_with_f0_near_fmin",
        ])
        for key, fmin in sorted(fmins.items()):
            runs = rec_by_key[key]["runs"]
            near = 0
            for run in runs:
                vals = get_values_from_run(run, "smooth")
                vals = vals[np.isfinite(vals)]
                if vals.size and abs(float(vals[0]) - fmin) <= (
                    1.0e-14 * max(1.0, abs(float(vals[0])))
                ):
                    near += 1
            writer.writerow([*key, fmin, len(runs), near])
    print(f"[SAVED FMIN] {filename}: {len(fmins)} method-specific references")


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

    For method-specific fmin, no improvement (f0 approximately fmin) is
    NOT counted as solving; otherwise a stagnant method appears successful
    at budget zero.
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

    # A run with no available improvement range should not be reported
    # as having solved an optimization task at budget zero.
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

        return np.inf

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
    fmins=None,
):
    """
    Build solve budgets using a distinct smooth f_min for each
    (noise model, sampling rule, aggregator, problem) group.

    Returns
    -------
    solve_budgets : dict

        solve_budgets[
            (noise, ns_rule, aggregator)
        ][problem] = list of run budgets
    """

    # fmins[(noise_model, ns_rule, aggregator, problem_name)]
    if fmins is None:
        fmins = compute_problem_fmin(records, profile_type)

    print(
        f"[DEBUG] Number of method-specific fmins "
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
            ns_rule,
            agg,
            problem,
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
        finite_ratios.extend([r for r in ratios if np.isfinite(r)])

    if not finite_ratios:
        return None, {}

    # Keep the performance-profile x-axis fixed at 1--32,
    # consistent with the original low-dimensional plotting code.
    max_observed_ratio = float(np.max(finite_ratios))

    xvals = np.logspace(
        0.0,
        5.0,
        200,
        base=2.0,
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
        f"[DEBUG PERF] noise={noise}, ns={ns_rule}: "
        f"{len(problems)} problems x {nruns} runs = {expected_nobs} observations; "
        f"max observed finite ratio={max_observed_ratio:.6g}; "
        f"plot x-axis fixed at 1--32"
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
        # Use the same fixed performance-profile x-axis as the
        # original low-dimensional paper plots:
        #
        #     1, 2, 4, 8, 16, 32
        #
        ax.set_xscale(
            "log",
            base=2,
        )

        ticks = [
            1,
            2,
            4,
            8,
            16,
            32,
        ]

        ax.set_xlim(
            1.0,
            32.0,
        )

        ax.set_xticks(
            ticks
        )

        ax.set_xticklabels(
            [
                str(tick)
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
    """Two-pass JSON loading so that large converted arrays are never stored together.

    Pass 1: compute fmin by (noise, sampling rule, aggregator, problem).
    Pass 2: compute each run's first success and aggregate only scalar budgets.
    Plot/statistics remain based on the original helper functions.
    """
    import gc

    root = find_converted_root()
    paths = sorted(glob.glob(os.path.join(root, "*", "*.json")))
    paths = [p for p in paths if os.path.basename(p) != "index.json"]
    print(f"[START] {len(paths)} converted JSON files to process", flush=True)
    if not paths:
        raise RuntimeError(f"No converted JSON files found under {root}")

    fmins = {}
    metadata = []
    # PHASE 1: one JSON at a time, keeping only metadata and scalar fmin.
    for i, path in enumerate(paths, 1):
        data = read_json(path)
        runs = data.get("runs", [])
        if not isinstance(runs, list) or not runs:
            print(f"[WARN] No runs: {path}", flush=True)
            del data
            continue
        meta = {
            "path": path,
            "noise_model": str(data["noise_model"]),
            "ns_rule": str(data["ns_rule"]),
            "aggregator": str(data["aggregator"]),
            "problem_name": str(data["problem_name"]),
            "n": int(data.get("n", 0)),
            "nruns": len(runs),
        }
        metadata.append(meta)
        key = (meta["noise_model"], meta["ns_rule"],
               meta["aggregator"], meta["problem_name"])
        for run in runs:
            vals = get_values_from_run(run, "smooth")
            if vals.size:
                finite = vals[np.isfinite(vals)]
                if finite.size:
                    fmins[key] = min(fmins.get(key, np.inf), float(np.min(finite)))
        del runs, data
        if i % 25 == 0 or i == len(paths):
            gc.collect()
            print(f"[PASS 1/2] {i}/{len(paths)} files", flush=True)

    print(f"[FMIN] Computed {len(fmins)} distinct smooth fmin values", flush=True)
    if not fmins:
        raise RuntimeError("No finite smooth fmin values were found")

    # PHASE 2: one JSON at a time, keeping only success budgets and small counters.
    solve_budgets = defaultdict(lambda: defaultdict(list))
    counts = defaultdict(lambda: [0, 0])  # total run count, starts near fmin
    max_budget = 0.0
    tau = float(TAUS["tau2"])
    for i, meta in enumerate(metadata, 1):
        data = read_json(meta["path"])
        key = (meta["noise_model"], meta["ns_rule"],
               meta["aggregator"], meta["problem_name"])
        if key not in fmins:
            del data
            continue
        fmin = fmins[key]
        method_key = (meta["noise_model"], meta["ns_rule"], meta["aggregator"])
        for run in data["runs"]:
            vals = get_values_from_run(run, "smooth")
            axis = get_budget_axis_from_run(run)
            if axis.size:
                finite_axis = axis[np.isfinite(axis)]
                if finite_axis.size:
                    max_budget = max(max_budget, float(np.max(finite_axis)))
            finite_vals = vals[np.isfinite(vals)]
            counts[key][0] += 1
            if finite_vals.size and abs(float(finite_vals[0]) - fmin) <= (
                    1.0e-14 * max(1.0, abs(float(finite_vals[0])))):
                counts[key][1] += 1
            b = solve_budget_for_run(
                run=run, fmin=fmin, tau=tau, profile_type="smooth"
            )
            solve_budgets[method_key][meta["problem_name"]].append(b)
        del data
        if i % 25 == 0 or i == len(metadata):
            gc.collect()
            print(f"[PASS 2/2] {i}/{len(metadata)} files", flush=True)

    noises, ns_rules, aggregators = get_available_groups(metadata)
    noises = filter_selected(noises, SELECT_NOISE_MODELS)
    ns_rules = filter_selected(ns_rules, SELECT_NS_RULES)
    aggregators = filter_selected(aggregators, SELECT_AGGREGATORS)
    print(f"[DATA] noises={noises}, sampling={ns_rules}, aggregators={aggregators}", flush=True)
    if not noises or not ns_rules or not aggregators:
        raise RuntimeError("Selection left no data")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fmin_path = os.path.join(OUTPUT_DIR, "fmin_smooth_per_group.csv")
    with open(fmin_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["noise_model", "ns_rule", "aggregator", "problem_name",
                         "fmin_smooth", "runs", "runs_with_f0_near_fmin"])
        for key, val in sorted(fmins.items()):
            writer.writerow([*key, val, counts[key][0], counts[key][1]])
    print(f"[SAVED FMIN] {fmin_path}", flush=True)

    if MAX_BUDGET_IN_GRADIENTS is not None:
        max_budget = float(MAX_BUDGET_IN_GRADIENTS)
    if max_budget <= 0:
        max_budget = 1.0
    xvals_data = np.linspace(0.0, max_budget, 201)
    print(f"[DATA] max budget={max_budget}", flush=True)
    success_rows = []
    for noise in noises:
        for ns in ns_rules:
            subdir = os.path.join(OUTPUT_DIR, noise, ns)
            os.makedirs(subdir, exist_ok=True)
            for agg in aggregators:
                by_problem = solve_budgets.get((noise, ns, agg), {})
                nobs = sum(len(b) for b in by_problem.values())
                nsuccess = sum(np.isfinite(b) for runs in by_problem.values() for b in runs)
                success_rows.append([noise, ns, agg, "tau2", len(by_problem), nobs, nsuccess])
                print(f"[SUCCESS] {noise} {ns} {agg}: {nsuccess}/{nobs}", flush=True)
            title = f"{pretty_noise_label(noise)}, {ns}, tau2, smooth (own fmin)"
            data_curves = build_data_profile_curves(
                solve_budgets, noise, ns, aggregators, xvals_data
            )
            if data_curves:
                data_path = os.path.join(subdir, f"{noise}_{ns}_tau2_data_profile_smooth.pdf")
                plot_curves(xvals_data, data_curves, aggregators, data_path,
                            title, "Budget in evaluations / (n+1)",
                            "Proportion of problem-run pairs solved", logx=False)
                print(f"[PLOT] {data_path}", flush=True)
            xperf, perf_curves = build_perf_profile_curves(
                solve_budgets, noise, ns, aggregators
            )
            if xperf is None or not perf_curves:
                print(f"[SKIP PERF] {noise} {ns}: no finite solved budgets", flush=True)
            else:
                perf_path = os.path.join(subdir, f"{noise}_{ns}_tau2_perf_profile_smooth.pdf")
                plot_curves(xperf, perf_curves, aggregators, perf_path,
                            title, "Performance ratio (budget / best budget)",
                            "Proportion of problem-run pairs solved", logx=True)
                print(f"[PLOT] {perf_path}", flush=True)

    success_path = os.path.join(OUTPUT_DIR, "success_summary_smooth_tau2.csv")
    with open(success_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["noise_model", "ns_rule", "aggregator", "tau",
                         "problems", "observations", "successful_runs"])
        writer.writerows(success_rows)
    print(f"[SAVED SUCCESS] {success_path}", flush=True)
    print("All plots completed.", flush=True)
    print(f"Output directory: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
