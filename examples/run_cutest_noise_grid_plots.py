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

Definitions used here
---------------------
For a minimization problem, a run is considered solved at tolerance tau if

    f(x_k) <= f_min + tau * (f_0 - f_min),

where f_min is the best value found for that problem across all loaded
methods and runs, using either smooth values or noisy values depending
on the plot type.

Data profile:
    x-axis: budget in evaluations / (n + 1)
    y-axis: average proportion of successful runs over problems.

Performance profile:
    x-axis: solve budget / best solve budget among methods
    y-axis: proportion of problems solved within that ratio.

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
CONVERTED_ROOT = os.path.join(PROJECT_ROOT, "converted_results_cutest_parallel")

# If CONVERTED_ROOT is None, the script searches these candidates and uses
# the newest existing one.
# CANDIDATE_CONVERTED_ROOTS = [
#     os.path.join(PROJECT_ROOT, "converted_results_cutest_const30_budget10"),
#     os.path.join(PROJECT_ROOT, "converted_results_cutest_const10_budget20"),
#     os.path.join(PROJECT_ROOT, "converted_results_cutest_pilot"),
#     os.path.join(PROJECT_ROOT, "examples", "converted_results_cutest_const30_budget10"),
#     os.path.join(PROJECT_ROOT, "examples", "converted_results_cutest_const10_budget20"),
#     os.path.join(PROJECT_ROOT, "examples", "converted_results_cutest_pilot"),
# ]


OUTPUT_DIR = os.path.join(PROJECT_ROOT, "plots_cutest_profiles_parallel")


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
# MAX_BUDGET_IN_GRADIENTS = None
MAX_BUDGET_IN_GRADIENTS = 100


# If True, each problem contributes the average of its repeated runs.
# If False, each run is treated as an individual observation.
AVERAGE_OVER_RUNS_FOR_DATA_PROFILE = True


# Use smooth objective values or noisy objective values.
PROFILE_TYPES = ["smooth", "noisy"]


# Use LaTeX text rendering. Set False to avoid LaTeX installation issues on Mac.
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
    label = label.replace("failure uniform1e4 p08", "Failure uniform 1e4 p=0.8")
    return label


def pretty_agg_label(agg: str) -> str:
    label_map = {
        "mean": "Mean",
        "median": "Median",
        "trimmed_mean_10": "Trimmed mean",
        "mom_K5": "MoM",
    }
    return label_map.get(agg, agg)


def safe_name(s: str) -> str:
    return str(s).replace("/", "_").replace("\\", "_").replace(" ", "_")


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
        converted_results_cutest_pilot_100

    It will not search other converted_results_cutest* folders.
    """
    print("[DEBUG] Requested converted root:", CONVERTED_ROOT)
    print("[DEBUG] Exists:", os.path.isdir(CONVERTED_ROOT))

    if not os.path.isdir(CONVERTED_ROOT):
        raise FileNotFoundError(
            "The required converted folder does not exist:\n"
            f"{CONVERTED_ROOT}\n\n"
            "Please check whether the converter has written output to this folder."
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
    pattern = os.path.join(converted_root, "*", "*.json")
    paths = sorted(glob.glob(pattern))

    print("[DEBUG] Converted root:", converted_root)
    print("[DEBUG] Converted JSON files found:", len(paths))

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
                "path": path,
                "run_name": data.get("run_name"),
                "problem_name": data.get("problem_name"),
                "objfun_name": data.get("objfun_name"),
                "noise_model": data.get("noise_model"),
                "aggregator": data.get("aggregator"),
                "ns_rule": data.get("ns_rule"),
                "n": int(data.get("n", 0)),
                "nruns": int(data.get("nruns", len(data.get("runs", [])))),
                "runs": data.get("runs", []),
            }

            records.append(record)

        except Exception as e:
            print(f"[WARN] Failed to read {path}: {type(e).__name__}: {e}")

    if not records:
        raise RuntimeError(f"No converted result files found in {converted_root}")

    return records


# ============================================================
# Profile calculations
# ============================================================

def get_values_from_run(run: dict, profile_type: str):
    if profile_type == "smooth":
        return np.asarray(run.get("fvals_smooth", []), dtype=float)

    if profile_type == "noisy":
        return np.asarray(run.get("fvals_noisy", []), dtype=float)

    raise ValueError(f"Unknown profile_type: {profile_type}")


def get_budget_axis_from_run(run: dict):
    axis = run.get("budget_axis", None)

    if axis is not None and len(axis) > 0:
        return np.asarray(axis, dtype=float)

    nf = int(run.get("nf", 0))
    n = int(run.get("n", 0))

    if nf <= 0:
        vals = run.get("fvals_smooth", [])
        nf = len(vals)

    if n > 0:
        return np.arange(nf, dtype=float) / float(n + 1)

    return np.arange(nf, dtype=float)


def compute_problem_fmin(records, profile_type: str):
    """
    For each problem, compute f_min across all loaded methods and runs.
    """
    fmins = {}

    for rec in records:
        problem = rec["problem_name"]

        vals_all = []
        for run in rec["runs"]:
            vals = get_values_from_run(run, profile_type)
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                vals_all.append(np.min(vals))

        if not vals_all:
            continue

        this_min = float(np.min(vals_all))

        if problem not in fmins:
            fmins[problem] = this_min
        else:
            fmins[problem] = min(fmins[problem], this_min)

    return fmins


def solve_budget_for_run(run: dict, fmin: float, tau: float, profile_type: str):
    """
    Return first budget where f <= fmin + tau * (f0 - fmin).

    If not solved, return np.inf.
    """
    vals = get_values_from_run(run, profile_type)
    budgets = get_budget_axis_from_run(run)

    if len(vals) == 0 or len(budgets) == 0:
        return np.inf

    L = min(len(vals), len(budgets))
    vals = vals[:L]
    budgets = budgets[:L]

    finite_mask = np.isfinite(vals) & np.isfinite(budgets)
    if not np.any(finite_mask):
        return np.inf

    vals = vals[finite_mask]
    budgets = budgets[finite_mask]

    f0 = float(vals[0])

    if not np.isfinite(f0) or not np.isfinite(fmin):
        return np.inf

    # If there is essentially no observed improvement range, count as solved at 0.
    if abs(f0 - fmin) <= 1.0e-14 * max(1.0, abs(f0)):
        return 0.0

    target = fmin + tau * (f0 - fmin)

    solved_idx = np.where(vals <= target)[0]

    if len(solved_idx) == 0:
        return np.inf

    return float(budgets[int(solved_idx[0])])


def build_solve_budgets(records, tau: float, profile_type: str):
    """
    Build solve budgets.

    Returns
    -------
    solve_budgets : dict
        solve_budgets[(noise, ns_rule, aggregator)][problem] = list of run budgets
    """
    fmins = compute_problem_fmin(records, profile_type)

    solve_budgets = defaultdict(lambda: defaultdict(list))

    for rec in records:
        noise = rec["noise_model"]
        ns_rule = rec["ns_rule"]
        agg = rec["aggregator"]
        problem = rec["problem_name"]

        if problem not in fmins:
            continue

        key = (noise, ns_rule, agg)

        for run in rec["runs"]:
            b = solve_budget_for_run(run, fmins[problem], tau, profile_type)
            solve_budgets[key][problem].append(b)

    return solve_budgets


def get_available_groups(records):
    """
    Return discovered noise models, ns rules, aggregators.
    """
    noise_models = sorted(set(r["noise_model"] for r in records))
    ns_rules = sorted(set(r["ns_rule"] for r in records))
    aggregators = sorted(set(r["aggregator"] for r in records))

    # Preferred aggregator order.
    preferred = ["mean", "median", "trimmed_mean_10", "mom_K5"]
    aggregators = [a for a in preferred if a in aggregators] + [a for a in aggregators if a not in preferred]

    return noise_models, ns_rules, aggregators


def filter_selected(discovered, selected):
    if selected is None:
        return discovered
    return [x for x in selected if x in discovered]


# ============================================================
# Data profile
# ============================================================

def build_data_profile_curves(solve_budgets, noise, ns_rule, aggregators, xvals):
    """
    Build data profile y-values for each aggregator.
    """
    curves = {}

    # Use the union of problems across aggregators.
    problems = sorted(set(
        problem
        for agg in aggregators
        for problem in solve_budgets.get((noise, ns_rule, agg), {}).keys()
    ))

    if not problems:
        return curves

    for agg in aggregators:
        key = (noise, ns_rule, agg)
        problem_to_budgets = solve_budgets.get(key, {})

        yvals = []

        for x in xvals:
            problem_scores = []

            for problem in problems:
                budgets = problem_to_budgets.get(problem, [])

                if not budgets:
                    problem_scores.append(0.0)
                    continue

                budgets = np.asarray(budgets, dtype=float)

                if AVERAGE_OVER_RUNS_FOR_DATA_PROFILE:
                    # For this problem, average the success indicator over repeated runs.
                    score = float(np.mean(budgets <= x))
                else:
                    # Alternative: count as solved if the median run budget is <= x.
                    score = float(np.median(budgets) <= x)

                problem_scores.append(score)

            yvals.append(float(np.mean(problem_scores)))

        curves[agg] = np.asarray(yvals, dtype=float)

    return curves


# ============================================================
# Performance profile
# ============================================================

def build_perf_profile_curves(solve_budgets, noise, ns_rule, aggregators):
    """
    Build performance profile ratios and curves.

    Uses median solve budget over repeated runs for each method/problem.
    """
    problems = sorted(set(
        problem
        for agg in aggregators
        for problem in solve_budgets.get((noise, ns_rule, agg), {}).keys()
    ))

    if not problems:
        return None, {}

    ratios_by_agg = {agg: [] for agg in aggregators}

    for problem in problems:
        costs = {}

        for agg in aggregators:
            key = (noise, ns_rule, agg)
            budgets = solve_budgets.get(key, {}).get(problem, [])

            if not budgets:
                costs[agg] = np.inf
                continue

            budgets = np.asarray(budgets, dtype=float)
            costs[agg] = float(np.median(budgets))

        finite_costs = [v for v in costs.values() if np.isfinite(v)]

        if not finite_costs:
            # No method solved this problem.
            for agg in aggregators:
                ratios_by_agg[agg].append(np.inf)
            continue

        best = float(np.min(finite_costs))

        # Avoid division by zero if solved at zero budget.
        if best <= 0:
            best = 1.0e-12

        for agg in aggregators:
            c = costs[agg]
            if np.isfinite(c):
                ratios_by_agg[agg].append(max(1.0, c / best))
            else:
                ratios_by_agg[agg].append(np.inf)

    finite_ratios = []
    for ratios in ratios_by_agg.values():
        finite_ratios.extend([r for r in ratios if np.isfinite(r)])

    if not finite_ratios:
        return None, {}

    max_ratio = max(2.0, float(np.max(finite_ratios)))

    # Keep the x-axis readable.
    max_ratio = min(max_ratio, 64.0)

    xvals = np.unique(np.concatenate([
        np.array([1.0]),
        np.logspace(0.0, np.log2(max_ratio), 200, base=2.0),
    ]))

    curves = {}

    for agg, ratios in ratios_by_agg.items():
        ratios = np.asarray(ratios, dtype=float)
        yvals = []

        for x in xvals:
            yvals.append(float(np.mean(ratios <= x)))

        curves[agg] = np.asarray(yvals, dtype=float)

    return xvals, curves


# ============================================================
# Plotting
# ============================================================

def linestyle_for_agg(agg: str):
    style_map = {
        "mean": "-",
        "median": "--",
        "trimmed_mean_10": "-.",
        "mom_K5": ":",
    }
    return style_map.get(agg, "-")


def plot_curves(xvals, curves, aggregators, filename, title, xlabel, ylabel, logx=False):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    plt.figure()
    plt.clf()

    plt.rc("text", usetex=USE_TEX)
    plt.rc("font", family="serif")

    ax = plt.gca()
    plot_fun = ax.semilogx if logx else ax.plot

    for idx, agg in enumerate(aggregators):
        if agg not in curves:
            continue

        plot_fun(
            xvals,
            curves[agg],
            linestyle=linestyle_for_agg(agg),
            linewidth=2.0,
            label=pretty_agg_label(agg),
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="best")

    if logx:
        ax.minorticks_off()
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    converted_root = find_converted_root()
    records = load_converted_results(converted_root)

    discovered_noises, discovered_ns_rules, discovered_aggs = get_available_groups(records)

    noise_models = filter_selected(discovered_noises, SELECT_NOISE_MODELS)
    ns_rules = filter_selected(discovered_ns_rules, SELECT_NS_RULES)
    aggregators = filter_selected(discovered_aggs, SELECT_AGGREGATORS)

    print("============================================================")
    print("CUTEst plotting script")
    print(f"Project root:     {PROJECT_ROOT}")
    print(f"Converted root:   {converted_root}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Loaded records:   {len(records)}")
    print(f"Noise models:     {noise_models}")
    print(f"Sample rules:     {ns_rules}")
    print(f"Aggregators:      {aggregators}")
    print(f"Taus:             {TAUS}")
    print("============================================================")

    # Determine max budget if not specified.
    if MAX_BUDGET_IN_GRADIENTS is None:
        max_budget = 0.0
        for rec in records:
            for run in rec["runs"]:
                axis = get_budget_axis_from_run(run)
                if len(axis) > 0:
                    finite_axis = axis[np.isfinite(axis)]
                    if len(finite_axis) > 0:
                        max_budget = max(max_budget, float(np.max(finite_axis)))

        if max_budget <= 0:
            max_budget = 1.0
    else:
        max_budget = float(MAX_BUDGET_IN_GRADIENTS)

    xvals_data = np.linspace(0.0, max_budget, 201)

    print(f"Data profile max budget: {max_budget}")

    for profile_type in PROFILE_TYPES:
        print(f"[INFO] Building profiles using {profile_type} values.")

        for tau_key, tau in TAUS.items():
            print(f"[INFO] tau={tau_key} ({tau})")

            solve_budgets = build_solve_budgets(records, tau=tau, profile_type=profile_type)

            for noise in noise_models:
                for ns_rule in ns_rules:
                    subdir = os.path.join(OUTPUT_DIR, noise, ns_rule)
                    os.makedirs(subdir, exist_ok=True)

                    # Data profile
                    data_curves = build_data_profile_curves(
                        solve_budgets=solve_budgets,
                        noise=noise,
                        ns_rule=ns_rule,
                        aggregators=aggregators,
                        xvals=xvals_data,
                    )

                    if data_curves:
                        data_file = os.path.join(
                            subdir,
                            f"{noise}_{ns_rule}_{tau_key}_data_profile_{profile_type}.pdf",
                        )

                        title = f"{pretty_noise_label(noise)}, {ns_rule}, {tau_key}, {profile_type}"
                        plot_curves(
                            xvals=xvals_data,
                            curves=data_curves,
                            aggregators=aggregators,
                            filename=data_file,
                            title=title,
                            xlabel="Budget in evaluations / (n+1)",
                            ylabel="Proportion of problems solved",
                            logx=False,
                        )

                        print(f"[PLOT] {data_file}")

                    # Performance profile
                    xvals_perf, perf_curves = build_perf_profile_curves(
                        solve_budgets=solve_budgets,
                        noise=noise,
                        ns_rule=ns_rule,
                        aggregators=aggregators,
                    )

                    if xvals_perf is None or not perf_curves:
                        print(
                            f"[SKIP PERF] noise={noise}, ns={ns_rule}, tau={tau_key}, "
                            f"profile={profile_type}: no finite solved budgets",
                            flush=True,
                        )

                    if xvals_perf is not None and perf_curves:
                        perf_file = os.path.join(
                            subdir,
                            f"{noise}_{ns_rule}_{tau_key}_perf_profile_{profile_type}.pdf",
                        )

                        title = f"{pretty_noise_label(noise)}, {ns_rule}, {tau_key}, {profile_type}"
                        plot_curves(
                            xvals=xvals_perf,
                            curves=perf_curves,
                            aggregators=aggregators,
                            filename=perf_file,
                            title=title,
                            xlabel="Budget / best budget",
                            ylabel="Proportion of problems solved",
                            logx=True,
                        )

                        print(f"[PLOT] {perf_file}")

    print("============================================================")
    print("All plots completed.")
    print(f"Output directory: {OUTPUT_DIR}")
    print("============================================================")


if __name__ == "__main__":
    main()