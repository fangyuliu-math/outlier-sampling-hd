"""
Run a grid of experiments on CUTEst problems:
(noise model) x (aggregator mean/median/trimmed/MoM) x (nsamples rule) x (problems).

This is the CUTEst/PyCUTEst version of run_noise_grid.py.

Main changes from the More-Wild version:
1. Use pycutest.import_problem(...) instead of more_wild.get_problem_as_scalar_objective(...).
2. PROBLEMS is now a list of (problem_name, sif_params) pairs.
3. Output folders are named by CUTEst problem names, e.g. ARWHEAD/gaussian1/mean/const10/.
4. Bounds are read from PyCUTEst when available and passed to Py-BOBYQA.
5. Result summaries store problem_name and sif_params instead of probnum.

Important:
- PyCUTEst usually works on Linux/macOS or Windows via WSL.
- Test with one smooth problem first before running the full grid.
"""

import os
import sys
import json
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

# Put PyCUTEst compiled cache outside OneDrive/project folder.
# This avoids slow syncing and repeated compilation.
# Fix project root and PyCUTEst cache path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.basename(SCRIPT_DIR) == "examples":
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
else:
    PROJECT_ROOT = SCRIPT_DIR

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Use one stable PyCUTEst cache location outside OneDrive/project sync.
PYCUTEST_CACHE = os.path.expanduser("~/pycutest_cache")
os.environ["PYCUTEST_CACHE"] = PYCUTEST_CACHE
os.makedirs(PYCUTEST_CACHE, exist_ok=True)

# Important: make the cache importable.
if PYCUTEST_CACHE not in sys.path:
    sys.path.insert(0, PYCUTEST_CACHE)

# os.environ.setdefault("PYCUTEST_CACHE", os.path.expanduser("~/pycutest_cache"))
# os.makedirs(os.environ["PYCUTEST_CACHE"], exist_ok=True)
#
# if os.environ["PYCUTEST_CACHE"] not in sys.path:
#     sys.path.insert(0, os.environ["PYCUTEST_CACHE"])

import pybobyqa_sampling
import pycutest

# print("PYCUTEST_CACHE =", os.environ["PYCUTEST_CACHE"])
# print("sys.path contains cache =", PYCUTEST_CACHE in sys.path)
#
# p = pycutest.import_problem("BDEXP", sifParams={"N": 100}, quiet=False)
# print("name:", p.name)
# print("n:", p.n)
# print("f0:", p.obj(p.x0))
#
# print("cached:", pycutest.all_cached_problems())


from objfun_wrapper_ext import ObjfunWrapper


# ============================================================
# CUTEst problem loader
# ============================================================

def _clean_cutest_bounds(prob):
    """
    Return bounds in the format expected by Py-BOBYQA: (lower, upper),
    or None if the problem is effectively unconstrained.

    PyCUTEst/CUTEst may use very large finite numbers to represent no bounds.
    This function treats +/- 1e19 as infinity.
    """
    if not (hasattr(prob, "bl") and hasattr(prob, "bu")):
        return None

    bl = np.asarray(prob.bl, dtype=float)
    bu = np.asarray(prob.bu, dtype=float)

    no_lower = np.isneginf(bl) | (bl <= -1.0e19)
    no_upper = np.isposinf(bu) | (bu >= 1.0e19)

    if np.all(no_lower) and np.all(no_upper):
        return None

    return bl, bu


def load_cutest_problem(problem_name: str, sif_params=None, noise_model: str = "smooth") -> ObjfunWrapper:
    """
    Wrap a CUTEst general objective problem with ObjfunWrapper.

    Parameters
    ----------
    problem_name : str
        CUTEst problem name, e.g. "ARWHEAD".
    sif_params : dict or None
        SIF parameters, e.g. {"N": 100}.
    noise_model : str
        Noise model used by ObjfunWrapper, e.g. "smooth", "gaussian1", "studentt_df3".

    Returns
    -------
    ObjfunWrapper
        A wrapped objective that returns noisy function values to the solver,
        while storing both smooth and noisy values.
    """
    if sif_params is None:
        sif_params = {}

    # Import CUTEst problem.
    # If a problem has already been compiled/imported, PyCUTEst should reuse it.
    prob = pycutest.import_problem(problem_name, sifParams=sif_params, quiet=True)

    x0 = np.asarray(prob.x0, dtype=float)
    n = int(prob.n)

    def objfun(x):
        x = np.asarray(x, dtype=float)
        return float(prob.obj(x))

    bounds = _clean_cutest_bounds(prob)

    return ObjfunWrapper(
        name=f"cutest_{problem_name}",
        objfun=objfun,
        n=n,
        x0=x0,
        bounds=bounds,
        noise_model=noise_model,
    )


# ============================================================
# Sampling rules
# ============================================================

def ns_rule_const(k: int):
    """Fixed N = k; accept any call signature used by the solver."""
    def rule(*args, **kwargs):
        return int(k)

    rule._name = f"const{k}"
    return rule


def ns_rule_inv_delta():
    """
    N = max(1, floor(1/delta)).

    The solver may pass delta as a positional argument or keyword argument.
    """
    def rule(*args, **kwargs):
        if "delta" in kwargs:
            delta = kwargs["delta"]
        elif len(args) >= 1:
            delta = args[0]
        else:
            delta = 1.0

        try:
            d = float(delta)
        except Exception:
            d = 1.0

        if not np.isfinite(d) or d <= 0:
            return 1

        return max(1, int(1.0 / d))

    rule._name = "inv_delta"
    return rule


# ============================================================
# Aggregators
# ============================================================

def trimmed_mean(values, alpha=0.1):
    """
    Trimmed mean estimator.

    Parameters
    ----------
    values : array-like
        Samples at the same evaluation point.
    alpha : float
        Trimming proportion on each side. Should satisfy 0 <= alpha < 0.5.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)

    if n == 0:
        return np.nan

    r = int(np.floor(alpha * n))

    # If sample size is too small for trimming, fall back to the ordinary mean.
    if 2 * r >= n:
        return float(np.mean(x))

    x_sorted = np.sort(x)
    return float(np.mean(x_sorted[r:n - r]))


def median_of_means(values, K=5):
    """
    Median-of-means estimator.

    Parameters
    ----------
    values : array-like
        Samples at the same evaluation point.
    K : int
        Number of blocks.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)

    if n == 0:
        return np.nan

    if K <= 0:
        raise ValueError("K must be positive")

    m = n // K

    # If too few samples for K blocks, fall back to median.
    if m == 0:
        return float(np.median(x))

    try:
        block_means = []
        for k in range(K):
            start = k * m
            end = (k + 1) * m if k < K - 1 else n
            block_means.append(np.mean(x[start:end]))

        return float(np.median(block_means))

    except Exception:
        print("MoM calculation failed for samples x = ", x)
        print("Using median instead for this evaluation")
        return float(np.median(x))


# ============================================================
# Saving and plotting helpers
# ============================================================

def as_serializable(results_dict):
    """Convert numpy arrays to python lists for JSON persistence."""
    out = {}
    for k, v in results_dict.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def safe_name(s: str) -> str:
    """
    Make a string safe for folder/file names.
    CUTEst names are usually already safe, but this helps with generality.
    """
    return str(s).replace("/", "_").replace("\\", "_").replace(" ", "_")


def plot_one_run(results, save_pdf, xaxis_in_gradients=True, title=None):
    """
    Save a quick PDF showing smooth and noisy objective values for one run.
    """
    plt.figure()

    n = int(results["n"])
    nf = int(results["nf"])
    xvals = np.arange(nf) / (n + 1) if xaxis_in_gradients else np.arange(nf)

    smooth_vals = np.asarray(results["fvals_smooth"], dtype=float)
    noisy_vals = np.asarray(results["fvals_noisy"], dtype=float)

    plt.semilogy(xvals, smooth_vals, "-", label="True value")
    plt.semilogy(xvals, noisy_vals, "--", label="Noisy value")

    if title:
        plt.title(title)

    plt.xlabel("Budget in gradients" if xaxis_in_gradients else "Objective evaluations")
    plt.ylabel("Objective value f(x)")
    plt.legend(loc="best")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_pdf, bbox_inches="tight")
    plt.close()


# ============================================================
# Grid settings
# ============================================================

# ------------------------------------------------------------
# PILOT PROBLEMS
# ------------------------------------------------------------
# Start with one or a few easy CUTEst problems.
# Do NOT run the full Appendix D list before this pilot works.
# PROBLEMS = [
#     # ("ARWHEAD", {"N": 100}),
#     ("BOX", {"N": 100}),
#     ("COSINE", {"N": 100}),
#     ("NONDQUAR", {"N": 100}),
#     ("POWER", {"N": 100}),
# ]


# ------------------------------------------------------------
# A safer larger subset after the pilot works
# ------------------------------------------------------------
# These are mostly Appendix D problems with direct N=100 parameter.
# Uncomment after the pilot works.
#
# PROBLEMS = [
#     ("ARWHEAD", {"N": 100}),
#     ("BOX", {"N": 100}),
#     ("BOXPOWER", {"N": 100}),
#     ("COSINE", {"N": 100}),
#     ("CURLY10", {"N": 100}),
#     ("CURLY20", {"N": 100}),
#     ("ENGVAL1", {"N": 100}),
#     ("NCB20", {"N": 100}),
#     ("NCB20B", {"N": 100}),
#     ("NONCVXU2", {"N": 100}),
#     ("NONCVXUN", {"N": 100}),
#     ("NONDQUAR", {"N": 100}),
#     ("POWER", {"N": 100}),
#     ("SCHMVETT", {"N": 100}),
#     ("SINQUAD", {"N": 100}),
#     ("TOINTGSS", {"N": 100}),
# ]


# ------------------------------------------------------------
# Full Appendix D list from Cartis--Fiala--Marteau--Roberts
# ------------------------------------------------------------
# Some parameter names must be checked with:
#     pycutest.print_available_sif_params("PROBLEMNAME")
#
# Problems marked with bounds in the paper, such as BDEXP and SINEALI,
# are included here but should be tested carefully.
#
# PROBLEMS = [
#     ("ARWHEAD", {"N": 100}),
#     ("BDEXP", {"N": 100}),
#     ("BOX", {"N": 100}),
#     ("BOXPOWER", {"N": 100}),
#     ("BROYDN7D", {"N": 50}),       # paper says N/2 = 50; confirm with print_available_sif_params
#     ("CHARDIS1", {"NP1": 50}),
#     ("COSINE", {"N": 100}),
#     ("CURLY10", {"N": 100}),
#     ("CURLY20", {"N": 100}),
#     ("DIXMAANA", {"M": 30}),
#     ("DIXMAANF", {"M": 30}),
#     ("DIXMAANP", {"M": 30}),
#     ("ENGVAL1", {"N": 100}),
#     ("FMINSRF2", {"P": 8}),
#     ("FMINSURF", {"P": 8}),
#     ("NCB20", {"N": 100}),
#     ("NCB20B", {"N": 100}),
#     ("NONCVXU2", {"N": 100}),
#     ("NONCVXUN", {"N": 100}),
#     ("NONDQUAR", {"N": 100}),
#     ("ODC", {"NX": 10, "NY": 10}),
#     ("PENALTY3", {"N": 50}),      # paper says N/2 = 50; confirm with print_available_sif_params
#     ("POWER", {"N": 100}),
#     ("RAYBENDL", {"NKNOTS": 32}),
#     ("SCHMVETT", {"N": 100}),
#     ("SINEALI", {"N": 100}),
#     ("SINQUAD", {"N": 100}),
#     ("TOINTGOR", {}),
#     ("TOINTGSS", {"N": 100}),
#     ("TOINTPSP", {}),
# ]
PROBLEMS = [
    ("ARWHEAD", {"N": 100}),
    ("BDEXP", {"N": 100}),
    ("BOX", {"N": 100}),
    ("BOXPOWER", {"N": 100}),
    ("BROYDN7D", {"N/2": 50}),
    ("CHARDIS1", {"NP1": 50}),  # has nonlinear inequality constraints; skip for Py-BOBYQA
    ("COSINE", {"N": 100}),
    ("CURLY10", {"N": 100}),
    ("CURLY20", {"N": 100}),
    ("DIXMAANA1", {"M": 30}),    # your MASTSIF says DIXMAANA.SIF does not exist
    ("DIXMAANF", {"M": 30}),
    ("DIXMAANP", {"M": 30}),
    ("ENGVAL1", {"N": 100}),
    ("FMINSRF2", {"P": 8}),
    ("FMINSURF", {"P": 8}),
    ("NCB20", {"N": 100}),
    ("NCB20B", {"N": 100}),
    ("NONCVXU2", {"N": 100}),
    ("NONCVXUN", {"N": 100}),
    ("NONDQUAR", {"N": 100}),
    # ("ODC", {"NX": 10, "NY": 10}),
    ("PENALTY3", {"N/2": 50}),
    ("POWER", {"N": 100}),
    ("RAYBENDL", {"NKNOTS": 32}),
    ("SCHMVETT", {"N": 100}),
    ("SINEALI", {"N": 100}),
    ("SINQUAD", {"N": 100}),
    ("TOINTGOR", {}),
    ("TOINTGSS", {"N": 100}),
    ("TOINTPSP", {}),
]

# ------------------------------------------------------------
# Noise models
# ------------------------------------------------------------
# For the first test, use only smooth.
# After smooth works, switch to noisy models.
# NOISE_MODELS = [
#     "smooth",
# ]

# After the pilot works, try:
#
NOISE_MODELS = [
    "gaussian1",
    "studentt_df1",
    "studentt_df3",
    "failure_low_p08",
]


# ------------------------------------------------------------
# Aggregators
# ------------------------------------------------------------
# For the first test, use only mean.
# AGGREGATORS = [
#     ("mean", np.mean),
# ]

# After the pilot works, try:
#
# AGGREGATORS = [
#     ("mean", np.mean),
#     ("median", np.median),
# ]
#
# Later, for the full robust comparison:
#

# import logging
# logging.basicConfig(level=logging.INFO, format='%(message)s')

# ... (call pybobyqa.solve)
AGGREGATORS = [
    ("mean", np.mean),
    ("median", np.median),
    ("trimmed_mean_10", lambda x: trimmed_mean(x, alpha=0.1)),
    ("mom_K5", lambda x: median_of_means(x, K=5)),
]


# ------------------------------------------------------------
# Sampling rules
# ------------------------------------------------------------
# For the first test, use only N=1.
# NS_RULES = [
#      ns_rule_inv_delta()
# ]

# NS_RULES = [
#     ns_rule_const(30)
# ]

# After the pilot works, try:
#
# NS_RULES = [
#     ns_rule_const(10),
# ]
#
# Later, for a fuller comparison:
#
NS_RULES = [
    # ns_rule_const(10),
    ns_rule_const(30),
    # ns_rule_const(50),
    ns_rule_const(100),
    ns_rule_inv_delta(),
]


# ------------------------------------------------------------
# Budget and output
# ------------------------------------------------------------
# IMPORTANT:
# For n ≈ 100, maxfun = BUDGET_IN_GRAD * (n+1).
# BUDGET_IN_GRAD=20 already means about 2020 evaluations per run.
BUDGET_IN_GRAD = 200

# For the pilot, use 1 run only.
NRUNS_PER_COMBO = 10

# Use a separate folder so this does not mix with your More-Wild results.
OUT_ROOT = os.path.join("raw_results_cutest_pilot_parallel", "noise_grid")

MAKE_PDF = True


# ============================================================
# Main experiment routine
# ============================================================

def run_one_combo(problem_name, sif_params, noise_model, agg_name, agg_fn, ns_rule, budget, nruns, out_root):
    """
    Run one combination:
    problem x noise model x aggregator x sample rule.
    """
    problem_name_safe = safe_name(problem_name)
    outdir = os.path.join(out_root, problem_name_safe, noise_model, agg_name, ns_rule._name)
    os.makedirs(outdir, exist_ok=True)

    for r in range(nruns):
        # Fresh wrapper/run each time.
        wrapper = load_cutest_problem(
            problem_name,
            sif_params=sif_params,
            noise_model=noise_model,
        )

        maxfun = int(budget * (wrapper._n + 1))

        # Use bounds from CUTEst if available.
        pybobyqa_sampling.solve(
            wrapper,
            wrapper._x0,
            bounds=wrapper.bounds_as_lower_upper_arrays(),
            maxfun=maxfun,
            npt=2*wrapper._n + 1,
            nsamples=ns_rule,
            nsamples_aggregator=agg_fn,
            objfun_has_noise=wrapper.is_noisy(),
            print_progress=False,
        )

        res = wrapper.get_results(vectors_as_numpy=True)

        res_summary = {
            "f_best_smooth": float(np.min(res["fvals_smooth"])),
            "f_best_noisy": float(np.min(res["fvals_noisy"])),
            "nf": int(res["nf"]),
            "n": int(res["n"]),
            "maxfun": int(maxfun),
            "budget_in_gradients": int(budget),
            "noise_model": noise_model,
            "aggregator": agg_name,
            "ns_rule": ns_rule._name,
            "problem_name": problem_name,
            "problem_name_safe": problem_name_safe,
            "sif_params": sif_params,
            "objfun_name": res["objfun_name"],
            "run": int(r),
        }

        # Save JSON.
        json_path = os.path.join(outdir, f"run{r:02d}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "results": as_serializable(res),
                    "summary": res_summary,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        # Save quick PDF curve for the first run of each combo.
        if MAKE_PDF and r == 0:
            pdf_path = os.path.join(outdir, f"run{r:02d}.pdf")
            plot_one_run(
                res,
                pdf_path,
                title=f"{problem_name} | {noise_model} | {agg_name} | {ns_rule._name}",
            )

        print(
            f"[done] {problem_name} "
            f"{noise_model} {agg_name} {ns_rule._name} "
            f"r={r:02d} nf={int(res['nf'])} n={int(res['n'])}"
        )


# def main():
#     os.makedirs(OUT_ROOT, exist_ok=True)
#
#     print("============================================================")
#     print("CUTEst noise grid experiment")
#     print(f"Number of problems: {len(PROBLEMS)}")
#     print(f"Noise models: {[x for x in NOISE_MODELS]}")
#     print(f"Aggregators: {[x[0] for x in AGGREGATORS]}")
#     print(f"Sample rules: {[x._name for x in NS_RULES]}")
#     print(f"Budget in gradients: {BUDGET_IN_GRAD}")
#     print(f"Runs per combo: {NRUNS_PER_COMBO}")
#     print(f"Output root: {OUT_ROOT}")
#     print("============================================================")
#
#     for problem_name, sif_params in PROBLEMS:
#         for nm in NOISE_MODELS:
#             for agg_name, agg_fn in AGGREGATORS:
#                 for ns in NS_RULES:
#                     run_one_combo(
#                         problem_name,
#                         sif_params,
#                         nm,
#                         agg_name,
#                         agg_fn,
#                         ns,
#                         BUDGET_IN_GRAD,
#                         NRUNS_PER_COMBO,
#                         OUT_ROOT,
#                     )
#
#     print("All done.")
#
#
#
# if __name__ == "__main__":
#     main()

# ============================================================
# Run all settings for one problem
# ============================================================

def main(problem):
    """
    Worker function used by multiprocessing.Pool.

    Each worker receives one CUTEst problem:
        (problem_name, sif_params)

    It then runs all noise models, aggregators, sampling rules,
    and repeated runs for that problem.
    """
    problem_name, sif_params = problem

    print(
        f"[START PROBLEM] {problem_name}, params={sif_params}",
        flush=True,
    )

    try:
        for nm in NOISE_MODELS:
            for agg_name, agg_fn in AGGREGATORS:
                for ns in NS_RULES:
                    run_one_combo(
                        problem_name=problem_name,
                        sif_params=sif_params,
                        noise_model=nm,
                        agg_name=agg_name,
                        agg_fn=agg_fn,
                        ns_rule=ns,
                        budget=BUDGET_IN_GRAD,
                        nruns=NRUNS_PER_COMBO,
                        out_root=OUT_ROOT,
                    )

        print(
            f"[FINISHED PROBLEM] {problem_name}",
            flush=True,
        )

        return {
            "problem_name": problem_name,
            "success": True,
            "error": None,
        }

    except Exception as exc:
        print(
            f"[FAILED PROBLEM] {problem_name}: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        return {
            "problem_name": problem_name,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


# ============================================================
# Parallel driver
# ============================================================

if __name__ == "__main__":
    os.makedirs(OUT_ROOT, exist_ok=True)

    nthreads = 16

    print("============================================================")
    print("Serial PyCUTEst cache preparation")
    print(f"PYCUTEST_CACHE: {PYCUTEST_CACHE}")
    print("============================================================")

    valid_problems = []
    cache_failures = []

    for problem_name, sif_params in PROBLEMS:
        try:
            prob = pycutest.import_problem(
                problem_name,
                sifParams=sif_params,
                quiet=True,
            )

            print(
                f"[CACHE OK] {problem_name}, "
                f"params={sif_params}, n={prob.n}",
                flush=True,
            )

            valid_problems.append((problem_name, sif_params))

        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"

            print(
                f"[CACHE FAILED] {problem_name}, "
                f"params={sif_params}: {error_message}",
                flush=True,
            )

            cache_failures.append({
                "problem_name": problem_name,
                "sif_params": sif_params,
                "error": error_message,
            })

    if cache_failures:
        cache_failure_file = os.path.join(
            OUT_ROOT,
            "cache_failures.json",
        )

        with open(cache_failure_file, "w", encoding="utf-8") as f:
            json.dump(
                cache_failures,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print("Cache failures saved to:", cache_failure_file)

    print("============================================================")
    print("Parallel CUTEst noise-grid experiment")
    print(f"Requested problems: {len(PROBLEMS)}")
    print(f"Valid problems: {len(valid_problems)}")
    print(f"Worker processes: {nthreads}")
    print(f"Noise models: {NOISE_MODELS}")
    print(f"Aggregators: {[name for name, _ in AGGREGATORS]}")
    print(f"Sample rules: {[rule._name for rule in NS_RULES]}")
    print(f"Budget in gradients: {BUDGET_IN_GRAD}")
    print(f"Runs per combination: {NRUNS_PER_COMBO}")
    print(f"Output root: {OUT_ROOT}")
    print("============================================================")

    with Pool(processes=nthreads) as p:
        completed_results = p.map(
            main,
            valid_problems,
            chunksize=1,
        )

    successful = [
        result for result in completed_results
        if result["success"]
    ]

    failed = [
        result for result in completed_results
        if not result["success"]
    ]

    print("============================================================")
    print("Parallel experiment finished.")
    print(f"Successful problems: {len(successful)}")
    print(f"Failed problems: {len(failed)}")

    if failed:
        failed_file = os.path.join(
            OUT_ROOT,
            "parallel_problem_failures.json",
        )

        with open(failed_file, "w", encoding="utf-8") as f:
            json.dump(
                failed,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print("Failure information saved to:", failed_file)

    print("============================================================")