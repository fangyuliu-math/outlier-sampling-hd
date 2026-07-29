"""
Convert nested CUTEst noise-grid JSON files into a plotting-friendly format.

Input structure produced by run_cutest_noise_grid.py:

raw_results_cutest_const10_budget20/noise_grid/
    ARWHEAD/
        gaussian1/
            mean/
                const10/
                    run00.json
                    run01.json
                    run02.json

Output structure:

converted_results_cutest_const10_budget20/
    gaussian1_mean_const10/
        gaussian1_mean_const10_cutest_ARWHEAD.json
        gaussian1_mean_const10_cutest_BDEXP.json
        ...
    studentt_df3_median_const10/
        studentt_df3_median_const10_cutest_ARWHEAD.json
        ...

The output keeps both:
1. compact fields useful for plotting;
2. original results and summaries for later debugging.

This script is designed for CUTEst problem names, not More--Wild prob numbers.
"""

import os
import json
import glob
from collections import defaultdict
import numpy as np


# ============================================================
# Path setup
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.basename(SCRIPT_DIR) == "examples":
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
else:
    PROJECT_ROOT = SCRIPT_DIR


# Candidate input roots.
# The script will use the first one that exists.
CANDIDATE_IN_ROOTS = [
    os.path.join(PROJECT_ROOT, "raw_results_cutest_parallel", "noise_grid"),
    os.path.join(PROJECT_ROOT, "examples", "raw_results_cutest_parallel", "noise_grid"),
]

OUT_ROOT = os.path.join(PROJECT_ROOT, "converted_results_cutest_parallel")

# Default output root.
# If the input root contains "pilot", output goes to converted_results_cutest_pilot.
# Otherwise output goes to converted_results_cutest_const10_budget20.
# OUT_ROOT_FORMAL = os.path.join(PROJECT_ROOT, "converted_results_cutest_const10_budget20")
# OUT_ROOT_PILOT = os.path.join(PROJECT_ROOT, "converted_results_cutest_pilot")


# ============================================================
# Helpers
# ============================================================

# def find_input_root():
#     for path in CANDIDATE_IN_ROOTS:
#         if os.path.isdir(path):
#             return path
#
#     msg = "Could not find any CUTEst raw result directory. Tried:\n"
#     msg += "\n".join(f"  - {p}" for p in CANDIDATE_IN_ROOTS)
#     raise FileNotFoundError(msg)
def find_input_root():
    print("[DEBUG] Candidate input roots:")
    for path in CANDIDATE_IN_ROOTS:
        print(f"  {path} | exists = {os.path.isdir(path)}")

    for path in CANDIDATE_IN_ROOTS:
        if os.path.isdir(path):
            print("[DEBUG] Using input root:", path)
            return path

    msg = "Could not find raw_results_cutest_pilot_100/noise_grid. Tried:\n"
    msg += "\n".join(f"  - {p}" for p in CANDIDATE_IN_ROOTS)
    raise FileNotFoundError(msg)


def safe_name(s: str) -> str:
    return str(s).replace("/", "_").replace("\\", "_").replace(" ", "_")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_run_path(path, in_root):
    """
    Parse one run path.

    Expected relative path:
        PROBLEM/noise/aggregator/ns_rule/runXX.json

    Example:
        ARWHEAD/gaussian1/mean/const10/run00.json
    """
    rel = os.path.relpath(path, in_root)
    parts = rel.split(os.sep)

    if len(parts) != 5:
        raise ValueError(f"Unexpected path format: {path}")

    problem_name, noise_model, aggregator, ns_rule, filename = parts
    return problem_name, noise_model, aggregator, ns_rule, filename


def extract_run_number(filename, fallback=0):
    """
    Extract run number from run00.json, run01.json, etc.
    """
    base = os.path.basename(filename)
    base_no_ext = os.path.splitext(base)[0]

    if base_no_ext.startswith("run"):
        try:
            return int(base_no_ext[3:])
        except Exception:
            return fallback

    return fallback


def to_float_list(x):
    """
    Convert a list-like object to a plain list of floats when possible.
    """
    if x is None:
        return []

    try:
        return [float(v) for v in x]
    except Exception:
        return list(x)


def summarize_one_run(data, path, fallback_problem, fallback_noise, fallback_agg, fallback_ns, fallback_run):
    """
    Extract standard fields from one raw run JSON.

    Raw JSON structure from run_cutest_noise_grid.py:
        {
            "results": {...},
            "summary": {...}
        }
    """
    results = data.get("results", {})
    summary = data.get("summary", {})

    problem_name = summary.get("problem_name", fallback_problem)
    problem_name_safe = summary.get("problem_name_safe", safe_name(problem_name))
    objfun_name = summary.get("objfun_name", results.get("objfun_name", f"cutest_{problem_name}"))

    noise_model = summary.get("noise_model", fallback_noise)
    aggregator = summary.get("aggregator", fallback_agg)
    ns_rule = summary.get("ns_rule", fallback_ns)
    run = int(summary.get("run", fallback_run))

    n = int(summary.get("n", results.get("n", 0)))
    nf = int(summary.get("nf", results.get("nf", 0)))

    fvals_smooth = to_float_list(results.get("fvals_smooth", []))
    fvals_noisy = to_float_list(results.get("fvals_noisy", []))

    # If nf was not saved correctly, infer it from fvals.
    if nf <= 0:
        nf = max(len(fvals_smooth), len(fvals_noisy))

    if n > 0:
        budget_axis = [i / float(n + 1) for i in range(nf)]
    else:
        budget_axis = list(range(nf))

    f_best_smooth = summary.get("f_best_smooth", None)
    if f_best_smooth is None and len(fvals_smooth) > 0:
        f_best_smooth = float(np.nanmin(np.asarray(fvals_smooth, dtype=float)))

    f_best_noisy = summary.get("f_best_noisy", None)
    if f_best_noisy is None and len(fvals_noisy) > 0:
        f_best_noisy = float(np.nanmin(np.asarray(fvals_noisy, dtype=float)))

    compact = {
        "run": run,
        "source_path": path,
        "problem_name": problem_name,
        "problem_name_safe": problem_name_safe,
        "objfun_name": objfun_name,
        "noise_model": noise_model,
        "aggregator": aggregator,
        "ns_rule": ns_rule,
        "n": n,
        "nf": nf,
        "npt": summary.get("npt", None),
        "npt_mode": summary.get("npt_mode", None),
        "maxfun": summary.get("maxfun", None),
        "budget_in_gradients": summary.get("budget_in_gradients", None),
        "f_best_smooth": f_best_smooth,
        "f_best_noisy": f_best_noisy,
        "budget_axis": budget_axis,
        "fvals_smooth": fvals_smooth,
        "fvals_noisy": fvals_noisy,
        "summary": summary,
        "results": results,
    }

    return compact


# ============================================================
# Main conversion
# ============================================================

def main():
    in_root = find_input_root()

    # if "pilot" in in_root:
    #     out_root = OUT_ROOT_PILOT
    # else:
    #     out_root = OUT_ROOT_FORMAL

    # Always output to converted_results_cutest_pilot_100.
    out_root = OUT_ROOT

    print("============================================================")
    print("CUTEst converter")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input root:   {in_root}")
    print(f"Output root:  {out_root}")
    print("============================================================")

    pattern = os.path.join(in_root, "*", "*", "*", "*", "run*.json")
    paths = sorted(glob.glob(pattern))

    if not paths:
        raise FileNotFoundError(f"No run JSON files found with pattern:\n{pattern}")

    print(f"Found raw run files: {len(paths)}")

    # Group by one algorithm/noise setting and one problem.
    # Key:
    #   (run_name, problem_name)
    grouped = defaultdict(list)

    # Store metadata for summary.
    seen_run_names = set()
    seen_problem_names = set()
    errors = []

    for path in paths:
        try:
            problem_from_path, noise_from_path, agg_from_path, ns_from_path, filename = parse_run_path(path, in_root)
            run_from_path = extract_run_number(filename)

            data = read_json(path)

            run_data = summarize_one_run(
                data=data,
                path=path,
                fallback_problem=problem_from_path,
                fallback_noise=noise_from_path,
                fallback_agg=agg_from_path,
                fallback_ns=ns_from_path,
                fallback_run=run_from_path,
            )

            problem_name = run_data["problem_name"]
            noise_model = run_data["noise_model"]
            aggregator = run_data["aggregator"]
            ns_rule = run_data["ns_rule"]

            run_name = f"{noise_model}_{aggregator}_{ns_rule}"

            grouped[(run_name, problem_name)].append(run_data)
            seen_run_names.add(run_name)
            seen_problem_names.add(problem_name)

        except Exception as e:
            errors.append({
                "path": path,
                "error_type": type(e).__name__,
                "error_message": str(e),
            })

    print(f"Grouped problem/run-name pairs: {len(grouped)}")
    print(f"Unique run names: {len(seen_run_names)}")
    print(f"Unique problems: {len(seen_problem_names)}")

    converted_files = []

    for (run_name, problem_name), runs in sorted(grouped.items()):
        # Sort repeated runs by run index.
        runs = sorted(runs, key=lambda d: d["run"])

        objfun_name = runs[0].get("objfun_name", f"cutest_{problem_name}")
        problem_name_safe = runs[0].get("problem_name_safe", safe_name(problem_name))

        n_values = sorted(set(int(r["n"]) for r in runs if r.get("n") is not None))
        n = n_values[0] if n_values else None

        noise_model = runs[0].get("noise_model")
        aggregator = runs[0].get("aggregator")
        ns_rule = runs[0].get("ns_rule")

        # Compatibility aliases:
        # Some plotting code expects all_results/all_summaries style objects.
        all_results = [r["results"] for r in runs]
        all_summaries = [r["summary"] for r in runs]

        out_obj = {
            "format": "cutest_converted_v1",
            "run_name": run_name,
            "problem_name": problem_name,
            "problem_name_safe": problem_name_safe,
            "objfun_name": objfun_name,
            "n": n,
            "noise_model": noise_model,
            "aggregator": aggregator,
            "ns_rule": ns_rule,
            "nruns": len(runs),

            # Main plotting-friendly payload.
            "runs": runs,

            # Compatibility payload for old-style plotting scripts.
            "all_results": all_results,
            "all_summaries": all_summaries,

            # Convenience arrays.
            "f_best_smooth_runs": [r["f_best_smooth"] for r in runs],
            "f_best_noisy_runs": [r["f_best_noisy"] for r in runs],
            "nf_runs": [r["nf"] for r in runs],
        }

        out_dir = os.path.join(out_root, run_name)
        out_file = os.path.join(out_dir, f"{run_name}_cutest_{problem_name_safe}.json")

        write_json(out_file, out_obj)
        converted_files.append(out_file)

    # Write index files for plotting.
    index = {
        "format": "cutest_converted_index_v1",
        "input_root": in_root,
        "output_root": out_root,
        "num_raw_run_files": len(paths),
        "num_converted_files": len(converted_files),
        "run_names": sorted(seen_run_names),
        "problem_names": sorted(seen_problem_names),
        "errors": errors,
    }

    write_json(os.path.join(out_root, "index.json"), index)

    # Also write plain text lists for easy copy-paste.
    os.makedirs(out_root, exist_ok=True)

    with open(os.path.join(out_root, "run_names.txt"), "w", encoding="utf-8") as f:
        for rn in sorted(seen_run_names):
            f.write(rn + "\n")

    with open(os.path.join(out_root, "problem_names.txt"), "w", encoding="utf-8") as f:
        for pn in sorted(seen_problem_names):
            f.write(pn + "\n")

    print("============================================================")
    print("Conversion complete.")
    print(f"Converted files: {len(converted_files)}")
    print(f"Index file: {os.path.join(out_root, 'index.json')}")
    print(f"Run names:  {os.path.join(out_root, 'run_names.txt')}")
    print(f"Problems:   {os.path.join(out_root, 'problem_names.txt')}")

    if errors:
        print(f"Errors while reading raw files: {len(errors)}")
        print(f"See index.json for details.")
    else:
        print("No raw-file reading errors.")

    print("============================================================")


if __name__ == "__main__":
    main()