import os

os.environ.setdefault("PYCUTEST_CACHE", os.path.expanduser("~/pycutest_cache"))
os.makedirs(os.environ["PYCUTEST_CACHE"], exist_ok=True)

import pycutest


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
    # ("DIXMAANA", {"M": 30}),    # your MASTSIF says DIXMAANA.SIF does not exist
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
    ("ODC", {"NX": 10, "NY": 10}),
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


def main():
    good = []
    bad = []

    for name, params in PROBLEMS:
        print("=" * 70)
        print(f"Testing {name} with params {params}")

        try:
            prob = pycutest.import_problem(name, sifParams=params, quiet=False)
            f0 = prob.obj(prob.x0)
            print(f"[OK] {name}: n={prob.n}, f(x0)={f0}")
            good.append((name, params, prob.n, f0))

        except Exception as e:
            print(f"[BAD] {name}: {type(e).__name__}: {e}")
            bad.append((name, params, repr(e)))

    print("\n\n================ SUMMARY ================")
    print("GOOD problems:")
    for item in good:
        print(item)

    print("\nBAD problems:")
    for item in bad:
        print(item)


if __name__ == "__main__":
    main()