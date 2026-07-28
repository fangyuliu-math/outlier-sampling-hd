"""
Generic wrapper for objective functions, which saves all the evaluations (with and without noise)

To add a new noise model:
- Add the name to VALID_NOISE_MODELS
- Include a new 'elif' case to the apply_noise function at the end of this file. Inputs to this function are:
  (a) x = the point being evaluated (NumPy array)
  (b) fx = true (no noise) function value (float)
  (c) noise_model = noise model to use (string)

Lindon Roberts, 2020-2025
"""

import numpy as np

__all__ = ['ObjfunWrapper', 'VALID_NOISE_MODELS']


class ObjfunWrapper(object):
    def __init__(self, name, objfun, n, x0, bounds=None, noise_model='smooth'):
        self._name = name
        self._objfun = objfun
        self._n = int(n)
        self._x0 = x0
        self._f0 = float(objfun(x0))
        if bounds is not None:
            self._bl = bounds[0]
            self._bu = bounds[1]
        else:
            self._bl = None
            self._bu = None
        assert noise_model in VALID_NOISE_MODELS, "Unknown noise model: %s" % noise_model
        self._noise_model = noise_model

        # Data stored for each run
        self._fvals_smooth = []
        self._fvals_noisy = []
        self._eval_time = []
        self._eval_wall_time = []
        self._nf = 0

        self.run_id = None
        self.solver_info = None
        self.budget = None
        self.maxfev = None
        self.outfolder = None

    def is_noisy(self):
        return self._noise_model != 'smooth'

    def bounds_as_lower_upper_arrays(self):
        return self._bl, self._bu

    def bounds_on_each_variable(self, pairs_as_list=True):
        if self._bl is None or self._bu is None:
            return None
        # List [(l1,u1), ..., (ln,un)] of length n
        # Each entry is either tuple or list
        bounds = []
        for i in range(self._n):
            bounds.append([self._bl[i], self._bu[i]] if pairs_as_list else (self._bl[i], self._bu[i]))
        return bounds

    def __call__(self, x):
        # Evaluate objective, but save everything too
        f_smooth = self._objfun(x)
        f_noisy = apply_noise(x, f_smooth, noise_model=self._noise_model)
        self._fvals_smooth.append(f_smooth)
        self._fvals_noisy.append(f_noisy)
        self._nf += 1
        return float(f_noisy)

    def get_results(self, vectors_as_numpy=True):
        # Build a dictionary of return values, based on Oliver's benchmark.py
        results = {}
        results['objfun_name'] = self._name
        results['nf'] = self._nf
        results['noise_model'] = self._noise_model
        results['n'] = self._n
        results['f0_smooth'] = self._f0
        # Iterate info
        if vectors_as_numpy:
            results['fvals_noisy'] = np.array(self._fvals_noisy)
            results['fvals_smooth'] = np.array(self._fvals_smooth)
        else:
            results['fvals_noisy'] = self._fvals_noisy  # list of length nf, each entry is float
            results['fvals_smooth'] = self._fvals_smooth  # list of length nf, each entry is float
        return results

    def clear(self):
        self._fvals_smooth = []
        self._fvals_noisy = []
        self._nf = 0
        return


# Update this list when adding new noise types
VALID_NOISE_MODELS = [
    'smooth',
    'gaussian1',
    # New:
    'uniform', 'uniform1', 'uniform2',          # 幅度不同的均匀噪声
    'studentt_df1', 'studentt_df2', 'studentt_df3', 'studentt_df5',  # 不同自由度的 Student-t
    'failure_p06', 'failure_p07', 'failure_p08',  # 成功率 p 返回真值
'failure_low_p06', 'failure_low_p07', 'failure_low_p08', 'failure_low_p09',
    'failure_high_p06','failure_high_p07','failure_high_p08',
'failure_uniform1e4_p07','failure_uniform1e4_p08','failure_uniform1e4_p09',
]

# 统一参数
FAILURE_LOW_VALUE  = -1.0e4   # or -1.0e6
FAILURE_HIGH_VALUE =  1.0e4

# TODO: add noise models that you want (e.g. uniform, student t)

def apply_noise(x, fx, noise_model='smooth'):
    assert noise_model in VALID_NOISE_MODELS, "Unknown noise model: %s" % noise_model
    if noise_model == 'smooth':
        # f_noisy(x) = f(x), i.e. no noise
        return fx
    elif noise_model == 'gaussian1':
        # f_noisy(x) = f(x) + N(0,sigma^2) for sigma=1e-2
        return fx + np.random.normal(0.0, 0.01)

    # ========== New noise models ==========
    elif noise_model == 'uniform':
        # f(x) + U[-1e-0, 1e-0]
        return fx + np.random.uniform(-1e-0, 1e-0)
    elif noise_model == 'uniform1':
        # f(x) + U[-1e-1, 1e-1]
        return fx + np.random.uniform(-1e-1, 1e-1)
    elif noise_model == 'uniform2':#dont use #After that's done, make new noise models using 0.1 and/or 1 instead
        # 更重一些 more heavy
        return fx + np.random.uniform(-1e-2, 1e-2)

    elif noise_model == 'studentt_df1':#1e-1
        # 重尾：df=3，缩放到 ~1e-2 量级
        return fx + (1e-2 * np.random.standard_t(df=1))#try large scale
    elif noise_model == 'studentt_df2':
        return fx + (1e-2 * np.random.standard_t(df=2))

    elif noise_model == 'studentt_df3':#df=2
        # 重尾：df=3，缩放到 ~1e-2 量级
        return fx + (1e-2 * np.random.standard_t(df=3))#try large scale
    elif noise_model == 'studentt_df5':
        return fx + (1e-2 * np.random.standard_t(df=5))

    elif noise_model == 'failure_low_p06':
        return fx if (np.random.rand() < 0.6) else FAILURE_LOW_VALUE
    elif noise_model == 'failure_low_p07':
        return fx if (np.random.rand() < 0.7) else FAILURE_LOW_VALUE
    elif noise_model == 'failure_low_p08':
        return fx if (np.random.rand() < 0.8) else FAILURE_LOW_VALUE
    elif noise_model == 'failure_low_p09':
        return fx if (np.random.rand() < 0.9) else FAILURE_LOW_VALUE


    elif noise_model == 'failure_high_p06':
        return fx if (np.random.rand() < 0.6) else FAILURE_HIGH_VALUE
    elif noise_model == 'failure_high_p07':
        return fx if (np.random.rand() < 0.7) else FAILURE_HIGH_VALUE
    elif noise_model == 'failure_high_p08':
        return fx if (np.random.rand() < 0.8) else FAILURE_HIGH_VALUE

    elif noise_model == 'failure_uniform1e4_p07':
        return fx if (np.random.rand() < 0.7) else FAILURE_LOW_VALUE
    elif noise_model == 'failure_uniform1e4_p08':
        return fx if (np.random.rand() < 0.8) else FAILURE_LOW_VALUE
    elif noise_model == 'failure_uniform1e4_p09':
        return fx if (np.random.rand() < 0.9) else FAILURE_LOW_VALUE

    elif noise_model == 'failure_p07':
        return fx if (np.random.rand() < 0.7) else np.random.uniform(FAILURE_LOW_VALUE, FAILURE_HIGH_VALUE)
    elif noise_model == 'failure_p08':
        return fx if (np.random.rand() < 0.8) else np.random.uniform(FAILURE_LOW_VALUE, FAILURE_HIGH_VALUE)
    elif noise_model == 'failure_p09':
        return fx if (np.random.rand() < 0.9) else np.random.uniform(FAILURE_LOW_VALUE, FAILURE_HIGH_VALUE)



    # elif noise_model == 'failure_p06':#try failure value very small (e.g. -10,000) or very large
    #     # 以 p 返回真值，否则返回“错误值”（离群）
    #     return fx if (np.random.rand() < 0.6) else fx + np.random.normal(0.0, 1.0)#
    # elif noise_model == 'failure_p07':
    #     return fx if (np.random.rand() < 0.7) else fx + np.random.normal(0.0, 1.0)
    # elif noise_model == 'failure_p08':
    #     return fx if (np.random.rand() < 0.8) else fx + np.random.normal(0.0, 1.0)
    # ======================================

    else:
        raise RuntimeError(f"Unknown noise model: {noise_model}")