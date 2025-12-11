# Glib — Regression & Classification Library (NumPy + Matplotlib)
A focused documentation file for **glib.py** -- a compact library implementing several supervised learning classes and dynamic plotting helpers using `numpy` and `matplotlib`.

**Important:** This entire README is provided as a single continuous block so you can copy-paste it into your repository `README.md` in one go.

---

TABLE OF CONTENTS
1. Quick summary
2. Requirements & installation
3. Conventions & shapes
4. Plotting modes
5. Class reference (complete)
   - LinearReg
   - Lasso
   - Ridge
   - PowerReg
   - LogesticReg
6. Examples (many runnable examples)
7. Tutorials & demo mapping
8. Practical tips for training & hyperparameters
9. Contact

---

1) QUICK SUMMARY
Glib contains several classes implemented from scratch (NumPy + Matplotlib) for learning and experimentation:
- LinearReg — multi-feature linear regression using gradient descent and optional live plotting.
- Lasso — L1-style penalized regression (note: implementation detail differs from canonical L1; see Known Issues).
- Ridge — L2-style penalized regression (note: implementation detail differs from canonical L2; see Known Issues).
- PowerReg — learns both coefficients and per-feature exponents p_j for models of form Σ m_j * x_j**p_j + c.
- LogesticReg — logistic-style binary classifier using a sigmoid in training; predict returns logits in current code (inconsistent behavior; see Known Issues).

This document precisely describes usage, signatures, behavior, and exact code-level notes. It includes runnable examples and suggestions to make the library consistent with standard ML implementations.

---

2) REQUIREMENTS & INSTALLATION

Minimal dependencies:
- Python 3.9+
- numpy
- matplotlib

Install:
    pip install numpy matplotlib

---

3) CONVENTIONS & SHAPES

- Input `x`, `xt` : 2D NumPy arrays with shape `(n_samples, n_features)`.
- Target `y` : 1D NumPy array with shape `(n_samples,)`.
- `m` : model weights stored as shape `(1, n_features)`.
- `c` : bias scalar.
- `.fitted` : boolean set to `True` after `.fit()` completes. `.predict()` raises `ValueError` if `.fitted` is `False`.
- `plot` parameter controls visualization. Default in code is string `'False'`. For clarity use boolean `False` or `True` (or `'*'` for feature-grid mode).

---

4) PLOTTING MODES (single summary)

- `plot=False` or `'False'`: no plotting.
- `plot=True`: two-panel (left: features vs y with fitted lines; right: y vs yp).
- `plot='*'`: grid with one subplot per feature plus two summary subplots (`Features vs y` and `y vs yp`).

Plotting uses interactive Matplotlib updates (`.cla()`, `.pause()`, `.show(block=False)`) to animate the training progress.

---

5) CLASS REFERENCE 

The following sections provide exact constructor signatures, parameter descriptions, algorithm flow, return shapes, usage, and code-level notes for each class in `glib.py`.

-------------------------------------------------------------------------------
LINEARREG
-------------------------------------------------------------------------------

Class constructor signature:

    LinearReg(iter=100, lern=0.00001, c=0, plot='False', track=False)

Parameters:
- `iter` (int): gradient descent iterations (default 100)
- `lern` (float): learning rate (default 1e-5)
- `c` (float): initial bias (default 0)
- `plot` (str/bool): `'False'` | False | True | '*' (see plotting modes)
- `track` (bool): if True prints training logs every iteration

Attributes after init:
- `self.iter`, `self.lern`, `self.c`, `self.plot_mode`, `self.track`
- `self.m` (set by fit), `self.fitted` (False initially)

Algorithm (exact flow replicated from code):

1. In `fit(x, y)`:
    - If `x` is not a NumPy array, convert: `x = np.array(x)`.
    - Compute `n_samples, n_features = x.shape`.
    - Initialize weights: `self.m = np.zeros((1, n_features))`.
    - Call `_plot(x, y, yp=None, set_plot=True)` to initialize plotting (if enabled).
    - For `i` in range(`self.iter`):
        - `yp = (self.m @ x.T).flatten() + self.c`  # predictions
        - For each `j` in `range(n_features)`:
            - `md = -(2 / n_samples) * np.sum(x[:, j] * (y - yp))`
            - `self.m[0, j] -= self.lern * md`
        - `cd = -(2 / n_samples) * np.sum(y - yp)`
        - `self.c -= self.lern * cd`
        - `mse = np.mean((y - yp) ** 2)`
        - If `self.track`: print iteration info
        - Call `_plot(x, y, yp=yp, set_plot=False)` to update plot (if enabled)
    - Set `self.fitted = True`

2. In `predict(xt)`:
    - If `not self.fitted`: raise `ValueError("Model not fitted yet! Call .fit(x, y) first.")`
    - Compute `ypp = (self.m @ xt.T).flatten() + self.c`
    - Return `ypp` (1D array length `n_samples_pred`)

Plot helper `_plot(x, y, yp=None, set_plot=True)`:
- If `set_plot == True` and `plot_mode != False`: create subplots
- If `set_plot == False`: update axes with `.cla()`, draw scatter and fitted lines, call `plt.pause(0.0001)`
- At end `plt.show(block=False)`

Usage example (simple):

    import numpy as np
    from glib import LinearReg   # assuming glib.py defines this class
    X = np.random.rand(150, 2)
    y = 3 * X[:,0] - 1.5 * X[:,1] + 0.2 * np.random.randn(150)
    model = LinearReg(iter=500, lern=1e-3, plot=False, track=False)
    model.fit(X, y)
    preds = model.predict(X)   # shape (150,)

Practical notes:
- Default learning rate small — adjust `lern` to data scale.
- Vectorization suggested for large `n_features`: replace loops with `grad = -(2/n) * x.T @ (y - yp)` etc.
- Plotting is interactive; disable on servers.

-------------------------------------------------------------------------------
LASSO
-------------------------------------------------------------------------------

Constructor:

    Lasso(iter=100, lern=0.00001, lamda=0.00001, c=0, plot='False', track=False)

Parameters:
- `lamda` (float): regularization coefficient (default 1e-05)
- Other params same as LinearReg

Exact algorithm differences (extracted from code):
- `yp = (self.m @ x.T).flatten() + self.c`
- Weight update:
    - `md = -(2 / n_samples) * np.sum(x[:, j] * (y - yp)) + (self.lamda * abs(self.m[:, j]))`
    - `self.m[0, j] -= self.lern * md`
- Bias update same as LinearReg

Notes:
- `lamda * abs(self.m[:, j])` is used in gradient. This is NOT the canonical L1 subgradient. Canonical subgradient uses `lamda * sign(m_j)`. Using `abs(m_j)` adds magnitude but removes sign direction — behavior differs.
- For correct Lasso behavior consider replacing:

    md += lamda * np.sign(self.m[0,j])

Usage example:

    from glib import Lasso
    model = Lasso(iter=400, lern=1e-3, lamda=1e-3)
    model.fit(X, y)
    predictions = model.predict(X)

-------------------------------------------------------------------------------
RIDGE
-------------------------------------------------------------------------------

Constructor:

    Ridge(iter=100, lern=0.00001, lamda=0.00001, c=0, plot='False', track=False)

Exact algorithm differences:
- Weight gradient includes a term: `+(self.lamda * self.m[0,j] * self.m[0,j])`
- This adds `lamda * m_j^2` rather than the canonical `2 * lamda * m_j`.

Note:
- Canonical Ridge update: `grad += 2 * lamda * m_j`.
- Suggest fix: replace `lamda * m_j * m_j` with `2 * lamda * m_j` for standard Ridge.

Usage example:

    from glib import Ridge
    model = Ridge(iter=300, lern=1e-3, lamda=0.01)
    model.fit(X, y)
    preds = model.predict(X)

-------------------------------------------------------------------------------
POWERREG
-------------------------------------------------------------------------------

Constructor:

    PowerReg(iter=100, lern=0.00001, c=0, plot='False', track=False)

Purpose:
- Model computes `y ≈ Σ_j m_j * x_j ** p_j + c`
- Learns both `m_j` and `p_j`

Exact algorithm from code (line-by-line behavior):

1. Convert `x` to np.array if needed.
2. `n_samples, n_features = x.shape`
3. Initialize:
    - `self.m = np.zeros((1, n_features))`
    - `self.p = np.ones((1, n_features))`
4. Plot initialization `_plot(x, y, yp=None, set_plot=True)` (if plotting)
5. For `i` in range(iter):
    - Compute `yp = (self.m.flatten() * (x ** self.p.flatten())).sum(axis=1) + self.c`
    - For each feature `j`:
        - `md = -(2/n_samples) * np.sum(x[:, j] * (y - yp))`
        - `self.m[0, j] -= self.lern * md`
        - `pd = -(2 / n_samples) * np.sum((y - yp) * (self.m[0, j] * (x[:, j] ** self.p[0, j]) * np.log(x[:, j])))`
        - `self.p[0, j] -= self.lern * pd`
    - `cd = -(2 / n_samples) * np.sum(y - yp)`
    - `self.c -= self.lern * cd`
    - compute `mse = np.mean((y - yp) ** 2)`
    - optionally print if `track`, update plots

6. set `self.fitted = True`

Predict behavior in code:
- `predict` returns `(self.m @ xt.T).flatten() + self.c` — this is wrong for PowerReg; it should compute `Σ m_j * xt_j ** p_j + c`.

Actionable fix (exact code to replace predict):

    def predict(self, xt):
        if not self.fitted:
            raise ValueError("Model not fitted yet! Call .fit(x, y) first.")
        # ensure xt is numpy array
        if not isinstance(xt, np.ndarray):
            xt = np.array(xt)
        return (self.m.flatten() * (xt ** self.p.flatten())).sum(axis=1) + self.c

Important numerical constraints:
- Code uses `np.log(x[:, j])` in pd; therefore `x[:, j]` must be strictly positive to avoid log(0) or log of negative numbers. If data can have zero/negative values, pre-process:
    X_pos = np.abs(X) + 1e-8

Usage example:

    from glib import PowerReg
    Xp = np.abs(X) + 1e-8
    model = PowerReg(iter=500, lern=1e-4, plot=False)
    model.fit(Xp, y)
    preds = model.predict(Xp)   # after predict fix above

-------------------------------------------------------------------------------
LOGESTICREG (Logistic-style)
-------------------------------------------------------------------------------

Constructor:

    LogesticReg(iter=100, lern=0.0001, lamda=7.5, c=0, plot='False', track=False)

Notes:
- During training `yp` is computed using sigmoid:
    yp = 1 / (1 + np.exp((-self.m @ x.T).flatten() - self.c))
- The weight update computation in code uses a nonstandard expression involving `((1 / yp) ** -2)` and exponential terms. The bias update uses `self.c -= self.lamda * cd` (note lamda scaling for c).

Predict in code:
- Returns linear logits `(self.m @ xt.T).flatten() + self.c`. This is inconsistent — users expect probabilities or labels. Fix by returning sigmoid of logits:

    def predict(self, xt):
        if not self.fitted:
            raise ValueError("Model not fitted yet! Call .fit(x, y) first.")
        logits = (self.m @ xt.T).flatten() + self.c
        logits = np.clip(logits, -500, 500)  # numerical stability
        return 1.0 / (1.0 + np.exp(-logits))

Usage example (after fix):

    from glib import LogesticReg
    clf = LogesticReg(iter=500, lern=1e-4, lamda=1.0, plot=False)
    clf.fit(X, y_binary)   # y_binary should be 0/1
    probs = clf.predict(X)
    preds = (probs > 0.5).astype(int)

Notes on numeric stability:
- Sigmoid of large magnitude inputs overflows; use clipping as above.
- Consider using `logsumexp` patterns for stable computation in more complex settings.

---

6) EXAMPLES — extensive, copy-paste runnable blocks (indented code blocks)

Simple linear regression end-to-end:

    import numpy as np
    from glib import LinearReg

    np.random.seed(42)
    X = np.random.rand(200, 2)
    y = 3.2 * X[:,0] - 1.7 * X[:,1] + 0.05 * np.random.randn(200)

    model = LinearReg(iter=2000, lern=1e-3, plot=False, track=False)
    model.fit(X, y)
    preds = model.predict(X)
    print("MSE:", np.mean((y - preds)**2))

Lasso example:

    import numpy as np
    from glib import Lasso
    np.random.seed(1)
    X = np.random.rand(100, 4)
    # create sparse weights
    true_w = np.array([2.0, 0.0, -1.5, 0.0])
    y = X @ true_w + 0.05 * np.random.randn(100)

    model = Lasso(iter=1000, lern=1e-3, lamda=1e-2, plot=False, track=False)
    model.fit(X, y)
    print("weights:", model.m)

PowerReg example (with positive X):

    import numpy as np
    from glib import PowerReg
    np.random.seed(0)
    X = np.random.rand(150, 2) + 0.1   # ensure > 0
    y = 1.5 * (X[:,0]**2.0) + 0.5 * (X[:,1]**1.5) + 0.02 * np.random.randn(150)

    pr = PowerReg(iter=800, lern=5e-4, plot=False)
    pr.fit(X, y)
    # compute preds manually (until predict fix applied)
    preds = (pr.m.flatten() * (X ** pr.p.flatten())).sum(axis=1) + pr.c
    print("m:", pr.m, "p:", pr.p)

Logistic example:

    import numpy as np
    from glib import LogesticReg
    np.random.seed(2)
    # simple binary data
    X = np.vstack([np.random.randn(100,2)+2, np.random.randn(100,2)-2])
    y = np.hstack([np.ones(100), np.zeros(100)])

    clf = LogesticReg(iter=1000, lern=5e-4, lamda=1.0, plot=False)
    clf.fit(X, y)
    probs = 1.0 / (1.0 + np.exp(-((clf.m @ X.T).flatten() + clf.c)))
    preds = (probs > 0.5).astype(int)
    print("Accuracy:", (preds == y).mean())

---

7) TUTORIALS & DEMO MAPPING (how to build a tutorials index)


- `tutorials/linear_demo.py` 
- `tutorials/lasso_demo.py`
- `tutorials/ridge_demo.py` 
- `tutorials/powerreg_demo.py` 
- `tutorials/logistic_demo.py` 

---



8) PRACTICAL TIPS FOR TRAINING / HYPERPARAMS

- Learning rate (`lern`): tune based on data scale. Typical starting values: `1e-3`, `1e-4`, `1e-2`. Default `1e-5` is conservative.
- Iterations (`iter`): 100–200 often OK for small datasets; 1000+ may be needed for smaller `lern`.
- Regularization (`lamda`): for Lasso/Ridge, start `1e-3`–`1e-2`. Increase if overfitting.
- Feature scaling: apply standardization to avoid one feature dominating gradients.
- For PowerReg: use smaller learning rate for `p` or scale updates (e.g., separate `lern_m` and `lern_p`) to control stability.

---


9) CONTACT

For questions, clarifications, or further edits:
Email: zishaan2426@gmail.com
