"""mldash in 20 lines.

Plants three classic bugs in synthetic data, runs mldash.check, and prints
the report. For the full walk-through with metrics + PDF/HTML output, see
demo.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mldash

rng = np.random.default_rng(0)
n = 800

X = pd.DataFrame({
    "age":      rng.integers(18, 80, n),
    "income":   rng.normal(50_000, 15_000, n),
    "region":   rng.choice(["N", "S", "E", "W"], n),
    "dead_col": 1,                                       # TL004 - constant
    "row_id":   np.arange(n),                            # TL010 - ID-like
})
y = pd.Series((X["age"] > 40).astype(int), name="target")
X["target_leaked"] = y.map({0: "no", 1: "yes"})          # TL003 - leakage

split = int(n * 0.7)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

report = mldash.check(X_train, y_train, X_test, y_test)
print(report)
print("\nready to ship?", report.ok())
