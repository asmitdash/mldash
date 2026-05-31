# dashml

**Lint for ML training pipelines.** One import, one call, one PDF report — catch the silent bugs that ruin models in production *before* you ship them.

```bash
pip install dashml          # core (pandas + numpy)
pip install dashml[pdf]     # adds PDF report support (fpdf2)
```

```python
import dashml

report = dashml.check(X_train, y_train, X_test=X_test, y_test=y_test)
print(report)

if not report.ok():
    raise SystemExit("Fix the critical issues before training.")
```

That's the whole API. Pandas DataFrames, NumPy arrays, dicts, and lists all work as inputs. dashml does **not** train any model — it's deterministic, runs in seconds, and depends only on `pandas` + `numpy` (PDF output is an optional extra).

---

## Why this exists

Every ML pipeline has small mistakes that go unnoticed: a column derived from the label sneaks in, the test set was sampled before the split was made, two columns are byte-identical, the same user appears in train and test. Each one looks fine in code review and silently inflates your accuracy. Then production happens.

**dashml catches those mistakes before they break your pipeline.** It's a static-analysis layer for training data — the way `eslint` is for JavaScript.

It's deliberately scoped: only training-data and pipeline integrity. It doesn't train models, tune hyperparameters, or visualize distributions — pandas, sklearn, and ydata-profiling already do those things well.

---

## What it catches

| Code | Severity | What it catches |
|------|----------|-----------------|
| `TL001` | critical / warning | Exact-duplicate rows leaking from train into test |
| `TL002` | warning | Near-duplicate rows (numeric round-off contamination) |
| `TL003` | critical / warning / **info** | Target leakage — feature ↔ label association, tiered (≥0.98 / ≥0.85 / ≥0.70) |
| `TL004` | warning | Constant or near-constant features |
| `TL005` | warning | Duplicate feature columns |
| `TL006` | warning | Train/test distribution drift (KS for numeric, PSI for categorical) |
| `TL007` | critical / warning | Severe class imbalance |
| `TL008` | warning | Missingness rate differs between train and test |
| `TL009` | critical | Schema mismatch (columns or dtypes differ) |
| `TL010` | warning | ID-like features (cardinality ≈ row count) |
| **`TL011`** | critical / warning | **Temporal leakage** — test rows at or before the latest train timestamp |
| **`TL012`** | critical / warning | **Group leakage** — same group ID (user / session / patient) in train and test |
| **`TL013`** | critical | **Preprocessing leakage** — pipeline state depends on data outside the train split |
| **`TL014`** | warning | **Target-aware encoder** without cross-validation wrapping |

Each finding tells you the affected **column(s)**, the **severity**, and **how to fix it** — not just that something is wrong.

---

## Why it actually helps

The big-deal bugs in production ML aren't algorithm bugs. They're data hygiene bugs that pass code review:

- A feature derived from the label sneaks in. The model gets 99% accuracy. Production gets 60%.
- The same user's rows end up in train *and* test. Cross-validation looks great. Production looks terrible.
- A timestamp column is fed in as a feature. The model overfits to row identity.
- The test set was shuffled across time. Your "evaluation" is measuring transfer, not skill.
- `StandardScaler.fit_transform(X)` was called *before* the train/test split. Test statistics leaked into training.

`dashml.check(...)` is a single call that catches these before training, with concrete fixes.

---

## Demo: with vs without dashml

The repo ships [`examples/demo.py`](examples/demo.py) — a synthetic fraud-detection dataset (8 000 transactions, 600 users, 90-day window) with **three mistakes** baked into the naive pipeline:

1. Shuffled split instead of chronological → **temporal leakage**
2. Row-level split that puts the same users in train *and* test → **group leakage**
3. `StandardScaler.fit_transform(X)` before splitting → **preprocessing leakage**

Run it:

```bash
cd examples
pip install -r requirements.txt
pip install dashml[pdf]
python demo.py
```

You get this verdict:

| Metric | Naive (3 bugs) | Honest (dashml-cleaned) | Inflation |
|---|---|---|---|
| accuracy | **0.8717** | 0.8495 | +0.0222 |
| f1 | **0.6805** | 0.6569 | +0.0236 |
| roc_auc | **0.9065** | 0.8959 | +0.0106 |

The naive numbers look fine. They're not — they're the score of a model that's secretly cheating. dashml flags all three bugs as **critical** and refuses to ok() the run.

The demo also writes a single audit document — see [`examples/sample_report.pdf`](examples/sample_report.pdf) and [`examples/sample_report.html`](examples/sample_report.html) for what the output looks like.

---

## Generate a PDF / HTML audit report

```python
report = dashml.check(
    X_train, y_train, X_test, y_test,
    time_col="timestamp",        # enables TL011 (temporal leakage)
    group_key="user_id",         # enables TL012 (group leakage)
)

report.to_pdf(
    "audit.pdf",
    title="dashml audit -- fraud model v3",
    dataset_name="transactions Q1 2024",
    metrics_before={"accuracy": 0.8717, "f1": 0.6805, "roc_auc": 0.9065},
    metrics_after ={"accuracy": 0.8495, "f1": 0.6569, "roc_auc": 0.8959},
)

# Or, for embedding in a notebook / dashboard:
html = report.to_html(title="...", metrics_before=..., metrics_after=...)
```

The report contains: pass/fail banner, summary cards, performance comparison with deltas, every finding with `what` / `detail` / `fix` / `columns` — designed to print or share with a stakeholder.

---

## Audit a sklearn pipeline

`dashml.check()` looks at data. `dashml.audit_pipeline()` looks at code:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import dashml

candidate = Pipeline([
    ("scale", StandardScaler()),
    ("clf",   GradientBoostingClassifier(random_state=42)),
])

report = dashml.audit_pipeline(candidate, X, y)   # raw, unsplit X, y
print(report)
```

It clones the pipeline twice, fits one on the train split and one on the full dataset, and compares `transform(X_test)` outputs. If they diverge, the pipeline has data-dependent state (scaler stats, imputer means, encoder maps) that would leak when fit on full data — flagged as `TL013` critical.

It also flags target-aware encoders (`TargetEncoder`, `CatBoostEncoder`, etc.) as `TL014` if they appear without explicit CV wrapping.

---

## API reference

```python
dashml.check(
    X_train, y_train,
    X_test=None, y_test=None,
    *,
    task="auto",                      # "auto" | "classification" | "regression"
    time_col=None,                    # column name in X_train/X_test for TL011
    group_key=None,                   # column name OR Series for TL012
    group_key_test=None,              # defaults to group_key when it's a string
) -> Report

dashml.audit_pipeline(
    pipeline, X, y,
    *,
    task="auto",
    test_size=0.30,
    random_state=42,
    atol=1e-6,
) -> Report
```

`Report`:

- `report.ok()` — `True` if no critical findings.
- `report.findings`, `report.critical`, `report.warnings`, `report.infos` — lists of `Finding`.
- `print(report)` — human-readable terminal summary.
- `report.to_dict()` — JSON-serializable dict (good for CI logs / artifacts).
- `report.to_html(...)` — single-page self-contained HTML.
- `report.to_pdf(path, ...)` — single audit document. Requires `dashml[pdf]`.

Each `Finding` has: `code`, `severity` (`critical` / `warning` / `info`), `message`, `fix`, `columns`, `details`.

---

## Use it in CI

```python
import dashml, sys

report = dashml.check(X_train, y_train, X_test, y_test,
                     time_col="timestamp", group_key="user_id")
report.to_pdf("audit.pdf", title="CI audit")   # optional artifact
sys.exit(0 if report.ok() else 1)
```

A failed `report.ok()` blocks the merge. The PDF / HTML can be uploaded as a CI artifact for review.

---

## Scope, on purpose

dashml is **only** a linter for training-data and pipeline-integrity bugs. It doesn't:

- train models (use sklearn / lightning / xgboost),
- tune hyperparameters (use Optuna / Ray Tune),
- track experiments (use MLflow / W&B),
- profile data (use ydata-profiling / sweetviz),
- explain predictions (use SHAP / lime).

Doing one thing well is the point. If `dashml.check()` returns clean, you can trust your pipeline isn't silently broken — and that's all it claims to do.

---

## Development

```bash
git clone https://github.com/<your-username>/dashml
cd dashml
pip install -e ".[dev]"
pytest                        # 29 tests, ~3 seconds
```

---

## License

MIT — see [LICENSE](LICENSE).
