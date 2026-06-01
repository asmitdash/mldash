"""mldash -- lint for ML training pipelines.

Single-file library. Catches silent bugs in your training data before they
ruin your model: target leakage, train/test contamination, schema drift,
ID-like features, duplicate columns, severe imbalance, and more.

Quickstart:

    import mldash

    report = mldash.check(X_train, y_train, X_test=X_test, y_test=y_test)
    print(report)

    if not report.ok():
        raise SystemExit("Fix the critical issues above before training.")

The linter is deterministic and dependency-light (pandas, numpy, scipy).
It does NOT train models. It runs in seconds on most datasets.

Author: Asmit Dash
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

__version__ = "0.3.0"
__all__ = ["check", "audit_pipeline", "Report", "Finding", "Severity", "__version__"]


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


_SEVERITY_ORDER = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
_SEVERITY_GLYPH = {
    Severity.CRITICAL: "[X]",
    Severity.WARNING: "[!]",
    Severity.INFO: "[i]",
}

_TL_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "TL001": {
        "title": "Exact-duplicate rows leaking train -> test",
        "what":  "One or more rows in your test set are byte-identical to rows in train. Your test accuracy is partly memorization, not generalization.",
    },
    "TL002": {
        "title": "Near-duplicate rows leaking train -> test",
        "what":  "Rows match across train/test after rounding numerics. Often happens when the same entity (user, session) is split across both sides at the row level instead of the group level.",
    },
    "TL003": {
        "title": "Target leakage",
        "what":  "A feature has near-perfect statistical association with the label. Almost always means the feature is derived from the label, computed post-event, or otherwise unavailable at inference time.",
    },
    "TL004": {
        "title": "Constant or near-constant features",
        "what":  "Features with a single value (or one value covering >=99% of rows). They contribute zero predictive signal and can mislead regularization heuristics.",
    },
    "TL005": {
        "title": "Duplicate feature columns",
        "what":  "Two or more columns hold identical values. They inflate feature-importance for the underlying signal and slow training.",
    },
    "TL006": {
        "title": "Train/test distribution drift",
        "what":  "A feature's distribution differs sharply between train and test (KS>=0.2 numeric, PSI>=0.25 categorical). Your evaluation is measuring transfer, not generalization.",
    },
    "TL007": {
        "title": "Severe class imbalance",
        "what":  "The minority class is below 5% (or below 1% -- critical). Plain accuracy will be misleading and a constant predictor may beat your model.",
    },
    "TL008": {
        "title": "Train/test missingness mismatch",
        "what":  "Missing-rate differs by >=10 percentage points between train and test on the same column. Imputers fit on train will misbehave on test.",
    },
    "TL009": {
        "title": "Schema mismatch",
        "what":  "Train and test have different columns or different dtypes for the same column. Inference-time behavior will silently diverge from training.",
    },
    "TL010": {
        "title": "ID-like features",
        "what":  "A column has cardinality >=95% of rows. Likely a row ID, timestamp, or hash. If used as-is the model will overfit to row identity.",
    },
    "TL011": {
        "title": "Temporal leakage",
        "what":  "Test rows include timestamps at or before the latest train timestamp. The model sees the future during training, then evaluates on the past -- production accuracy will be much worse.",
    },
    "TL012": {
        "title": "Group leakage",
        "what":  "The same group identifier (user_id, session_id, patient_id) appears in both train and test splits. Cross-validation overstates accuracy because the model has seen this entity.",
    },
    "TL013": {
        "title": "Preprocessing fit-on-full-data leakage",
        "what":  "The pipeline produces different transform() outputs for X_test depending on whether it was fit on the training subset or the full dataset. This means scaler / imputer / encoder state depends on test rows -- silent leakage.",
    },
    "TL014": {
        "title": "Target-aware encoder without cross-validation",
        "what":  "The pipeline contains a target encoder (TargetEncoder, CatBoost, etc.) that uses the label to compute features. These leak label information into training rows unless wrapped in cross-validated fitting.",
    },
}


@dataclass(frozen=True)
class Finding:
    code: str
    severity: Severity
    message: str
    fix: str
    columns: tuple[str, ...] = field(default_factory=tuple)
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        glyph = _SEVERITY_GLYPH[self.severity]
        cols = f" cols={list(self.columns)}" if self.columns else ""
        return (
            f"{glyph} {self.code} {self.severity.value.upper()}: "
            f"{self.message}{cols}\n    fix: {self.fix}"
        )


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    def add(self, finding: Finding | None) -> None:
        if finding is not None:
            self.findings.append(finding)

    def extend(self, findings: Iterable[Finding]) -> None:
        for f in findings:
            self.add(f)

    @property
    def critical(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.CRITICAL]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.WARNING]

    @property
    def infos(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.INFO]

    def ok(self) -> bool:
        """True if no CRITICAL findings."""
        return not self.critical

    def sorted(self) -> list[Finding]:
        return sorted(
            self.findings, key=lambda f: (_SEVERITY_ORDER[f.severity], f.code)
        )

    def summary(self) -> str:
        c, w, i = len(self.critical), len(self.warnings), len(self.infos)
        return f"mldash: {c} critical, {w} warning, {i} info"

    def __str__(self) -> str:
        if not self.findings:
            return "mldash: no issues found."
        body = "\n".join(str(f) for f in self.sorted())
        return f"{self.summary()}\n{body}"

    def __bool__(self) -> bool:
        return bool(self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "critical": len(self.critical),
                "warning": len(self.warnings),
                "info": len(self.infos),
            },
            "findings": [
                {
                    "code": f.code,
                    "severity": f.severity.value,
                    "message": f.message,
                    "fix": f.fix,
                    "columns": list(f.columns),
                    "details": f.details,
                }
                for f in self.sorted()
            ],
        }

    def to_html(
        self,
        *,
        title: str = "mldash report",
        dataset_name: str | None = None,
        metrics_before: dict[str, Any] | None = None,
        metrics_after: dict[str, Any] | None = None,
    ) -> str:
        """Render the report as a single-page, self-contained HTML document.

        Pass ``metrics_before`` / ``metrics_after`` (e.g. ``{"accuracy": 0.99}``)
        to include a side-by-side performance comparison. The HTML inlines all
        styles -- save it, open in any browser, print to PDF.
        """
        return _render_html(
            self,
            title=title,
            dataset_name=dataset_name,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
        )

    def to_pdf(
        self,
        path: str | "Path",
        *,
        title: str = "mldash report",
        dataset_name: str | None = None,
        metrics_before: dict[str, Any] | None = None,
        metrics_after: dict[str, Any] | None = None,
    ) -> str:
        """Render the report as a single-page PDF and write it to ``path``.

        Requires the optional ``fpdf2`` dependency. Install with::

            pip install mldash[pdf]
            # or:
            pip install fpdf2

        Returns the absolute path the PDF was written to.
        """
        try:
            from fpdf import FPDF  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "to_pdf() requires fpdf2. Install with `pip install mldash[pdf]` "
                "or `pip install fpdf2`."
            ) from e
        from pathlib import Path as _Path

        out = _Path(path).resolve()
        _render_pdf(
            self,
            out,
            FPDF=FPDF,
            title=title,
            dataset_name=dataset_name,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
        )
        return str(out)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_dataframe(X, name: str = "X") -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    if isinstance(X, (list, tuple, dict)):
        return pd.DataFrame(X)
    raise TypeError(
        f"{name} must be DataFrame, ndarray, dict, or list -- got {type(X).__name__}"
    )


def _to_series(y, name: str = "y") -> pd.Series:
    if isinstance(y, pd.Series):
        return y
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(f"{name} must be 1-D; got {y.shape[1]} columns")
        return y.iloc[:, 0]
    arr = np.asarray(y).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D; got shape {arr.shape}")
    return pd.Series(arr, name=name)


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(exclude=[np.number]).columns.tolist()


def _infer_task(y: pd.Series) -> str:
    if y.dtype == bool or y.dtype.name == "category" or y.dtype == object:
        return "classification"
    nunique = y.nunique(dropna=True)
    if np.issubdtype(y.dtype, np.integer) and nunique <= max(20, int(0.05 * len(y))):
        return "classification"
    if nunique <= 20:
        return "classification"
    return "regression"


def _hash_rows(df: pd.DataFrame) -> pd.Series:
    safe = df.copy()
    for col in safe.columns:
        if safe[col].dtype == object:
            safe[col] = safe[col].astype(str)
    return pd.util.hash_pandas_object(safe, index=False)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _check_schema_mismatch(
    X_train: pd.DataFrame, X_test: pd.DataFrame | None
) -> list[Finding]:
    """TL009 -- column / dtype mismatch between train and test."""
    if X_test is None:
        return []

    findings: list[Finding] = []
    train_cols = list(X_train.columns)
    test_cols = list(X_test.columns)

    missing_in_test = [c for c in train_cols if c not in test_cols]
    extra_in_test = [c for c in test_cols if c not in train_cols]

    if missing_in_test or extra_in_test:
        findings.append(
            Finding(
                code="TL009",
                severity=Severity.CRITICAL,
                message=(
                    f"Train/test column mismatch -- "
                    f"missing in test: {missing_in_test or 'none'}, "
                    f"extra in test: {extra_in_test or 'none'}"
                ),
                fix=(
                    "Align columns before training. "
                    "Use X_test = X_test[X_train.columns] after verifying both "
                    "sides have the same fields."
                ),
                columns=tuple(missing_in_test + extra_in_test),
            )
        )

    shared = [c for c in train_cols if c in test_cols]
    dtype_mismatches: list[str] = []
    for col in shared:
        if str(X_train[col].dtype) != str(X_test[col].dtype):
            dtype_mismatches.append(
                f"{col} ({X_train[col].dtype} vs {X_test[col].dtype})"
            )
    if dtype_mismatches:
        findings.append(
            Finding(
                code="TL009",
                severity=Severity.CRITICAL,
                message=(
                    f"Train/test dtype mismatch on {len(dtype_mismatches)} "
                    f"column(s): {dtype_mismatches[:5]}"
                ),
                fix=(
                    "Cast both sides to the same dtype before training. "
                    "Mismatched dtypes silently change model behavior at inference."
                ),
                columns=tuple(c.split(" ")[0] for c in dtype_mismatches),
            )
        )

    return findings


def _check_exact_duplicates(
    X_train: pd.DataFrame, X_test: pd.DataFrame | None
) -> list[Finding]:
    """TL001 -- rows in test that exactly match rows in train."""
    if X_test is None or len(X_test) == 0:
        return []

    shared = [c for c in X_train.columns if c in X_test.columns]
    if not shared:
        return []

    train_hashes = set(_hash_rows(X_train[shared]).tolist())
    test_hashes = _hash_rows(X_test[shared])

    n_leak = int(test_hashes.isin(train_hashes).sum())
    if n_leak == 0:
        return []

    pct = 100 * n_leak / len(X_test)
    severity = Severity.CRITICAL if pct >= 1 else Severity.WARNING
    return [
        Finding(
            code="TL001",
            severity=severity,
            message=(
                f"{n_leak} rows in X_test exactly match rows in X_train "
                f"({pct:.2f}% of test set)."
            ),
            fix=(
                "Drop the overlapping rows from X_test before evaluation. "
                "Re-split with a deterministic seed and a stable key "
                "(e.g., user_id) to prevent recurrence."
            ),
            details={"n_leaking_rows": n_leak, "pct_of_test": round(pct, 4)},
        )
    ]


def _check_near_duplicates(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None,
    *,
    decimals: int = 4,
) -> list[Finding]:
    """TL002 -- rows that match after rounding numerics to N decimals."""
    if X_test is None or len(X_test) == 0:
        return []

    shared = [c for c in X_train.columns if c in X_test.columns]
    if not shared:
        return []

    train = X_train[shared].copy()
    test = X_test[shared].copy()

    num_cols = _numeric_columns(train)
    if num_cols:
        train[num_cols] = train[num_cols].round(decimals)
        test[num_cols] = test[num_cols].round(decimals)

    cat_cols = _categorical_columns(train)
    for c in cat_cols:
        train[c] = train[c].astype(str)
        test[c] = test[c].astype(str)

    train_hashes = set(_hash_rows(train).tolist())
    test_hashes = _hash_rows(test)
    n_near = int(test_hashes.isin(train_hashes).sum())

    exact_train = set(_hash_rows(X_train[shared]).tolist())
    n_exact = int(_hash_rows(X_test[shared]).isin(exact_train).sum())
    n_only_near = n_near - n_exact
    if n_only_near <= 0:
        return []

    pct = 100 * n_only_near / len(X_test)
    return [
        Finding(
            code="TL002",
            severity=Severity.WARNING,
            message=(
                f"{n_only_near} rows in X_test are near-duplicates of train rows "
                f"(match after rounding numerics to {decimals} decimals; "
                f"{pct:.2f}% of test)."
            ),
            fix=(
                "If your splitting is groupwise (per user, per session), "
                "train/test contamination is happening at the group level. "
                "Use GroupKFold or split by the entity, not the row."
            ),
            details={
                "n_near_duplicate_rows": n_only_near,
                "pct_of_test": round(pct, 4),
            },
        )
    ]


# --- target leakage helpers ----------


def _chi2_stat(observed: np.ndarray) -> float:
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    if total == 0:
        return 0.0
    expected = row_sums @ col_sums / total
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(expected > 0, (observed - expected) ** 2 / expected, 0.0)
    return float(terms.sum())


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    contingency = pd.crosstab(x, y)
    if contingency.size == 0 or min(contingency.shape) < 2:
        return 0.0
    chi2 = _chi2_stat(contingency.values)
    n = contingency.values.sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = contingency.shape
    denom = min(k - 1, r - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(max(0.0, phi2) / denom))


def _numeric_target_assoc(x: pd.Series, y: pd.Series, task: str) -> float:
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]
    if len(x) < 5 or x.nunique() < 2:
        return 0.0
    if task == "regression":
        return float(abs(pd.Series(x.values).rank().corr(pd.Series(y.values).rank())))
    if y.nunique() <= 2:
        y_enc = pd.Categorical(y).codes
        if np.std(x.values) == 0 or np.std(y_enc) == 0:
            return 0.0
        return float(abs(np.corrcoef(x.values, y_enc)[0, 1]))
    # Multiclass: eta = sqrt(1 - within_class_var / total_var). Captures
    # numeric features that separate classes (incl. label + small noise leaks).
    grand_var = float(np.var(x.values))
    if grand_var == 0:
        return 0.0
    n = len(x)
    within = 0.0
    for _, grp in x.groupby(y.values):
        within += float(np.var(grp.values)) * len(grp) / n
    eta2 = max(0.0, 1.0 - within / grand_var)
    return float(np.sqrt(eta2))


def _categorical_target_assoc(x: pd.Series, y: pd.Series, task: str) -> float:
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]
    if len(x) < 5 or x.nunique() < 2:
        return 0.0
    if task == "classification":
        return _cramers_v(x, y)
    grand_var = float(np.var(y.values))
    if grand_var == 0:
        return 0.0
    within = 0.0
    n = len(y)
    for _, grp in y.groupby(x.values):
        within += np.var(grp.values) * len(grp) / n
    eta2 = max(0.0, 1.0 - within / grand_var)
    return float(np.sqrt(eta2))


_LEAK_CRITICAL = 0.98
_LEAK_WARN = 0.85
_LEAK_INFO = 0.70


def _check_target_leakage(
    X_train: pd.DataFrame, y_train: pd.Series, task: str
) -> list[Finding]:
    """TL003 -- feature association with the label, tiered by severity.

    Three tiers cover the realistic gray zone:
        critical  >= 0.98  -- almost certainly leakage
        warning   >= 0.85  -- very suspicious; verify inference-time availability
        info      >= 0.70  -- worth a glance, especially if it dominates importance
    """
    findings: list[Finding] = []
    crit: list[tuple[str, float]] = []
    warn: list[tuple[str, float]] = []
    info: list[tuple[str, float]] = []

    def bucket(col: str, s: float) -> None:
        if s >= _LEAK_CRITICAL:
            crit.append((col, s))
        elif s >= _LEAK_WARN:
            warn.append((col, s))
        elif s >= _LEAK_INFO:
            info.append((col, s))

    for col in _numeric_columns(X_train):
        bucket(col, _numeric_target_assoc(X_train[col], y_train, task))

    for col in _categorical_columns(X_train):
        bucket(col, _categorical_target_assoc(X_train[col], y_train, task))

    if crit:
        findings.append(
            Finding(
                code="TL003",
                severity=Severity.CRITICAL,
                message=(
                    f"{len(crit)} feature(s) have >={_LEAK_CRITICAL:.2f} association "
                    "with the target -- almost certainly leakage."
                ),
                fix=(
                    "A feature this predictive of the label is almost always "
                    "derived from the label, computed post-event, or unavailable "
                    "at inference time. Trace its origin and drop it if it leaks."
                ),
                columns=tuple(c for c, _ in crit),
                details={c: round(s, 4) for c, s in crit},
            )
        )
    if warn:
        findings.append(
            Finding(
                code="TL003",
                severity=Severity.WARNING,
                message=(
                    f"{len(warn)} feature(s) have very high "
                    f"(>={_LEAK_WARN:.2f}) association with the target."
                ),
                fix=(
                    "Verify these features are computable at inference time using "
                    "only data available before the prediction is made. If they "
                    "summarize post-event state, they are leakage."
                ),
                columns=tuple(c for c, _ in warn),
                details={c: round(s, 4) for c, s in warn},
            )
        )
    if info:
        findings.append(
            Finding(
                code="TL003",
                severity=Severity.INFO,
                message=(
                    f"{len(info)} feature(s) in the gray zone "
                    f"(>={_LEAK_INFO:.2f} association with target)."
                ),
                fix=(
                    "These could be legitimate strong signal OR partial leakage. "
                    "If a single such feature dominates feature_importances_, "
                    "audit it: is it computed using future / target-derived data?"
                ),
                columns=tuple(c for c, _ in info),
                details={c: round(s, 4) for c, s in info},
            )
        )
    return findings


def _check_constant_features(X_train: pd.DataFrame) -> list[Finding]:
    """TL004 -- constant or near-constant features."""
    constant: list[str] = []
    near_constant: list[tuple[str, float]] = []
    for col in X_train.columns:
        s = X_train[col]
        non_null = s.dropna()
        if len(non_null) == 0 or non_null.nunique() <= 1:
            constant.append(col)
            continue
        top = non_null.value_counts(normalize=True).iloc[0]
        if top >= 0.99:
            near_constant.append((col, float(top)))

    findings: list[Finding] = []
    if constant:
        findings.append(
            Finding(
                code="TL004",
                severity=Severity.WARNING,
                message=f"{len(constant)} constant feature(s) -- they carry zero signal.",
                fix=(
                    "Drop them. They cost training time, hurt regularization "
                    "heuristics, and bloat feature-importance plots."
                ),
                columns=tuple(constant),
            )
        )
    if near_constant:
        findings.append(
            Finding(
                code="TL004",
                severity=Severity.WARNING,
                message=f"{len(near_constant)} near-constant feature(s) (>=99% one value).",
                fix=(
                    "Likely safe to drop. If you keep them, ensure your model "
                    "handles low-variance inputs (tree models tolerate this; "
                    "regularized linear models do not)."
                ),
                columns=tuple(c for c, _ in near_constant),
                details={c: round(p, 4) for c, p in near_constant},
            )
        )
    return findings


def _check_duplicate_columns(X_train: pd.DataFrame) -> list[Finding]:
    """TL005 -- columns with identical values."""
    if X_train.shape[1] < 2:
        return []

    groups: dict[int, list[str]] = {}
    for col in X_train.columns:
        try:
            h = int(pd.util.hash_pandas_object(X_train[col], index=False).sum())
        except TypeError:
            h = int(
                pd.util.hash_pandas_object(X_train[col].astype(str), index=False).sum()
            )
        groups.setdefault(h, []).append(col)

    confirmed: list[list[str]] = []
    for g in groups.values():
        if len(g) < 2:
            continue
        first = X_train[g[0]]
        same = [g[0]]
        for other in g[1:]:
            if X_train[other].equals(first):
                same.append(other)
        if len(same) > 1:
            confirmed.append(same)

    if not confirmed:
        return []

    flat = tuple(c for g in confirmed for c in g[1:])
    return [
        Finding(
            code="TL005",
            severity=Severity.WARNING,
            message=f"{len(confirmed)} group(s) of duplicate columns: {confirmed}",
            fix=(
                "Drop the redundant copies. Duplicates inflate feature-importance "
                "for the underlying signal and slow training."
            ),
            columns=flat,
        )
    ]


def _check_id_like_features(X_train: pd.DataFrame) -> list[Finding]:
    """TL010 -- features with cardinality ~= row count."""
    n = len(X_train)
    if n == 0:
        return []
    suspects: list[tuple[str, float]] = []
    for col in X_train.columns:
        s = X_train[col]
        # Continuous floats are expected to be ~100% unique but are not IDs.
        # Only flag integer/string columns (the realistic ID carriers).
        if pd.api.types.is_float_dtype(s):
            continue
        nunique = s.nunique(dropna=True)
        ratio = nunique / n
        if ratio >= 0.95 and nunique >= 50:
            suspects.append((col, float(ratio)))
    if not suspects:
        return []
    return [
        Finding(
            code="TL010",
            severity=Severity.WARNING,
            message=(
                f"{len(suspects)} ID-like feature(s) (cardinality >=95% of rows): "
                f"{tuple(c for c, _ in suspects)}"
            ),
            fix=(
                "Likely a row identifier, timestamp, or hash. If used as a "
                "categorical feature the model will overfit to row identity. "
                "Drop, or extract real signal (date parts from a timestamp, "
                "prefix from an ID)."
            ),
            columns=tuple(c for c, _ in suspects),
            details={c: round(r, 4) for c, r in suspects},
        )
    ]


def _check_class_imbalance(y_train: pd.Series, task: str) -> list[Finding]:
    """TL007 -- severe class imbalance."""
    if task != "classification":
        return []
    counts = y_train.value_counts(dropna=True)
    if len(counts) < 2:
        return [
            Finding(
                code="TL007",
                severity=Severity.CRITICAL,
                message=f"y_train contains only one class: {list(counts.index)}.",
                fix="A classifier cannot be trained on a single class. Re-check your split and labels.",
            )
        ]
    minority_pct = 100 * counts.min() / counts.sum()
    if minority_pct < 1:
        sev = Severity.CRITICAL
    elif minority_pct < 5:
        sev = Severity.WARNING
    else:
        return []
    return [
        Finding(
            code="TL007",
            severity=sev,
            message=(
                f"Severe class imbalance -- minority class is {minority_pct:.2f}% "
                f"of training rows. Class counts: {counts.to_dict()}."
            ),
            fix=(
                "Plain accuracy will be misleading. Use class weighting "
                "(class_weight='balanced'), resampling (SMOTE/undersampling), "
                "stratified splits, and report precision/recall/PR-AUC instead "
                "of accuracy."
            ),
            details={"counts": {str(k): int(v) for k, v in counts.items()}},
        )
    ]


def _ks_2samp(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic D in [0, 1]. Manual to avoid scipy hard-dep."""
    a = np.sort(a[~np.isnan(a)])
    b = np.sort(b[~np.isnan(b)])
    if len(a) == 0 or len(b) == 0:
        return 0.0
    all_vals = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, all_vals, side="right") / len(a)
    cdf_b = np.searchsorted(b, all_vals, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _psi(a: np.ndarray, b: np.ndarray, *, bins: int = 10) -> float:
    """Population Stability Index for categorical-like distributions."""
    a_counts = pd.Series(a).value_counts(normalize=True)
    b_counts = pd.Series(b).value_counts(normalize=True)
    keys = a_counts.index.union(b_counts.index)
    eps = 1e-6
    psi = 0.0
    for k in keys:
        pa = float(a_counts.get(k, eps)) or eps
        pb = float(b_counts.get(k, eps)) or eps
        psi += (pa - pb) * np.log(pa / pb)
    return float(psi)


def _check_train_test_drift(
    X_train: pd.DataFrame, X_test: pd.DataFrame | None
) -> list[Finding]:
    """TL006 -- per-feature distribution drift between train and test."""
    if X_test is None or len(X_test) == 0:
        return []
    shared = [c for c in X_train.columns if c in X_test.columns]
    if not shared:
        return []

    drifted: list[tuple[str, float, str]] = []  # col, score, method
    for col in shared:
        if col in _numeric_columns(X_train):
            a = X_train[col].to_numpy(dtype=float, copy=False)
            b = X_test[col].to_numpy(dtype=float, copy=False)
            d = _ks_2samp(a, b)
            if d >= 0.2:
                drifted.append((col, d, "KS"))
        else:
            a = X_train[col].astype(str).to_numpy()
            b = X_test[col].astype(str).to_numpy()
            psi = _psi(a, b)
            if psi >= 0.25:
                drifted.append((col, psi, "PSI"))

    if not drifted:
        return []
    return [
        Finding(
            code="TL006",
            severity=Severity.WARNING,
            message=(
                f"{len(drifted)} feature(s) drift between train and test "
                "(KS>=0.2 numeric / PSI>=0.25 categorical)."
            ),
            fix=(
                "Train and test should be drawn from the same distribution. "
                "If they're not, your evaluation is measuring transfer, not "
                "generalization. Re-shuffle, fix temporal leakage, or "
                "explicitly model distribution shift."
            ),
            columns=tuple(c for c, _, _ in drifted),
            details={c: {"score": round(s, 4), "method": m} for c, s, m in drifted},
        )
    ]


def _check_missingness_mismatch(
    X_train: pd.DataFrame, X_test: pd.DataFrame | None
) -> list[Finding]:
    """TL008 -- missingness rate differs sharply between train and test."""
    if X_test is None or len(X_test) == 0:
        return []
    shared = [c for c in X_train.columns if c in X_test.columns]
    if not shared:
        return []

    bad: list[tuple[str, float, float]] = []
    for col in shared:
        a = X_train[col].isna().mean()
        b = X_test[col].isna().mean()
        if abs(a - b) >= 0.1:
            bad.append((col, float(a), float(b)))

    if not bad:
        return []
    return [
        Finding(
            code="TL008",
            severity=Severity.WARNING,
            message=(
                f"{len(bad)} feature(s) have >=10pp difference in missing-rate "
                "between train and test."
            ),
            fix=(
                "Missingness shift breaks imputers fit on train. Either re-fit "
                "the imputer on the union, encode missing-as-its-own-category, "
                "or investigate why the data pipeline produces different gaps."
            ),
            columns=tuple(c for c, _, _ in bad),
            details={
                c: {"train_na_rate": round(a, 4), "test_na_rate": round(b, 4)}
                for c, a, b in bad
            },
        )
    ]


# ---------------------------------------------------------------------------
# TL011 -- temporal leakage
# ---------------------------------------------------------------------------


def _coerce_time(s: pd.Series) -> pd.Series:
    """Convert a column to datetime ns. Accepts datetime, ISO strings, ints (epoch)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        # Heuristic: < 1e10 -> seconds, otherwise ns/ms; treat as seconds-since-epoch
        return pd.to_datetime(s, unit="s", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def _check_temporal_leakage(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None,
    time_col: str | None,
) -> list[Finding]:
    """TL011 -- test rows are not strictly after train rows on time_col."""
    if time_col is None:
        return []
    if time_col not in X_train.columns:
        return [
            Finding(
                code="TL011",
                severity=Severity.WARNING,
                message=f"time_col={time_col!r} not present in X_train -- temporal check skipped.",
                fix="Pass the exact column name that holds the row timestamp.",
            )
        ]

    findings: list[Finding] = []
    train_t = _coerce_time(X_train[time_col]).dropna()
    if train_t.empty:
        return [
            Finding(
                code="TL011",
                severity=Severity.WARNING,
                message=f"time_col={time_col!r} could not be parsed as datetime.",
                fix="Convert the column to pandas datetime before calling check().",
            )
        ]

    train_min, train_max = train_t.min(), train_t.max()

    if X_test is not None and time_col in X_test.columns:
        test_t = _coerce_time(X_test[time_col]).dropna()
        if not test_t.empty:
            test_min, test_max = test_t.min(), test_t.max()

            n_before = int((test_t < train_min).sum())
            n_overlap = int((test_t <= train_max).sum())  # at or before train's last
            pct_overlap = 100 * n_overlap / len(test_t) if len(test_t) else 0

            if pct_overlap >= 1:
                sev = Severity.CRITICAL if pct_overlap >= 5 else Severity.WARNING
                findings.append(
                    Finding(
                        code="TL011",
                        severity=sev,
                        message=(
                            f"{n_overlap} of {len(test_t)} test rows ({pct_overlap:.2f}%) "
                            f"have timestamps at or before the latest train timestamp. "
                            f"train: [{train_min} -> {train_max}]; "
                            f"test:  [{test_min} -> {test_max}]"
                        ),
                        fix=(
                            "Split chronologically: every test timestamp must be strictly "
                            "greater than every train timestamp. Use TimeSeriesSplit "
                            "or sort by time before splitting."
                        ),
                        details={
                            "train_min": str(train_min),
                            "train_max": str(train_max),
                            "test_min": str(test_min),
                            "test_max": str(test_max),
                            "test_rows_overlapping_train_window": n_overlap,
                            "test_rows_strictly_before_train": n_before,
                        },
                    )
                )

    # Also warn if time_col appears as a feature without a clear semantic role.
    # (We don't drop it; downstream code may convert it. This is informational.)
    return findings


# ---------------------------------------------------------------------------
# TL012 -- group leakage
# ---------------------------------------------------------------------------


def _resolve_group_key(
    name_or_series: Any,
    X: pd.DataFrame,
    side: str,
) -> pd.Series | None:
    if name_or_series is None:
        return None
    if isinstance(name_or_series, str):
        if name_or_series not in X.columns:
            return None
        return X[name_or_series]
    g = pd.Series(name_or_series)
    if len(g) != len(X):
        raise ValueError(
            f"group_key length ({len(g)}) does not match {side} length ({len(X)})"
        )
    return g


def _check_group_leakage(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None,
    group_key_train: Any,
    group_key_test: Any,
) -> list[Finding]:
    """TL012 -- the same group ID appears in both train and test."""
    if group_key_train is None:
        return []
    g_train = _resolve_group_key(group_key_train, X_train, "X_train")
    if g_train is None:
        return [
            Finding(
                code="TL012",
                severity=Severity.WARNING,
                message="group_key column not found in X_train -- group leakage check skipped.",
                fix="Pass either a column name present in X_train or a Series of length len(X_train).",
            )
        ]

    findings: list[Finding] = []
    if X_test is not None:
        g_test = _resolve_group_key(group_key_test, X_test, "X_test")
        if g_test is None:
            return [
                Finding(
                    code="TL012",
                    severity=Severity.WARNING,
                    message="group_key column not found in X_test -- group leakage check skipped.",
                    fix="Pass the same group identifier for X_test as you did for X_train.",
                )
            ]
        train_groups = set(g_train.dropna().tolist())
        test_groups = set(g_test.dropna().tolist())
        overlap = train_groups & test_groups
        if overlap:
            n_test_rows_in_overlap = int(g_test.isin(overlap).sum())
            pct = 100 * n_test_rows_in_overlap / len(g_test) if len(g_test) else 0
            sev = Severity.CRITICAL if pct >= 5 else Severity.WARNING
            sample = sorted(str(x) for x in list(overlap)[:10])
            findings.append(
                Finding(
                    code="TL012",
                    severity=sev,
                    message=(
                        f"{len(overlap)} group IDs appear in both train and test "
                        f"({n_test_rows_in_overlap} test rows, {pct:.2f}% of test). "
                        f"Sample overlapping groups: {sample}."
                    ),
                    fix=(
                        "Group leakage: the same entity (user / patient / session) is "
                        "in train AND test. Cross-validation will overstate accuracy. "
                        "Use sklearn.model_selection.GroupKFold or split by the entity "
                        "ID, not by row."
                    ),
                    details={
                        "n_overlapping_groups": len(overlap),
                        "n_test_rows_in_overlap": n_test_rows_in_overlap,
                        "pct_of_test": round(pct, 4),
                    },
                )
            )

    # Sanity: very few distinct groups -> CV strategy is suspect.
    n_groups = g_train.nunique(dropna=True)
    if n_groups < 5:
        findings.append(
            Finding(
                code="TL012",
                severity=Severity.WARNING,
                message=f"Only {n_groups} distinct group(s) in X_train.",
                fix=(
                    "With this few groups, group-aware CV will have very high variance. "
                    "Either collect more groups or treat this as a small-data problem "
                    "with held-out groups as the test set."
                ),
            )
        )
    return findings


# ---------------------------------------------------------------------------
# HTML report renderer
# ---------------------------------------------------------------------------


def _html_escape(s: Any) -> str:
    text = str(s)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


_HTML_CSS = """
:root {
  --bg: #fafafa; --fg: #1a1a1a; --muted: #6b6b6b; --border: #e5e5e5;
  --critical: #b91c1c; --critical-bg: #fef2f2;
  --warning: #b45309; --warning-bg: #fffbeb;
  --info: #1d4ed8; --info-bg: #eff6ff;
  --good: #15803d; --good-bg: #f0fdf4;
}
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: var(--bg); color: var(--fg); margin: 0; padding: 32px;
  line-height: 1.5; max-width: 920px; margin: 0 auto; }
h1 { margin: 0 0 4px; font-size: 28px; }
h2 { margin: 32px 0 12px; font-size: 18px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }
h3 { margin: 0 0 4px; font-size: 15px; }
.meta { color: var(--muted); font-size: 13px; margin-bottom: 24px; }
.summary { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 16px 0 8px; }
.summary .card { background: white; border: 1px solid var(--border); border-radius: 8px;
  padding: 16px; text-align: center; }
.summary .card .num { font-size: 28px; font-weight: 600; }
.summary .card .lbl { font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }
.summary .critical .num { color: var(--critical); }
.summary .warning .num { color: var(--warning); }
.summary .info .num { color: var(--info); }
.banner { padding: 12px 16px; border-radius: 8px; margin: 16px 0;
  font-size: 14px; font-weight: 500; }
.banner.fail { background: var(--critical-bg); color: var(--critical); border: 1px solid #fecaca; }
.banner.pass { background: var(--good-bg); color: var(--good); border: 1px solid #bbf7d0; }
.metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.metrics .col { background: white; border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; }
.metrics .col h3 { font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
.metrics table { width: 100%; border-collapse: collapse; font-size: 14px; }
.metrics td { padding: 4px 0; }
.metrics td.k { color: var(--muted); }
.metrics td.v { text-align: right; font-variant-numeric: tabular-nums; font-weight: 500; }
.delta { font-size: 13px; color: var(--muted); margin-top: 8px; }
.delta b { color: var(--fg); }
.finding { background: white; border: 1px solid var(--border); border-left-width: 4px;
  border-radius: 6px; padding: 14px 16px; margin: 10px 0; }
.finding.critical { border-left-color: var(--critical); }
.finding.warning  { border-left-color: var(--warning); }
.finding.info     { border-left-color: var(--info); }
.finding .head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 6px; }
.finding .code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; }
.finding .badge { font-size: 11px; font-weight: 700; text-transform: uppercase;
  padding: 2px 8px; border-radius: 999px; letter-spacing: 0.05em; }
.finding.critical .badge { background: var(--critical-bg); color: var(--critical); }
.finding.warning  .badge { background: var(--warning-bg); color: var(--warning); }
.finding.info     .badge { background: var(--info-bg); color: var(--info); }
.finding .what, .finding .fix, .finding .cols { font-size: 14px; margin: 6px 0; }
.finding .label { color: var(--muted); font-size: 11px; text-transform: uppercase;
  letter-spacing: 0.05em; margin-right: 6px; }
.finding code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px; background: #f3f4f6; padding: 1px 5px; border-radius: 3px; }
.empty { color: var(--muted); font-style: italic; padding: 24px; text-align: center;
  background: white; border: 1px dashed var(--border); border-radius: 8px; }
footer { margin-top: 40px; color: var(--muted); font-size: 12px; text-align: center;
  border-top: 1px solid var(--border); padding-top: 12px; }
@media print { body { background: white; padding: 16px; } .finding, .summary .card,
  .metrics .col { break-inside: avoid; } }
"""


def _format_metric_value(v: Any) -> str:
    if isinstance(v, float):
        if 0 <= v <= 1:
            return f"{v:.4f}"
        return f"{v:,.4f}"
    if isinstance(v, int):
        return f"{v:,}"
    return _html_escape(v)


def _render_metrics_block(
    before: dict[str, Any] | None, after: dict[str, Any] | None
) -> str:
    if not before and not after:
        return ""
    keys: list[str] = []
    for d in (before or {}, after or {}):
        for k in d:
            if k not in keys:
                keys.append(k)

    def col(title: str, data: dict[str, Any] | None) -> str:
        if not data:
            rows = '<tr><td class="k">(not provided)</td><td class="v">--</td></tr>'
        else:
            rows = "".join(
                f'<tr><td class="k">{_html_escape(k)}</td>'
                f'<td class="v">{_format_metric_value(data.get(k, "--"))}</td></tr>'
                for k in keys
            )
        return (
            f'<div class="col"><h3>{_html_escape(title)}</h3>'
            f"<table>{rows}</table></div>"
        )

    delta_html = ""
    if before and after:
        deltas = []
        for k in keys:
            a, b = before.get(k), after.get(k)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                d = b - a
                sign = "+" if d > 0 else ""
                deltas.append(
                    f"<b>{_html_escape(k)}</b>: {_format_metric_value(a)} -> "
                    f"{_format_metric_value(b)} ({sign}{d:.4f})"
                )
        if deltas:
            delta_html = f'<div class="delta">{" &nbsp;|&nbsp; ".join(deltas)}</div>'

    return (
        '<h2>Performance: before vs after mldash</h2>'
        f'<div class="metrics">{col("Before (naive)", before)}'
        f'{col("After (mldash-cleaned)", after)}</div>'
        f"{delta_html}"
    )


def _render_finding(f: Finding) -> str:
    desc = _TL_DESCRIPTIONS.get(f.code, {})
    title = desc.get("title", f.code)
    what = desc.get("what", f.message)
    sev = f.severity.value
    cols_html = ""
    if f.columns:
        cols_html = (
            '<div class="cols"><span class="label">columns</span>'
            + " ".join(f"<code>{_html_escape(c)}</code>" for c in f.columns[:50])
            + "</div>"
        )
    return (
        f'<div class="finding {sev}">'
        f'<div class="head">'
        f'<h3><span class="code">{_html_escape(f.code)}</span> &middot; {_html_escape(title)}</h3>'
        f'<span class="badge">{sev}</span>'
        f"</div>"
        f'<div class="what"><span class="label">what</span>{_html_escape(what)}</div>'
        f'<div class="what"><span class="label">detail</span>{_html_escape(f.message)}</div>'
        f'<div class="fix"><span class="label">fix</span>{_html_escape(f.fix)}</div>'
        f"{cols_html}"
        f"</div>"
    )


def _render_html(
    report: "Report",
    *,
    title: str,
    dataset_name: str | None,
    metrics_before: dict[str, Any] | None,
    metrics_after: dict[str, Any] | None,
) -> str:
    n_c, n_w, n_i = len(report.critical), len(report.warnings), len(report.infos)
    banner = (
        '<div class="banner fail">FAIL -- critical issues must be resolved before training.</div>'
        if n_c > 0
        else '<div class="banner pass">PASS -- no critical issues. Review warnings below.</div>'
    )

    findings_html = (
        "".join(_render_finding(f) for f in report.sorted())
        if report.findings
        else '<div class="empty">No issues found.</div>'
    )

    meta_bits = []
    if dataset_name:
        meta_bits.append(f"dataset: <b>{_html_escape(dataset_name)}</b>")
    meta_bits.append(f"mldash v{__version__}")
    meta_html = " &middot; ".join(meta_bits)

    return (
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{_html_escape(title)}</title>"
        f"<style>{_HTML_CSS}</style></head><body>"
        f"<h1>{_html_escape(title)}</h1>"
        f"<div class='meta'>{meta_html}</div>"
        f"{banner}"
        f"<h2>Summary</h2>"
        f"<div class='summary'>"
        f"<div class='card critical'><div class='num'>{n_c}</div><div class='lbl'>critical</div></div>"
        f"<div class='card warning'><div class='num'>{n_w}</div><div class='lbl'>warning</div></div>"
        f"<div class='card info'><div class='num'>{n_i}</div><div class='lbl'>info</div></div>"
        f"</div>"
        f"{_render_metrics_block(metrics_before, metrics_after)}"
        f"<h2>Findings</h2>"
        f"{findings_html}"
        f"<footer>Generated by mldash. One call: <code>mldash.check(X_train, y_train, X_test, y_test)</code></footer>"
        f"</body></html>"
    )


# ---------------------------------------------------------------------------
# PDF report renderer (uses fpdf2 -- optional dependency)
# ---------------------------------------------------------------------------


def _pdf_safe(text: Any) -> str:
    """fpdf2's core fonts are latin-1; transliterate non-latin chars."""
    s = str(text)
    repl = {
        "->": "->", "-->": "-->", "—": "--", "–": "-",
        "→": "->", "≥": ">=", "≤": "<=", "·": "*",
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "…": "...",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    # Last resort: drop anything outside latin-1
    return s.encode("latin-1", "replace").decode("latin-1")


_PDF_RGB = {
    "fg":         (26, 26, 26),
    "muted":      (107, 107, 107),
    "border":     (220, 220, 220),
    "critical":   (185, 28, 28),
    "crit_bg":    (254, 242, 242),
    "warning":    (180, 83, 9),
    "warn_bg":    (255, 251, 235),
    "info":       (29, 78, 216),
    "info_bg":    (239, 246, 255),
    "good":       (21, 128, 61),
    "good_bg":    (240, 253, 244),
}


def _render_pdf(
    report: "Report",
    out_path,
    *,
    FPDF,
    title: str,
    dataset_name: str | None,
    metrics_before: dict[str, Any] | None,
    metrics_after: dict[str, Any] | None,
) -> None:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(left=15, top=15, right=15)
    page_w = pdf.w - pdf.l_margin - pdf.r_margin

    # ---- header
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*_PDF_RGB["fg"])
    pdf.cell(0, 9, _pdf_safe(title), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*_PDF_RGB["muted"])
    meta = []
    if dataset_name:
        meta.append(f"dataset: {dataset_name}")
    meta.append(f"mldash v{__version__}")
    pdf.cell(0, 5, _pdf_safe(" | ".join(meta)), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # ---- pass/fail banner
    n_c = len(report.critical)
    n_w = len(report.warnings)
    n_i = len(report.infos)
    if n_c > 0:
        bg, fg = _PDF_RGB["crit_bg"], _PDF_RGB["critical"]
        msg = "FAIL -- critical issues must be resolved before training."
    else:
        bg, fg = _PDF_RGB["good_bg"], _PDF_RGB["good"]
        msg = "PASS -- no critical issues. Review warnings below."
    pdf.set_fill_color(*bg)
    pdf.set_text_color(*fg)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 8, _pdf_safe("  " + msg), fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ---- summary cards (3 columns)
    pdf.set_text_color(*_PDF_RGB["fg"])
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    card_w = (page_w - 4) / 3  # 2mm gaps between 3 cards
    card_h = 18
    cards = [("CRITICAL", n_c, _PDF_RGB["critical"]),
             ("WARNING",  n_w, _PDF_RGB["warning"]),
             ("INFO",     n_i, _PDF_RGB["info"])]
    y0 = pdf.get_y()
    for i, (lbl, num, color) in enumerate(cards):
        x = pdf.l_margin + i * (card_w + 2)
        pdf.set_xy(x, y0)
        pdf.set_draw_color(*_PDF_RGB["border"])
        pdf.set_line_width(0.2)
        pdf.rect(x, y0, card_w, card_h)
        pdf.set_xy(x, y0 + 2)
        pdf.set_text_color(*color)
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(card_w, 8, str(num), align="C", new_x="LEFT", new_y="NEXT")
        pdf.set_xy(x, y0 + 11)
        pdf.set_text_color(*_PDF_RGB["muted"])
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(card_w, 4, lbl, align="C", new_x="LEFT", new_y="NEXT")
    pdf.set_y(y0 + card_h + 4)

    # ---- metrics block
    if metrics_before or metrics_after:
        pdf.set_text_color(*_PDF_RGB["fg"])
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, "Performance: before vs after mldash", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

        keys: list[str] = []
        for d in (metrics_before or {}, metrics_after or {}):
            for k in d:
                if k not in keys:
                    keys.append(k)

        col_w = (page_w - 2) / 2
        y0 = pdf.get_y()
        # Determine the max number of rows to align both columns
        for ci, (heading, data) in enumerate(
            [("Before (naive)", metrics_before), ("After (mldash-cleaned)", metrics_after)]
        ):
            x = pdf.l_margin + ci * (col_w + 2)
            pdf.set_xy(x, y0)
            pdf.set_draw_color(*_PDF_RGB["border"])
            pdf.rect(x, y0, col_w, 8 + 5 * max(1, len(keys)))
            pdf.set_xy(x + 3, y0 + 2)
            pdf.set_text_color(*_PDF_RGB["muted"])
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(col_w - 6, 4, _pdf_safe(heading.upper()), new_x="LEFT", new_y="NEXT")
            pdf.set_text_color(*_PDF_RGB["fg"])
            pdf.set_font("Helvetica", "", 9)
            for ri, k in enumerate(keys):
                pdf.set_xy(x + 3, y0 + 7 + ri * 5)
                pdf.cell(col_w * 0.55 - 3, 5, _pdf_safe(k))
                v = (data or {}).get(k, "--")
                if isinstance(v, float):
                    v_str = f"{v:.4f}" if 0 <= v <= 1 else f"{v:,.4f}"
                elif isinstance(v, int):
                    v_str = f"{v:,}"
                else:
                    v_str = str(v)
                pdf.set_xy(x + col_w * 0.55, y0 + 7 + ri * 5)
                pdf.cell(col_w * 0.45 - 3, 5, _pdf_safe(v_str), align="R")
        pdf.set_y(y0 + 8 + 5 * max(1, len(keys)) + 2)

        # delta line
        if metrics_before and metrics_after:
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*_PDF_RGB["muted"])
            for k in keys:
                a, b = metrics_before.get(k), metrics_after.get(k)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    d = b - a
                    sign = "+" if d > 0 else ""
                    pdf.cell(0, 4, _pdf_safe(
                        f"  {k}: {a:.4f} -> {b:.4f} ({sign}{d:.4f})"
                    ), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

    # ---- findings
    pdf.set_text_color(*_PDF_RGB["fg"])
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Findings", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    if not report.findings:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*_PDF_RGB["muted"])
        pdf.cell(0, 6, "No issues found.", new_x="LMARGIN", new_y="NEXT")
    else:
        for f in report.sorted():
            sev_color = {
                Severity.CRITICAL: _PDF_RGB["critical"],
                Severity.WARNING:  _PDF_RGB["warning"],
                Severity.INFO:     _PDF_RGB["info"],
            }[f.severity]
            sev_bg = {
                Severity.CRITICAL: _PDF_RGB["crit_bg"],
                Severity.WARNING:  _PDF_RGB["warn_bg"],
                Severity.INFO:     _PDF_RGB["info_bg"],
            }[f.severity]
            desc = _TL_DESCRIPTIONS.get(f.code, {})
            heading = f"{f.code} - {desc.get('title', f.code)}"

            # left accent bar + title row
            y_start = pdf.get_y()
            pdf.set_fill_color(*sev_color)
            pdf.rect(pdf.l_margin, y_start, 1.2, 6, style="F")

            pdf.set_xy(pdf.l_margin + 3, y_start)
            pdf.set_text_color(*_PDF_RGB["fg"])
            pdf.set_font("Helvetica", "B", 10)
            title_w = page_w - 28
            pdf.cell(title_w, 6, _pdf_safe(heading))

            # severity badge
            pdf.set_xy(pdf.l_margin + 3 + title_w, y_start)
            pdf.set_fill_color(*sev_bg)
            pdf.set_text_color(*sev_color)
            pdf.set_font("Helvetica", "B", 7)
            pdf.cell(25, 6, _pdf_safe(f.severity.value.upper()),
                     align="C", fill=True, new_x="LMARGIN", new_y="NEXT")

            # body
            pdf.set_text_color(*_PDF_RGB["fg"])
            pdf.set_font("Helvetica", "", 9)
            pdf.set_x(pdf.l_margin + 3)
            pdf.multi_cell(page_w - 3, 4.5, _pdf_safe(f"What:   {desc.get('what', f.message)}"))
            pdf.set_x(pdf.l_margin + 3)
            pdf.multi_cell(page_w - 3, 4.5, _pdf_safe(f"Detail: {f.message}"))
            pdf.set_x(pdf.l_margin + 3)
            pdf.set_text_color(*_PDF_RGB["muted"])
            pdf.multi_cell(page_w - 3, 4.5, _pdf_safe(f"Fix:    {f.fix}"))
            if f.columns:
                pdf.set_x(pdf.l_margin + 3)
                pdf.multi_cell(page_w - 3, 4.5,
                               _pdf_safe(f"Cols:   {', '.join(f.columns[:30])}"))
            pdf.ln(3)

    # ---- footer
    pdf.set_y(-18)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(*_PDF_RGB["muted"])
    pdf.cell(0, 4, _pdf_safe(
        "Generated by mldash. One call: mldash.check(X_train, y_train, X_test, y_test)"
    ), align="C")

    pdf.output(str(out_path))


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def check(
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    *,
    task: str = "auto",
    time_col: str | None = None,
    group_key: Any = None,
    group_key_test: Any = None,
) -> Report:
    """Run all checks and return a :class:`Report`.

    Parameters
    ----------
    X_train, X_test : DataFrame | ndarray | dict | list
        Feature matrices. ``X_test`` is optional but unlocks contamination,
        drift, and schema checks when provided.
    y_train, y_test : Series | ndarray | list
        Labels. ``y_train`` is required; ``y_test`` is currently unused but
        accepted for API symmetry.
    task : ``"auto"`` | ``"classification"`` | ``"regression"``
        Inferred from ``y_train`` when ``"auto"``.
    time_col : str | None
        Column name (in both X_train and X_test) holding row timestamps.
        Enables TL011 -- temporal leakage detection. The column is also
        suppressed from TL010 (ID-like) since timestamps are expected to
        be near-unique.
    group_key : str | Series | array-like | None
        Either a column name in X_train, or a Series/array of length
        ``len(X_train)`` holding group identifiers (user_id, patient_id,
        session_id...). Enables TL012 -- group leakage detection.
        Suppressed from TL010.
    group_key_test : str | Series | array-like | None
        Group identifiers for X_test. Defaults to the same column name as
        ``group_key`` when ``group_key`` is a string.

    Returns
    -------
    Report
        ``report.ok()`` is ``True`` iff there are no critical findings.
    """
    Xtr = _to_dataframe(X_train, "X_train")
    ytr = _to_series(y_train, "y_train")
    if len(Xtr) != len(ytr):
        raise ValueError(
            f"X_train and y_train length mismatch: {len(Xtr)} vs {len(ytr)}"
        )

    Xte: pd.DataFrame | None = None
    if X_test is not None:
        Xte = _to_dataframe(X_test, "X_test")
        if y_test is not None:
            yte = _to_series(y_test, "y_test")
            if len(Xte) != len(yte):
                raise ValueError(
                    f"X_test and y_test length mismatch: {len(Xte)} vs {len(yte)}"
                )

    resolved_task = _infer_task(ytr) if task == "auto" else task
    if resolved_task not in ("classification", "regression"):
        raise ValueError(
            f"task must be 'auto', 'classification', or 'regression' -- got {task!r}"
        )

    # Default group_key_test to same column name when group_key is a string
    if group_key_test is None and isinstance(group_key, str):
        group_key_test = group_key

    # Suppress time_col / group_key columns from feature-only checks where
    # they would produce noisy false positives.
    suppress_cols: set[str] = set()
    if isinstance(time_col, str):
        suppress_cols.add(time_col)
    if isinstance(group_key, str):
        suppress_cols.add(group_key)

    Xtr_features = Xtr.drop(columns=[c for c in suppress_cols if c in Xtr.columns])

    report = Report()

    report.extend(_check_schema_mismatch(Xtr, Xte))
    report.extend(_check_constant_features(Xtr_features))
    report.extend(_check_duplicate_columns(Xtr_features))
    report.extend(_check_id_like_features(Xtr_features))
    report.extend(_check_class_imbalance(ytr, resolved_task))
    report.extend(_check_target_leakage(Xtr_features, ytr, resolved_task))

    if Xte is not None:
        report.extend(_check_exact_duplicates(Xtr, Xte))
        report.extend(_check_near_duplicates(Xtr, Xte))
        report.extend(_check_train_test_drift(Xtr, Xte))
        report.extend(_check_missingness_mismatch(Xtr, Xte))

    report.extend(_check_temporal_leakage(Xtr, Xte, time_col))
    report.extend(_check_group_leakage(Xtr, Xte, group_key, group_key_test))

    return report


# ---------------------------------------------------------------------------
# audit_pipeline -- detect preprocessing leakage
# ---------------------------------------------------------------------------


_TARGET_AWARE_TRANSFORMER_NAMES = {
    "TargetEncoder",
    "CatBoostEncoder",
    "LeaveOneOutEncoder",
    "MEstimateEncoder",
    "JamesSteinEncoder",
    "WOEEncoder",
}


def _walk_transformers(estimator: Any) -> list[Any]:
    """Yield every estimator inside a Pipeline / ColumnTransformer tree."""
    out: list[Any] = []
    seen: set[int] = set()

    def visit(obj: Any) -> None:
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        out.append(obj)
        # sklearn Pipeline.steps == [(name, estimator), ...]
        steps = getattr(obj, "steps", None)
        if steps:
            for _, step in steps:
                visit(step)
        # ColumnTransformer.transformers
        trs = getattr(obj, "transformers", None)
        if trs:
            for tup in trs:
                if len(tup) >= 2:
                    visit(tup[1])
        # FeatureUnion.transformer_list
        tl = getattr(obj, "transformer_list", None)
        if tl:
            for _, t in tl:
                visit(t)

    visit(estimator)
    return out


def audit_pipeline(
    pipeline: Any,
    X,
    y,
    *,
    task: str = "auto",
    test_size: float = 0.30,
    random_state: int = 42,
    atol: float = 1e-6,
) -> Report:
    """Detect preprocessing leakage in an unfitted sklearn pipeline.

    Strategy: clone the pipeline twice, fit one on ``X_train`` only and the
    other on the full ``X``. Transform the same ``X_test`` with each and
    compare. If outputs diverge beyond ``atol``, the pipeline has data-
    dependent state (scaler stats, imputer means, encoder maps) that would
    leak test information into training when fit on the full dataset --
    a textbook case of "fit on full data, then split".

    Also flags the *presence* of target-aware encoders, which are leakage-
    prone unless wrapped in proper cross-validation.

    Parameters
    ----------
    pipeline : sklearn estimator (typically a Pipeline)
        Should be UNFITTED. A fitted estimator will be cloned to drop state.
    X, y : DataFrame / Series (or array-like)
        Raw, un-split data. ``audit_pipeline`` does the split internally.
    task : ``"auto"`` | ``"classification"`` | ``"regression"``
    test_size, random_state : passed to train_test_split
    atol : float
        Absolute tolerance for "transforms agree". 1e-6 catches even small
        scaler/imputer divergence; raise it if you have stochastic steps.

    Returns
    -------
    Report
        ``report.ok()`` is True iff the pipeline does not silently leak.
    """
    try:
        from sklearn.base import clone
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        raise ImportError(
            "audit_pipeline requires scikit-learn. Install with `pip install scikit-learn`."
        ) from e

    Xdf = _to_dataframe(X, "X")
    ys = _to_series(y, "y")
    if len(Xdf) != len(ys):
        raise ValueError(f"X and y length mismatch: {len(Xdf)} vs {len(ys)}")

    resolved_task = _infer_task(ys) if task == "auto" else task
    stratify = ys if resolved_task == "classification" and ys.nunique() > 1 else None
    X_train, X_test, y_train, _ = train_test_split(
        Xdf, ys, test_size=test_size, random_state=random_state, stratify=stratify
    )

    report = Report()

    # --- TL014: target-aware encoders inside the pipeline -----------------
    target_aware: list[str] = []
    for est in _walk_transformers(pipeline):
        cls_name = type(est).__name__
        if cls_name in _TARGET_AWARE_TRANSFORMER_NAMES:
            target_aware.append(cls_name)
    if target_aware:
        report.add(
            Finding(
                code="TL014",
                severity=Severity.WARNING,
                message=(
                    f"Pipeline contains target-aware transformer(s): {target_aware}. "
                    "These leak label information unless wrapped in cross-validated fitting."
                ),
                fix=(
                    "Wrap target encoders in a cross-validation loop "
                    "(e.g. category_encoders.wrapper.NestedCVWrapper or sklearn's "
                    "TargetEncoder with CV folds). Never call fit() on the full dataset "
                    "before splitting."
                ),
            )
        )

    # --- TL013: fit-on-full-data leakage ---------------------------------
    pipe_train = clone(pipeline)
    pipe_full = clone(pipeline)

    try:
        pipe_train.fit(X_train, y_train)
        pipe_full.fit(Xdf, ys)
    except Exception as e:
        report.add(
            Finding(
                code="TL013",
                severity=Severity.WARNING,
                message=f"audit_pipeline could not fit the pipeline ({type(e).__name__}: {e}).",
                fix=(
                    "audit_pipeline expects an unfitted pipeline that accepts "
                    "(X, y) and exposes .transform() or .predict(). Ensure all "
                    "steps are configured correctly."
                ),
            )
        )
        return report

    transform_method = "transform" if hasattr(pipe_train, "transform") else (
        "predict_proba" if hasattr(pipe_train, "predict_proba") else "predict"
    )

    try:
        out_train = getattr(pipe_train, transform_method)(X_test)
        out_full = getattr(pipe_full, transform_method)(X_test)
    except Exception as e:
        report.add(
            Finding(
                code="TL013",
                severity=Severity.WARNING,
                message=f"audit_pipeline could not transform X_test ({type(e).__name__}: {e}).",
                fix="Ensure the pipeline produces stable transform() / predict() output for X_test.",
            )
        )
        return report

    a = np.asarray(out_train)
    b = np.asarray(out_full)
    if a.shape != b.shape:
        report.add(
            Finding(
                code="TL013",
                severity=Severity.CRITICAL,
                message=(
                    f"Pipeline output shape changed when fit on full vs train data "
                    f"({a.shape} vs {b.shape}) -- the pipeline depends on the row "
                    "count of fitted data, which is a leakage smell."
                ),
                fix=(
                    "Investigate steps whose output size depends on data "
                    "(e.g. variance-based feature selection, OneHotEncoder with "
                    "handle_unknown='error'). Refit on training data only."
                ),
            )
        )
        return report

    try:
        a_num = a.astype(float)
        b_num = b.astype(float)
        max_abs_diff = float(np.nanmax(np.abs(a_num - b_num)))
    except (TypeError, ValueError):
        max_abs_diff = 0.0 if np.array_equal(a, b) else float("inf")

    if max_abs_diff > atol:
        report.add(
            Finding(
                code="TL013",
                severity=Severity.CRITICAL,
                message=(
                    f"Pipeline output for X_test differs by {max_abs_diff:.6g} "
                    "between (fit on train) and (fit on full data). The pipeline "
                    "has data-dependent state -- if fit on the full dataset, test "
                    "information leaks into the model."
                ),
                fix=(
                    "Always fit preprocessing on training data only, then "
                    "transform(X_test) -- never .fit_transform(X) before splitting. "
                    "If using sklearn, wrap preprocessing inside a Pipeline and call "
                    ".fit() on (X_train, y_train) only."
                ),
                details={
                    "max_abs_diff": max_abs_diff,
                    "atol": atol,
                    "test_method": transform_method,
                },
            )
        )

    return report
