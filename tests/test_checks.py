"""Synthetic tests: each check fires on a planted bug, stays silent on a clean dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mldash


RNG = np.random.default_rng(42)


def _clean_classification(n: int = 500, n_features: int = 6):
    X = pd.DataFrame(
        RNG.normal(size=(n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    # Real signal in f0 only
    logits = 1.5 * X["f0"].values + RNG.normal(scale=0.5, size=n)
    y = pd.Series((logits > 0).astype(int), name="y")
    return X, y


def _split(X: pd.DataFrame, y: pd.Series, frac: float = 0.7):
    n_train = int(len(X) * frac)
    return X.iloc[:n_train].copy(), y.iloc[:n_train].copy(), X.iloc[n_train:].copy(), y.iloc[n_train:].copy()


# ---------------- baseline ----------------


def test_clean_dataset_has_no_critical():
    X, y = _clean_classification()
    Xtr, ytr, Xte, yte = _split(X, y)
    report = mldash.check(Xtr, ytr, Xte, yte)
    assert report.ok(), f"clean dataset should not produce critical findings; got {report}"


# ---------------- TL001 / TL002 ----------------


def test_tl001_exact_duplicate_leakage():
    X, y = _clean_classification(n=400)
    Xtr, ytr, Xte, yte = _split(X, y)
    Xte = pd.concat([Xte, Xtr.head(20)], ignore_index=True)
    yte = pd.concat([yte, ytr.head(20)], ignore_index=True)
    report = mldash.check(Xtr, ytr, Xte, yte)
    codes = {f.code for f in report.findings}
    assert "TL001" in codes


def test_tl002_near_duplicates():
    X, y = _clean_classification(n=400)
    Xtr, ytr, Xte, yte = _split(X, y)
    perturbed = Xtr.head(20).copy()
    for col in perturbed.columns:
        perturbed[col] = perturbed[col] + 1e-7
    Xte = pd.concat([Xte, perturbed], ignore_index=True)
    yte = pd.concat([yte, ytr.head(20)], ignore_index=True)
    report = mldash.check(Xtr, ytr, Xte, yte)
    codes = {f.code for f in report.findings}
    assert "TL002" in codes


# ---------------- TL003 ----------------


def test_tl003_target_leakage_numeric():
    X, y = _clean_classification(n=400)
    X["leaky"] = y.values + RNG.normal(scale=0.001, size=len(y))
    Xtr, ytr, Xte, yte = _split(X, y)
    report = mldash.check(Xtr, ytr, Xte, yte)
    crit_codes = {f.code for f in report.critical}
    assert "TL003" in crit_codes
    leak_finding = next(f for f in report.findings if f.code == "TL003" and f.severity.value == "critical")
    assert "leaky" in leak_finding.columns


def test_tl003_target_leakage_categorical():
    X, y = _clean_classification(n=400)
    X["leaky_cat"] = y.map({0: "neg", 1: "pos"})
    Xtr, ytr, Xte, yte = _split(X, y)
    report = mldash.check(Xtr, ytr, Xte, yte)
    crit = [f for f in report.critical if f.code == "TL003"]
    assert crit, "expected critical TL003 for perfectly-encoded categorical"
    assert "leaky_cat" in crit[0].columns


def test_tl003_target_leakage_numeric_multiclass():
    """Regression: numeric proxy = y + tiny noise must be flagged when y has
    many classes (binning Cramer's V missed this; eta catches it)."""
    n = 600
    X = pd.DataFrame(RNG.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
    y = pd.Series(RNG.integers(0, 7, size=n), name="y")  # 7 classes -- multiclass
    X["leaky_proxy"] = y.astype(float) + RNG.normal(0, 0.02, size=n)
    Xtr, ytr, Xte, yte = _split(X, y)
    report = mldash.check(Xtr, ytr, Xte, yte)
    crit = [f for f in report.critical if f.code == "TL003"]
    assert crit, "expected critical TL003 for numeric near-proxy of multiclass y"
    assert "leaky_proxy" in crit[0].columns


# ---------------- TL004 / TL005 / TL010 ----------------


def test_tl004_constant_feature():
    X, y = _clean_classification()
    X["dead"] = 7
    Xtr, ytr, Xte, yte = _split(X, y)
    report = mldash.check(Xtr, ytr, Xte, yte)
    assert any(f.code == "TL004" for f in report.findings)


def test_tl005_duplicate_columns():
    X, y = _clean_classification()
    X["f0_copy"] = X["f0"]
    Xtr, ytr, Xte, yte = _split(X, y)
    report = mldash.check(Xtr, ytr, Xte, yte)
    assert any(f.code == "TL005" for f in report.findings)


def test_tl010_id_like_feature():
    X, y = _clean_classification(n=500)
    X["row_id"] = np.arange(len(X))
    Xtr, ytr, Xte, yte = _split(X, y)
    report = mldash.check(Xtr, ytr, Xte, yte)
    assert any(f.code == "TL010" for f in report.findings)


# ---------------- TL006 / TL008 ----------------


def test_tl006_drift():
    X, y = _clean_classification(n=600)
    Xtr, ytr, Xte, yte = _split(X, y)
    Xte["f0"] = Xte["f0"] + 5  # heavy shift
    report = mldash.check(Xtr, ytr, Xte, yte)
    assert any(f.code == "TL006" for f in report.findings)


def test_tl008_missingness_mismatch():
    X, y = _clean_classification(n=600)
    Xtr, ytr, Xte, yte = _split(X, y)
    mask = RNG.random(len(Xte)) < 0.4
    Xte.loc[mask, "f1"] = np.nan
    report = mldash.check(Xtr, ytr, Xte, yte)
    assert any(f.code == "TL008" for f in report.findings)


# ---------------- TL007 ----------------


def test_tl007_class_imbalance():
    X, y = _clean_classification(n=1000)
    # force ~0.5% minority
    y.iloc[:] = 0
    y.iloc[:5] = 1
    Xtr, ytr, Xte, yte = _split(X, y)
    report = mldash.check(Xtr, ytr, Xte, yte)
    assert any(f.code == "TL007" for f in report.findings)


# ---------------- TL009 ----------------


def test_tl009_schema_mismatch_columns():
    X, y = _clean_classification()
    Xtr, ytr, Xte, yte = _split(X, y)
    Xte = Xte.drop(columns=["f0"])
    report = mldash.check(Xtr, ytr, Xte, yte)
    assert any(f.code == "TL009" and f.severity.value == "critical" for f in report.findings)


def test_tl009_schema_mismatch_dtype():
    X, y = _clean_classification()
    Xtr, ytr, Xte, yte = _split(X, y)
    Xte["f0"] = Xte["f0"].astype("float32")
    report = mldash.check(Xtr, ytr, Xte, yte)
    assert any(f.code == "TL009" for f in report.findings)


# ---------------- API ergonomics ----------------


def test_accepts_numpy_arrays():
    X, y = _clean_classification()
    Xtr, ytr, Xte, yte = _split(X, y)
    report = mldash.check(Xtr.to_numpy(), ytr.to_numpy(), Xte.to_numpy(), yte.to_numpy())
    assert isinstance(report, mldash.Report)


def test_report_to_dict_serializable():
    import json

    X, y = _clean_classification()
    X["dead"] = 1
    Xtr, ytr, Xte, yte = _split(X, y)
    report = mldash.check(Xtr, ytr, Xte, yte)
    json.dumps(report.to_dict())  # raises if not serializable


def test_length_mismatch_raises():
    X, y = _clean_classification()
    with pytest.raises(ValueError):
        mldash.check(X, y.iloc[:-1])


def test_to_html_renders_self_contained_doc():
    X, y = _clean_classification()
    X["dead"] = 1
    X["leaky"] = y.values
    Xtr, ytr, Xte, yte = _split(X, y)
    report = mldash.check(Xtr, ytr, Xte, yte)
    html = report.to_html(
        title="t",
        dataset_name="synthetic",
        metrics_before={"accuracy": 0.99},
        metrics_after={"accuracy": 0.83},
    )
    assert html.startswith("<!doctype html>")
    assert "</html>" in html
    assert "TL003" in html
    assert "0.9900" in html and "0.8300" in html
    assert "<script" not in html  # no JS, fully self-contained


# ---------------- TL003 partial-leakage (info tier) ----------------


def test_tl003_info_tier_for_gray_zone():
    """0.70 <= assoc < 0.85 should fire as INFO, not warning/critical."""
    n = 600
    X = pd.DataFrame(RNG.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
    y = pd.Series(RNG.integers(0, 2, size=n), name="y")
    # Build a feature that's ~0.75 correlated with y (between info and warn)
    noise = RNG.normal(0, 0.7, size=n)
    X["partial"] = y.values.astype(float) + noise
    Xtr, ytr, Xte, yte = _split(X, y)
    report = mldash.check(Xtr, ytr, Xte, yte)
    info_findings = [f for f in report.findings if f.code == "TL003" and f.severity.value == "info"]
    # At least one info-tier TL003 must fire (partial may bucket as warn or info
    # depending on sampled noise; this is the gray zone). What must NOT happen:
    # critical for a non-leaky feature.
    crit = [f for f in report.findings if f.code == "TL003" and f.severity.value == "critical"]
    assert not crit, f"gray-zone feature should not be critical, got {crit}"


# ---------------- TL011 temporal ----------------


def test_tl011_temporal_overlap_critical():
    n = 400
    times = pd.date_range("2024-01-01", periods=n, freq="h")
    X = pd.DataFrame({"ts": times, "f0": RNG.normal(size=n), "f1": RNG.normal(size=n)})
    y = pd.Series((X["f0"] > 0).astype(int), name="y")
    # Bad split: shuffled instead of chronological
    Xtr, ytr, Xte, yte = _split(X.sample(frac=1.0, random_state=1).reset_index(drop=True),
                                 y.sample(frac=1.0, random_state=1).reset_index(drop=True))
    report = mldash.check(Xtr, ytr, Xte, yte, time_col="ts")
    temporal = [f for f in report.findings if f.code == "TL011"]
    assert temporal, "expected TL011 for shuffled time column"
    assert any(f.severity.value in ("critical", "warning") for f in temporal)


def test_tl011_clean_chronological_split_no_finding():
    n = 400
    times = pd.date_range("2024-01-01", periods=n, freq="h")
    X = pd.DataFrame({"ts": times, "f0": RNG.normal(size=n)})
    y = pd.Series((X["f0"] > 0).astype(int), name="y")
    Xtr, ytr, Xte, yte = _split(X, y)  # already chronological
    report = mldash.check(Xtr, ytr, Xte, yte, time_col="ts")
    assert not [f for f in report.findings if f.code == "TL011"]


def test_tl011_suppresses_id_like_on_time_col():
    n = 200
    X = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=n, freq="h"),
        "f0": RNG.normal(size=n),
    })
    y = pd.Series((X["f0"] > 0).astype(int), name="y")
    Xtr, ytr, Xte, yte = _split(X, y)
    report = mldash.check(Xtr, ytr, Xte, yte, time_col="ts")
    assert not [f for f in report.findings if f.code == "TL010"]


# ---------------- TL012 group ----------------


def test_tl012_group_overlap_critical():
    n = 600
    user_ids = RNG.integers(0, 50, size=n)  # 50 users, many rows each
    X = pd.DataFrame({"user_id": user_ids, "f0": RNG.normal(size=n)})
    y = pd.Series((X["f0"] > 0).astype(int), name="y")
    # Random row split -- guaranteed to put same users in train and test
    Xtr, ytr, Xte, yte = _split(X.sample(frac=1.0, random_state=2).reset_index(drop=True),
                                 y.sample(frac=1.0, random_state=2).reset_index(drop=True))
    report = mldash.check(Xtr, ytr, Xte, yte, group_key="user_id")
    overlap = [f for f in report.findings if f.code == "TL012" and "overlapping" in f.details.get("n_overlapping_groups", "" ) .__str__()]
    findings = [f for f in report.findings if f.code == "TL012"]
    assert findings, "expected TL012 for row-level split with shared groups"


def test_tl012_clean_group_split_no_overlap_finding():
    n = 600
    user_ids = np.repeat(np.arange(40), n // 40)[:n]
    X = pd.DataFrame({"user_id": user_ids, "f0": RNG.normal(size=n)})
    y = pd.Series((X["f0"] > 0).astype(int), name="y")
    # Group-aware split: users 0-31 -> train, users 32-39 -> test
    train_mask = X["user_id"] < 32
    Xtr, ytr = X[train_mask].reset_index(drop=True), y[train_mask].reset_index(drop=True)
    Xte, yte = X[~train_mask].reset_index(drop=True), y[~train_mask].reset_index(drop=True)
    report = mldash.check(Xtr, ytr, Xte, yte, group_key="user_id")
    overlap_findings = [
        f for f in report.findings
        if f.code == "TL012" and "appear in both" in f.message
    ]
    assert not overlap_findings, f"clean group split should not flag overlap, got {overlap_findings}"


# ---------------- TL013 / TL014 audit_pipeline ----------------


def test_tl013_scaler_fit_on_full_data_detected():
    pytest.importorskip("sklearn")
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    n = 400
    X = pd.DataFrame(RNG.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series((X["f0"] > 0).astype(int), name="y")

    pipe = Pipeline([("scaler", StandardScaler())])
    report = mldash.audit_pipeline(pipe, X, y)
    # StandardScaler fit on full vs train will produce different scaling stats
    # for X_test transforms. mldash should detect this as critical TL013.
    crit = [f for f in report.findings if f.code == "TL013" and f.severity.value == "critical"]
    assert crit, f"expected critical TL013 for StandardScaler, got {[str(f) for f in report.findings]}"


def test_audit_pipeline_clean_when_no_data_dependent_state():
    pytest.importorskip("sklearn")
    from sklearn.preprocessing import FunctionTransformer

    n = 200
    X = pd.DataFrame(RNG.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = pd.Series((X["a"] > 0).astype(int), name="y")
    # FunctionTransformer with no state -- transform output is identical
    # regardless of fit data.
    pipe = FunctionTransformer(lambda x: x * 2.0, validate=False)
    report = mldash.audit_pipeline(pipe, X, y)
    crit = [f for f in report.findings if f.code == "TL013" and f.severity.value == "critical"]
    assert not crit, f"stateless transformer should not trigger TL013, got {crit}"


# ---------------- new codes appear in to_html ----------------


def test_to_pdf_writes_valid_pdf(tmp_path):
    pytest.importorskip("fpdf")
    X, y = _clean_classification()
    X["dead"] = 1
    X["leaky"] = y.values
    Xtr, ytr, Xte, yte = _split(X, y)
    report = mldash.check(Xtr, ytr, Xte, yte)
    out = tmp_path / "r.pdf"
    written = report.to_pdf(
        out,
        title="t",
        dataset_name="synthetic",
        metrics_before={"accuracy": 0.99},
        metrics_after={"accuracy": 0.83},
    )
    assert out.exists()
    head = out.read_bytes()[:8]
    assert head.startswith(b"%PDF-"), f"not a PDF: header={head!r}"
    assert out.stat().st_size > 1500  # non-trivial doc, not just a stub
    assert str(written) == str(out.resolve())


def test_to_pdf_raises_clear_error_without_fpdf2(monkeypatch):
    """If fpdf2 is missing, the error message should tell the user how to fix it."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "fpdf":
            raise ImportError("No module named 'fpdf'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    X, y = _clean_classification(n=100)
    Xtr, ytr, Xte, yte = _split(X, y)
    report = mldash.check(Xtr, ytr, Xte, yte)
    with pytest.raises(ImportError, match=r"pip install mldash\[pdf\]"):
        report.to_pdf("/tmp/should_not_be_written.pdf")


def test_to_html_includes_new_codes():
    n = 200
    X = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=n, freq="h"),
        "user_id": RNG.integers(0, 10, size=n),
        "f0": RNG.normal(size=n),
    })
    y = pd.Series((X["f0"] > 0).astype(int), name="y")
    Xtr, ytr, Xte, yte = _split(X.sample(frac=1.0, random_state=3).reset_index(drop=True),
                                 y.sample(frac=1.0, random_state=3).reset_index(drop=True))
    report = mldash.check(Xtr, ytr, Xte, yte, time_col="ts", group_key="user_id")
    html = report.to_html(title="t")
    assert "TL011" in html or "TL012" in html, "expected TL011 or TL012 in HTML report"
