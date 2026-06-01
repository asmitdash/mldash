"""dash_mlguard v0.2 demo: time + group + preprocessing leakage on a real-shape problem.

Synthetic but realistic: a fraud-detection-style stream of transactions.
Each transaction belongs to a `user_id` and has a `timestamp`. The
"correct" model is trained on past data and evaluated on strictly future
data, with no user appearing in both train and test.

We intentionally make THREE common mistakes in the naive pipeline:

    1. SHUFFLE-SPLIT on time      -> TL011 (temporal leakage)
    2. ROW-LEVEL split, not group -> TL012 (group leakage)
    3. StandardScaler.fit(X)
       BEFORE the split            -> TL013 (preprocessing leakage)

Then we re-run with dash_mlguard gating each stage. The dash_mlguard-gated pipeline
refuses to ship the model and tells the developer exactly what to fix.

Run inside the bundled venv:
    .venv\\Scripts\\python demo_v2.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import dash_mlguard


HERE = Path(__file__).resolve().parent
REPORT_HTML = HERE / "sample_report.html"
REPORT_PDF  = HERE / "sample_report.pdf"

print(f"dash_mlguard version: {dash_mlguard.__version__}\n")


# ---------------------------------------------------------------------------
# 1. Build a temporal, grouped dataset
# ---------------------------------------------------------------------------

rng = np.random.default_rng(7)
N = 8_000
N_USERS = 600

# Each user has a ~10-day "active window" centered on a per-user date in the
# 90-day span. This means most users are active in either the first or second
# half but not both -- so a chronological split will produce a clean test set
# with users that didn't appear in train.
user_centers = rng.uniform(5, 85, size=N_USERS)
user_ids = rng.integers(0, N_USERS, size=N)
days_offset = user_centers[user_ids] + rng.normal(0, 5, size=N)
days_offset = np.clip(days_offset, 0, 90)
ts = pd.Timestamp("2024-01-01") + pd.to_timedelta(days_offset, unit="D")

# Real signal: amount + a per-user offset + time-of-week effect
amount    = rng.gamma(2.0, 30, N)
hour      = ts.hour
day_of_wk = ts.dayofweek
user_bias = pd.Series(rng.normal(0, 0.7, size=N_USERS))

logit = (
    -2.0
    + 0.025 * amount
    + 0.10 * (hour < 6)
    + 0.30 * (day_of_wk >= 5)
    + user_bias.values[user_ids]
    + rng.normal(0, 0.5, size=N)
)
is_fraud = (logit > 0.5).astype(int)

df = pd.DataFrame({
    "timestamp":  ts,
    "user_id":    user_ids,
    "amount":     amount,
    "hour":       hour,
    "day_of_wk":  day_of_wk,
})
y = pd.Series(is_fraud, name="is_fraud")

print(f"Dataset: {len(df):,} transactions | {N_USERS} users | "
      f"timespan {df.timestamp.min().date()} -> {df.timestamp.max().date()} | "
      f"fraud rate {y.mean():.3f}\n")


# ---------------------------------------------------------------------------
# 2. NAIVE pipeline -- shuffle-split, no group awareness, scaler fit on full X
# ---------------------------------------------------------------------------

print("=" * 72)
print("PIPELINE 1: NAIVE -- 3 mistakes baked in")
print("=" * 72)

# MISTAKE 3: scale BEFORE splitting (data leakage from test stats)
scaler = StandardScaler()
X_scaled_full = pd.DataFrame(
    scaler.fit_transform(df[["amount", "hour", "day_of_wk"]]),
    columns=["amount", "hour", "day_of_wk"],
    index=df.index,
)
X_naive = X_scaled_full.copy()
X_naive["user_id"]   = df["user_id"]
X_naive["timestamp"] = df["timestamp"]

# MISTAKES 1 + 2: shuffle-split (no time order, no group awareness)
X_train, X_test, y_train, y_test = train_test_split(
    X_naive, y, test_size=0.30, random_state=42, stratify=y
)

# Drop timestamp before training (the model can't use a datetime directly)
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train.drop(columns=["timestamp"]), y_train)
proba = model.predict_proba(X_test.drop(columns=["timestamp"]))[:, 1]
pred = (proba >= 0.5).astype(int)

naive_metrics = {
    "accuracy": accuracy_score(y_test, pred),
    "f1": f1_score(y_test, pred),
    "roc_auc": roc_auc_score(y_test, proba),
}
print("Naive metrics (these will look fine but are inflated):")
for k, v in naive_metrics.items():
    print(f"  {k:<10}: {v:.4f}")
print()


# ---------------------------------------------------------------------------
# 3. dash_mlguard.check -- catch the leakage BEFORE training
# ---------------------------------------------------------------------------

print("=" * 72)
print("PIPELINE 2: DASH_MLGUARD -- run the linter first")
print("=" * 72)

report = dash_mlguard.check(
    X_train, y_train, X_test, y_test,
    time_col="timestamp",
    group_key="user_id",
)
print(report)
print()


# ---------------------------------------------------------------------------
# 4. audit_pipeline -- catch the StandardScaler-fit-on-full-data mistake
# ---------------------------------------------------------------------------

print("=" * 72)
print("PIPELINE 3: dash_mlguard.audit_pipeline -- catch preprocessing leakage")
print("=" * 72)

# Reconstruct what the developer "intended": a pipeline that scales then
# trains. dash_mlguard.audit_pipeline takes the UNFITTED pipeline + raw X, y and
# detects that scaling the full dataset before splitting is data-leaky.
candidate_pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf",   GradientBoostingClassifier(random_state=42)),
])

X_for_audit = df[["amount", "hour", "day_of_wk"]]
pipeline_report = dash_mlguard.audit_pipeline(candidate_pipe, X_for_audit, y)
print(pipeline_report)
print()


# ---------------------------------------------------------------------------
# 5. HONEST pipeline -- group-aware, time-ordered, scaler inside the split
# ---------------------------------------------------------------------------

print("=" * 72)
print("PIPELINE 4: HONEST -- fix all 3 mistakes")
print("=" * 72)

# FIX 1: chronological split.
df_sorted = df.sort_values("timestamp").reset_index(drop=True)
y_sorted = y.iloc[df_sorted.index].reset_index(drop=True)
# Hmm actually y is a Series with default range index already aligned to df,
# so after sort_values on df, we want y aligned by the original positions.
# Simpler: build df with y as a column, sort, then split.
combined = df.assign(_y=y.values).sort_values("timestamp").reset_index(drop=True)
y_sorted = combined["_y"]
df_sorted = combined.drop(columns=["_y"])

# FIX 2: also enforce group disjointness. We split chronologically first,
# then drop any test rows whose user_id appears in train.
split = int(len(df_sorted) * 0.7)
df_train = df_sorted.iloc[:split].copy()
df_test  = df_sorted.iloc[split:].copy()
y_train_h = y_sorted.iloc[:split].reset_index(drop=True)
y_test_h  = y_sorted.iloc[split:].reset_index(drop=True)

train_users = set(df_train["user_id"].unique())
keep_test = ~df_test["user_id"].isin(train_users)
df_test = df_test.loc[keep_test].reset_index(drop=True)
y_test_h = y_test_h.loc[keep_test.values].reset_index(drop=True)

print(f"After group-aware temporal split: {len(df_train):,} train rows, "
      f"{len(df_test):,} test rows ({len(df_test) / split:.1%} of train)")

# FIX 3: scaler fit ONLY on train, then transform both.
features = ["amount", "hour", "day_of_wk"]
honest_scaler = StandardScaler().fit(df_train[features])
Xh_train = pd.DataFrame(honest_scaler.transform(df_train[features]), columns=features)
Xh_test  = pd.DataFrame(honest_scaler.transform(df_test[features]), columns=features)

model_h = GradientBoostingClassifier(random_state=42)
model_h.fit(Xh_train, y_train_h)
proba_h = model_h.predict_proba(Xh_test)[:, 1]
pred_h = (proba_h >= 0.5).astype(int)

honest_metrics = {
    "accuracy": accuracy_score(y_test_h, pred_h),
    "f1": f1_score(y_test_h, pred_h),
    "roc_auc": roc_auc_score(y_test_h, proba_h),
}
print("Honest metrics (these are what production will actually deliver):")
for k, v in honest_metrics.items():
    print(f"  {k:<10}: {v:.4f}")
print()


# ---------------------------------------------------------------------------
# 6. Verdict and HTML report
# ---------------------------------------------------------------------------

print("=" * 72)
print("VERDICT")
print("=" * 72)
for k in ["accuracy", "f1", "roc_auc"]:
    delta = naive_metrics[k] - honest_metrics[k]
    print(f"  {k:<10}: naive {naive_metrics[k]:.4f} -> honest {honest_metrics[k]:.4f} "
          f"({delta:+.4f})")
print()

# Combine check() and audit_pipeline() findings into one report doc
combined = dash_mlguard.Report(findings=report.findings + pipeline_report.findings)
title = "dash_mlguard v0.3 audit -- fraud detection (temporal + grouped)"
dataset = f"Synthetic transactions, {N:,} rows / {N_USERS} users / 90 days"

# HTML (browser-friendly, embeddable)
html = combined.to_html(
    title=title,
    dataset_name=dataset,
    metrics_before=naive_metrics,
    metrics_after=honest_metrics,
)
REPORT_HTML.write_text(html, encoding="utf-8")
print(f"Wrote HTML report: {REPORT_HTML}")

# PDF (single-page audit doc -- requires `pip install dash-mlguard[pdf]`)
combined.to_pdf(
    REPORT_PDF,
    title=title,
    dataset_name=dataset,
    metrics_before=naive_metrics,
    metrics_after=honest_metrics,
)
print(f"Wrote PDF report:  {REPORT_PDF}")
